from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedTokenizer,
    LogitsProcessor,
)
import re
import pygtrie

from ..typedefs import StringType
from ..utils import is_interactive

GREEN_BG_COLOR_OPEN = "<span style='background-color: rgba(0, 165, 0, 0.25);'>"
BLUE_BG_COLOR_OPEN = "<span style='background-color: rgba(0, 0, 165, 0.25);'>"
SPAN_CLOSE = "</span>"


def set_seed(seed):
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformersStringBuilder:
    """This deals with the complexity of building up a string from tokens bit by bit."""

    STRING_TYPE_TO_FMT = {
        StringType.PROMPT: lambda s: f"<text style=color:black>{s}</text>",
        StringType.GENERATION: lambda s: f"{GREEN_BG_COLOR_OPEN}<text style=color:black>{s}</text>{SPAN_CLOSE}",
        StringType.DELETION: lambda s: f"<text style=color:red>{s}</text>",
        StringType.CANDIDATE_SELECTION: lambda s: f"{BLUE_BG_COLOR_OPEN}<text style=color:blue>{s}</text>{SPAN_CLOSE}",
    }

    def __init__(
        self,
        tokenizer,
        starting_ids: Optional[torch.Tensor] = None,
        log_changes: bool = False,
        write_to_html: bool = False,
    ):
        self.tokenizer = tokenizer
        self.token_strings = []
        self._joint_string = ""
        self.write_to_html = write_to_html
        self.log_changes = log_changes
        self.html = []
        self.contiguous_pops = 0
        if starting_ids is not None:
            self.extend(starting_ids, StringType.PROMPT)

    def extend(self, new_ids, string_type: Optional[StringType] = None):
        new_token_strings = self.tokenizer.convert_ids_to_tokens(new_ids)
        self.token_strings.extend(new_token_strings)
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = new_str[len(self._joint_string) :]
        self._joint_string = new_str
        if self.write_to_html:
            assert string_type is not None
            self.html += [
                self.STRING_TYPE_TO_FMT[string_type](self.tokenizer.decode(i))
                for i in new_ids
            ]
        if self.log_changes and not is_interactive():
            clear_and_print(self._joint_string)
        return diff_str

    def pop(self, i: Optional[int] = None, token_healing: bool = False):
        """Remove the last token from the string and return text it removed."""
        self.token_strings.pop()
        new_str = self.tokenizer.convert_tokens_to_string(self.token_strings)
        diff_str = self._joint_string[len(new_str) :]
        self._joint_string = new_str
        if self.write_to_html:
            assert i is not None
            i += 1
            # Change color to red
            self.html[-i] = re.sub(
                "(style=color:)(green|black|blue)",
                r"\1orange" if token_healing else r"\1red",
                self.html[-i],
            )
            # Remove green background
            self.html[-i] = re.sub(
                f"({re.escape(GREEN_BG_COLOR_OPEN)})(.*)({re.escape(SPAN_CLOSE)})",
                r"\2",
                self.html[-i],
            )
        if self.log_changes and not is_interactive():
            clear_and_print(self._joint_string)
        return diff_str

    def __str__(self):
        return self._joint_string

    def __len__(self):
        return len(self._joint_string)


class Model:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._token_prefix_map = self._build_token_prefix_map()

    def id_to_token(self, id):
        return self.tokenizer.convert_ids_to_tokens([id])[0]

    def token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    def new_string_builder(self, starting_ids=None):
        return TransformersStringBuilder(self.tokenizer, starting_ids)

    def _build_token_prefix_map(self):
        token_map = pygtrie.CharTrie()
        for i in range(self.tokenizer.vocab_size):
            s = self.id_to_token(i)
            if s in token_map:
                token_map[s].append(i)
            else:
                token_map[s] = [i]
        return token_map

    def prefix_matches(self, prefix: str):
        """Returns the list of tokens that match the given prefix."""
        return [v for arr in self._token_prefix_map.values(prefix=prefix) for v in arr]


def clear_and_print(s: str):
    print("\n" * 1000)
    print(s)


def assert_valid_token_state(
    tokens: torch.Tensor, tokenizer: PreTrainedTokenizer, start_pos: int
):
    if (tokens[start_pos:] != tokenizer.pad_token_id).count_nonzero() != 0 or (
        tokens[:start_pos] == tokenizer.pad_token_id
    ).count_nonzero() != 0:
        raise ValueError


def assert_valid_string_state(
    string_builder: TransformersStringBuilder, tokens: torch.Tensor
):
    return True
    assert string_builder._joint_string == string_builder.tokenizer.decode(
        tokens, skip_special_tokens=True
    )


def assert_valid_kv_cache_state(past_key_values: DynamicCache, start_pos: int):
    assert past_key_values.key_cache[0].shape[-2] == (start_pos - 1)


def initialize_tokens(total_len: int, pad_token_id: int, device: torch.device):
    return torch.full(
        (total_len,),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )


def sample_top_p(probs, p):
    # https://github.com/meta-llama/llama3/blob/main/llama/generation.py
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def contains_stop_sequence(
    tokens: torch.Tensor, stop_at_ids: torch.Tensor, pad_token_id: int
) -> bool:
    """
    Looks to be on-par with the huggingface StopStringCriteria?
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py#L141

    Given a `token` array of shape (max_seq_len,), returns `True` if `tokens` ends in any of the
    row-wise entries of `stop_at_ids` of shape (batch_size, max_seq_len), and `False` if not.

    Example:
        tokens = torch.Tensor(
            [1, 2, 3, 0, 0, 0]
        )
        stop_at_ids = torch.Tensor(
            [
                [2, 3, 0, 0],
                [5, 6, 7, 8]
            ]
        )
        pad_token_id = 0
        # The sequence ends in [2, 3], which is a pattern in `stop_at_ids`
        assert contains_stop_sequence(tokens, stop_at_ids, pad_token_id) == True
    """
    # Length of longest stop token
    T = stop_at_ids.shape[-1]
    non_pad_tokens = tokens[tokens != pad_token_id]
    s = F.pad(
        non_pad_tokens[-T:],
        (0, max(T - non_pad_tokens.shape[0], 0)),
        mode="constant",
        value=pad_token_id,
    ).expand(T, T)
    arange1 = torch.arange(T, device=tokens.device).view((T, 1)).repeat((1, T))
    arange2 = ((arange1 + torch.arange(T, device=tokens.device)) % T).T
    s = torch.gather(s, -1, arange2.T)
    mask = torch.fliplr(torch.ones_like(s, device=tokens.device).tril().T)
    s = torch.unique(s * mask, dim=0)
    stacked = torch.cat((s, stop_at_ids), dim=0)
    return torch.unique(stacked, dim=0).shape != stacked.shape


def prune_kv_cache(past_key_values: DynamicCache, up_to: int) -> DynamicCache:
    """Selects a subset of the key-value cache to use in a new generation.

    `past_key_values` has two attributes we care about, `key_cache` and `value_cache`.
    Both are tuples of len `num_hidden_layers`, where each entry is a tensor
    with the shape (batch_size, num_key_value_heads, seq_len, ??)
    # TODO: the last dimension is 64, not sure where this comes from)
    """
    # print(f"Previous kv cache size: {past_key_values.key_cache[0].shape[-2]}")
    # print(f"New size: {up_to}")
    _past_key_values = DynamicCache()
    _past_key_values.key_cache = [t[..., :up_to, :] for t in past_key_values.key_cache]
    _past_key_values.value_cache = [
        t[..., :up_to, :] for t in past_key_values.value_cache
    ]
    return _past_key_values


def forward_pass_no_sample(
    model: AutoModelForCausalLM,
    input_ids,
    past_key_values: Optional[DynamicCache] = None,
):
    """Used to pass bits of text where we don't care about the logits"""
    # tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    # print("Forward pass:")
    # print(repr(tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)))
    return model(
        input_ids=input_ids,
        past_key_values=past_key_values or DynamicCache(),
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
    ).past_key_values


# @profile
@torch.inference_mode
def _gen_loop(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokens: torch.Tensor,
    past_key_values: DynamicCache,
    start_pos: int,
    total_len: int,
    stop_at_ids: torch.Tensor,
    string_builder: Optional[TransformersStringBuilder] = None,
    processors: Optional[List[LogitsProcessor]] = None,
    max_new_tokens: Optional[int] = 32,
    top_p: float = 0.9,
    temperature: float = 0.6,
):
    if processors is None:
        processors = []
    prev_pos = start_pos - 1
    eos_reached = torch.tensor([False], device=model.device)
    # p = parser.parse_interactive() if parser else None
    new_token_pos = 0
    for new_token_pos, cur_pos in enumerate(range(start_pos, total_len)):
        # if p:
        #     # If under our grammar there's only 1 choice available, skip ahead
        #     if len(p.accepts()) == 1:
        #         singleton_accepted: TerminalDef = parser.get_terminal(p.accepts().pop())
        #         if isinstance(singleton_accepted.pattern, PatternStr):
        #             forced_str = singleton_accepted.pattern.value
        #             print("LARK MATCH")
        #             print(forced_str)
        #             force_new_token_ids = tokenizer(forced_str, return_tensors="pt", padding=False, add_special_tokens=False)['input_ids']
        #             past_key_values = forward_pass_no_sample(
        #                 force_new_token_ids,
        #                 past_key_values=past_key_values
        #             )
        #             for i, tok_id in enumerate(force_new_token_ids):
        #                 tokens[cur_pos+i] = tok_id
        #             feed_str_to_parser(parser, p, forced_str)
        # Skip, if we've already applied a forced token
        # if tokens[cur_pos] != tokenizer.pad_token_id:
        #     continue
        # https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaModel.forward.past_key_values
        # If past_key_values are used, the user can optionally input only the last input_ids
        # (those that don’t have their past key value states given to this model)
        # of shape (batch_size, 1) instead of all input_ids of shape (batch_size, sequence_length).
        model_output = model(
            input_ids=tokens[prev_pos:cur_pos].unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        # print(f"Received token:")
        # print(repr(tokenizer.decode(tokens[prev_pos:cur_pos])))
        # Each entry in cache is tuple of (key, value)
        past_key_values = model_output.past_key_values

        if tokens[cur_pos] != tokenizer.pad_token_id:
            raise ValueError
        # len(cache) == model.config.num_hidden_layers
        logits = model_output.logits.squeeze(0)
        for p in processors:
            logits = p(tokens[:cur_pos].unsqueeze(0), logits)
        last_logits = logits[-1, :]
        if temperature > 0:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(last_logits, dim=-1)
        tokens[cur_pos] = next_token
        if string_builder:
            string_builder.extend([next_token], StringType.GENERATION)
        # if p:
        #     try:
        #         feed_str_to_parser(parser, p, tokenizer.decode(next_token))
        #     except InvalidTokenState as error:
        #         # TODO: either backtrack and try to parse with constraints,
        #         # Or continue and see if the model can recover
        #         # print(error)
        #         pass
        eos_reached |= (new_token_pos >= max_new_tokens) | (
            next_token == tokenizer.eos_token_id
        )
        if not eos_reached and stop_at_ids is not None:
            eos_reached |= contains_stop_sequence(
                tokens=tokens[start_pos:],
                stop_at_ids=stop_at_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        prev_pos = cur_pos
        if eos_reached or cur_pos + 1 == tokens.shape[0]:
            break
    return tokens, cur_pos + 1, past_key_values, string_builder
