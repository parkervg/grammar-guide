from typing import Optional, List, Dict
from attr import attrs, attrib
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedTokenizer,
    PreTrainedModel,
    LogitsProcessor,
)
import pygtrie

torch.manual_seed(42)
device = torch.device("cpu")


@attrs
class Model:
    model: PreTrainedModel = attrib()
    tokenizer: PreTrainedTokenizer = attrib()

    _token_prefix_map: Dict = attrib(init=False)

    def __attrs_post_init__(self):
        self._token_prefix_map = self._build_token_prefix_map()

    def id_to_token(self, id):
        return self.tokenizer.convert_ids_to_tokens([id])[0]

    def token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    #
    # def id_to_token(self, id):
    #     return self.tokenizer.decode(id)
    #
    # def token_to_id(self, token):
    #     return self.tokenizer(token, add_special_tokens=False)["input_ids"][0]

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


def initialize_tokens(total_len, pad_token_id):
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
    tokens: torch.tensor, stop_at_ids: torch.tensor, pad_token_id: int
) -> bool:
    """
    Given a `token` array of shape (max_seq_len,), returns `True` if `tokens` ends in any of the
    row-wise entries of `stop_at_ids` of shape (batch_size, max_seq_len), and `False` if not.

    Example:
        tokens = torch.tensor(
            [1, 2, 3, 0, 0, 0]
        )
        stop_at_ids = torch.tensor(
            [
                [2, 3, 0, 0],
                [5, 6, 7, 8]
            ]
        )
        pad_token_id = 0
        # The sequence ends in [2, 3], which is a pattern in `stop_at_ids`
        assert contains_stop_sequence(tokens, stop_at_ids, pad_token_id) == True
    TODO:
        - handle case when tokens may begin with stop_at prefix
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
    arange1 = torch.arange(T).view((T, 1)).repeat((1, T))
    arange2 = ((arange1 + torch.arange(T)) % T).T
    s = torch.gather(s, -1, arange2.T)
    mask = torch.fliplr(torch.ones_like(s).tril().T)
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
    # print("Current key cache shape: {}".format(past_key_values.key_cache[0].shape[-2]))
    # print("New shape: {}".format(up_to))
    _past_key_values = DynamicCache()
    _past_key_values.key_cache = [t[..., :up_to, :] for t in past_key_values.key_cache]
    _past_key_values.value_cache = [
        t[..., :up_to, :] for t in past_key_values.value_cache
    ]
    # _past_key_values._seen_tokens = _past_key_values.key_cache[0].shape[2]
    return _past_key_values


def forward_pass_no_sample(
    model: AutoModelForCausalLM,
    input_ids,
    past_key_values: Optional[DynamicCache] = None,
):
    """Used to pass prompts (i.e. bits of text where we don't care about the logits)"""
    return model(
        input_ids=input_ids,
        past_key_values=past_key_values or DynamicCache(),
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
    ).past_key_values


@torch.inference_mode
def _gen_loop(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokens: torch.tensor,
    past_key_values: DynamicCache,
    start_pos: int,
    total_len: int,
    stop_at_ids: torch.tensor,
    processors: Optional[List[LogitsProcessor]] = None,
    max_new_tokens: Optional[int] = 32,
    top_p: float = 0.9,
    temperature: float = 0.6,
):
    if processors is None:
        processors = []
    prev_pos = start_pos - 1
    eos_reached = torch.tensor([False], device=device)
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
        # print()
        # print("Received token:")
        # print(tokenizer.decode(tokens[prev_pos:cur_pos]))
        # print()
        # Each entry in cache is tuple of (key, value)
        past_key_values = model_output.past_key_values
        # Don't update tokens if we've already put a token here
        if tokens[cur_pos] != tokenizer.pad_token_id:
            prev_pos = cur_pos
            continue
        # len(cache) == model.config.num_hidden_layers
        logits = model_output.logits.squeeze(0)
        for p in processors:
            logits = p(tokens[prev_pos:cur_pos].unsqueeze(0), logits)
        last_logits = logits[-1, :]
        if temperature > 0:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(last_logits, dim=-1)
        tokens[cur_pos] = next_token
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
    return tokens, new_token_pos, past_key_values
