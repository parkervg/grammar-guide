import random
import time
from typing import Optional, List, Tuple, Set, Any
from collections.abc import Collection

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from lark import Lark
from lark import exceptions as lark_exceptions
from lark.parsers.lalr_interactive_parser import InteractiveParser
from colorama import Fore

from .minEarley.parser import EarleyParser
from ._logger import logger
from .typedefs import GrammarGuideOutput, Correction


class InvalidTokenState(ValueError):
    pass


torch.manual_seed(42)
device = torch.device("cpu")


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


def forward_pass_no_sample(model: AutoModelForCausalLM, input_ids, past_key_values: Optional[DynamicCache] = None):
    """Used to pass prompts (i.e. bits of text where we don't care about the logits)"""
    return model.forward(
        input_ids=input_ids,
        past_key_values=past_key_values or DynamicCache(),
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
    ).past_key_values


def prune_kv_cache(past_key_values: DynamicCache, up_to: int) -> DynamicCache:
    """Selects a subset of the key-value cache to use in a new generation.

    `past_key_values` has two attributes we care about, `key_cache` and `value_cache`.
    Both are tuples of len `num_hidden_layers`, where each entry is a tensor
    with the shape (batch_size, num_key_value_heads, seq_len, ??)
    # TODO: the last dimension is 64, not sure where this comes from)
    """
    _past_key_values = DynamicCache()
    _past_key_values.key_cache = [t[..., :up_to, :] for t in past_key_values.key_cache]
    _past_key_values.value_cache = [
        t[..., :up_to, :] for t in past_key_values.value_cache
    ]
    _past_key_values._seen_tokens = _past_key_values.key_cache[0].shape[2]
    return _past_key_values


def validate_program(prediction: str, parser: EarleyParser) -> bool:
    try:
        parser.parse(prediction)
        return True
    except Exception:
        # logger.debug(Fore.LIGHTCYAN_EX + prediction + Fore.RESET)
        # logger.debug(f"Error: {str(runtime_e)}")
        return False


def obtain_correction_pairs(
    prediction: str,
    parser: EarleyParser,
    candidate_limit: int,
) -> Tuple[str, Set[str], int]:
    """
    Returns a list of candidates in the form of (prefix, candidates, error_position_index).
    """
    try:
        parser.parse(prediction)
        raise ValueError(
            "When calling obtain_correction_pairs, the passed prediction should already be assumed to fail the grammar constraints"
        )
    except Exception as runtime_e:
        return parser.handle_error(
            runtime_e,
            candidate_limit=candidate_limit,
        )


def feed_str_to_parser(parser: Lark, p: InteractiveParser, s: str):
    try:
        for t in parser.lex(s):
            if t.type in p.accepts():
                # I guess .feed_token() calls exhaust_lexer() behind the scenes?
                p.feed_token(t)
            else:
                raise InvalidTokenState(f"Token {s} is not in accept states")
    except lark_exceptions.UnexpectedCharacters:
        raise InvalidTokenState(f"Token {s} is invalid")


@torch.inference_mode
def _gen_loop(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokens: torch.tensor,
    past_key_values: DynamicCache,
    start_pos: int,
    total_len: int,
    stop_at_ids: torch.tensor,
    choices: Optional[List[str]] = None,
    max_new_tokens: Optional[int] = 32,
    top_p: float = 0.9,
    temperature: float = 0.6,
):
    prev_pos = start_pos - 1
    eos_reached = torch.tensor([False], device=device)
    # p = parser.parse_interactive() if parser else None
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
        model_output = model.forward(
            input_ids=tokens[prev_pos:cur_pos].unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        # len(cache) == model.config.num_hidden_layers
        # Each entry in cache is tuple of (key, value)
        past_key_values = model_output.past_key_values
        logits = model_output.logits.squeeze(0)
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
    return tokens[start_pos:], new_token_pos, past_key_values


def guide(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    draft_model: Optional[Any],
    seed_str: Optional[str] = None,
    lark_grammar_str: Optional[str] = None,
    lark_grammar_filepath: Optional[str] = None,
    max_grammar_corrections: int = 3,
    stop_at: Collection[str] = None,
    top_p: float = 0.9,
    temperature: float = 0.6,
    max_new_tokens: int = 32,
):
    if all([x is None for x in {lark_grammar_str, lark_grammar_filepath}]):
        raise ValueError(
            "One of `cfg_grammar_str`, `cfg_grammar_filepath` must be specified!"
        )
    elif lark_grammar_filepath:
        lark_grammar_str = open(lark_grammar_filepath).read()
    parser: EarleyParser = EarleyParser(
        grammar=lark_grammar_str,
        start="start",
        keep_all_tokens=True,
    )
    # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_at_ids = None
    if stop_at:
        tokenizer.padding_side = "right"
        stop_at_tokenizer_out = tokenizer(
            stop_at, return_tensors="pt", padding=True, add_special_tokens=False
        )
        stop_at_ids = stop_at_tokenizer_out["input_ids"]
    prompt_input_ids = tokenizer(prompt, return_tensors="pt", padding=True)[
        "input_ids"
    ].squeeze()
    total_len = min(
        model.config.max_position_embeddings,
        prompt_input_ids.shape[0] + (max_new_tokens * max_grammar_corrections),
    )
    tokens = torch.full(
        (total_len,),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    tokens[: len(prompt_input_ids)] = prompt_input_ids
    num_correction_left = max_grammar_corrections
    partial_program_prediction = seed_str or ""
    ret_prediction, initial_prediction = None, None
    corrections = []
    start = time.time()
    past_key_values = forward_pass_no_sample(
        model=model,
        input_ids=tokens[: len(prompt_input_ids)].unsqueeze(0)
    )
    start_pos = len(prompt_input_ids)
    while num_correction_left > 0 and ret_prediction is None:
        tokens, new_token_pos, past_key_values = _gen_loop(
            model=model,
            tokenizer=tokenizer,
            tokens=tokens,
            past_key_values=past_key_values,
            start_pos=start_pos,
            total_len=total_len,
            stop_at_ids=stop_at_ids,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        residual_program_prediction: str = tokenizer.decode(
            tokens, skip_special_tokens=True
        )
        # This is the representation
        program_prediction = partial_program_prediction + residual_program_prediction
        if validate_program(program_prediction, parser):
            ret_prediction = program_prediction
            continue
        prefix, candidates, has_re, pos_in_stream = obtain_correction_pairs(
            prediction=program_prediction,
            parser=parser,
            candidate_limit=64,
        )
        if len(candidates) == 0:
            logger.debug(
                Fore.LIGHTMAGENTA_EX + "No correction pairs found" + Fore.RESET
            )
            return prefix
        elif len(candidates) == 1 and not has_re:
            # If we only have 1 candidate, no need to call draft_gen
            selected_candidate = candidates.pop()
            correction_type = "single_candidate"
        else:
            if draft_model:
                selected_candidate = (
                    draft_model
                    + prompt
                    + program_prediction
                    + guidance.capture(
                        guidance.with_temperature(
                            guidance.select(options=candidates), "res", 0.0
                        )
                    )
                )
                # Generate the continuation candidate with the highest probability
                if False:
                    # TODO: implement own 'select' logic so we can reuse kv cache
                    pass
            else:
                selected_candidate = random.choice(candidates)
            correction_type = "draft_gen"

        # Now, try to use our selected candidate in a few ways
        # 1) Insert our selection into the index where the error occurred, and add left/right context
        #   Example: SELECT a b FROM table -> SELECT a, b FROM table
        inserted_candidate = (
            prefix + selected_candidate + program_prediction[pos_in_stream:]
        )
        partial_program_prediction = prefix + selected_candidate
        if validate_program(inserted_candidate, parser):
            ret_prediction = inserted_candidate
            correction_type += "_middle_fill"
            corrections.append(
                Correction(
                    original_pred=program_prediction,
                    corrected_pred=ret_prediction,
                    type=correction_type,
                )
            )
            continue
        # 2) Just keep up to the prefix + selected_candidate
        # For example, if we just forgot a semicolon at the end of a JavaScript line
        elif validate_program(partial_program_prediction, parser):
            ret_prediction = partial_program_prediction
            corrections.append(
                Correction(
                    original_pred=program_prediction,
                    corrected_pred=ret_prediction,
                    type=correction_type,
                )
            )
            continue
        else:
            # 3) If rest of our query is also broken, we just keep up to the prefix + candidate
            # and re-generate a continuation again
            corrections.append(
                Correction(
                    original_pred=program_prediction,
                    corrected_pred=partial_program_prediction,
                    type=correction_type,
                )
            )
            # assert len(tokenizer(prompt + program_prediction)['input_ids']) == past_key_values.key_cache[0].shape[2]
            valid_completion_ids = tokenizer(
                program_prediction[:pos_in_stream], add_special_tokens=False
            )["input_ids"]
            # Cut off our kv cache up to the valid grammar prefix
            past_key_values = prune_kv_cache(
                past_key_values=past_key_values,
                up_to=len(prompt_input_ids) + len(valid_completion_ids),
            )
            # Clear tokens after valid_completion_id index
            tokens[len(valid_completion_ids) :] = tokenizer.pad_token_id
            selected_candidate_ids = tokenizer(
                selected_candidate + " ", return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            if (
                selected_candidate_ids.shape[-1]
                > tokens[len(valid_completion_ids) :].shape[0]
            ):
                raise ValueError(
                    f"Trying to insert candidate {selected_candidate}, but it is too long!"
                )
            tokens[
                len(valid_completion_ids) : len(valid_completion_ids)
                + selected_candidate_ids.shape[-1]
            ] = selected_candidate_ids
            # Forward pass with new candidate
            past_key_values = forward_pass_no_sample(
                model=model,
                input_ids=selected_candidate_ids,
                past_key_values=past_key_values
            )
            # Now, setup generation from those new candidate_ids we added
            start_pos = len(valid_completion_ids) + len(selected_candidate_ids)
        num_correction_left -= 1
        logger.debug(Fore.YELLOW + f"Made a {corrections[-1].type} correction...")
    if ret_prediction is None:
        logger.debug(
            Fore.RED
            + f"Cannot find a valid prediction after {max_grammar_corrections} retries"
            + Fore.RESET
        )
        ret_prediction = prefix
    logger.debug(Fore.GREEN + ret_prediction + Fore.RESET)
    return GrammarGuideOutput(
        response=ret_prediction,
        num_grammar_corrections=len(corrections),
        correction_log=corrections,
        process_time_seconds=time.time() - start,
    )
