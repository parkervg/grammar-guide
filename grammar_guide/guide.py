import random
import time
from typing import Optional, Tuple, Set, Any
from collections.abc import Collection
from IPython.display import display

from transformers import AutoModelForCausalLM, AutoTokenizer
from lark import Lark
from lark import exceptions as lark_exceptions
from lark.parsers.lalr_interactive_parser import InteractiveParser
from colorama import Fore


from .model import modeling_utils as m
from .model.logits_processors import TokenHealingLogitsProcessor
from .minEarley.parser import EarleyParser
from ._logger import logger
from .typedefs import GrammarGuideOutput, Correction, StringType

DEVICE = "cpu"


class InvalidTokenState(ValueError):
    pass


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
        raise InvalidTokenState(f"Token {s} is invalid") from None


def guide(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    draft_model: Optional[Any] = None,
    seed_str: Optional[str] = None,
    lark_grammar_str: Optional[str] = None,
    lark_grammar_filepath: Optional[str] = None,
    max_grammar_corrections: int = 3,
    stop_at: Collection[str] = None,
    token_healing: Optional[bool] = True,
    top_p: float = 0.9,
    temperature: float = 0.6,
    max_new_tokens: int = 32,
):
    start = time.time()
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
    guide_model = m.Model(model=model, tokenizer=tokenizer)
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
    ].squeeze(0)
    prompt_ids_length = prompt_input_ids.shape[0]
    total_len = min(
        model.config.max_position_embeddings,
        512
        # prompt_ids_length + (max_new_tokens * max_grammar_corrections),
    )
    string_builder = m.TransformersStringBuilder(
        tokenizer, starting_ids=prompt_input_ids, write_to_html=True
    )
    tokens = m.initialize_tokens(total_len, tokenizer.pad_token_id)
    tokens[: len(prompt_input_ids)] = prompt_input_ids
    num_correction_left = max_grammar_corrections
    prefix = seed_str or ""
    ret_prediction, initial_prediction, selected_candidate = None, None, None
    corrections = []
    partial_guidance_model = draft_model + prompt
    start_pos = len(prompt_input_ids)
    if prefix:
        prefix_ids = tokenizer(prefix, return_tensors="pt", padding=True)[
            "input_ids"
        ].squeeze(0)
        tokens[
            len(prompt_input_ids) : len(prompt_input_ids) + len(prefix_ids)
        ] = prefix_ids
        start_pos += len(prefix_ids)
        string_builder.extend(prefix_ids, StringType.PROMPT)
        m.assert_valid_string_state(string_builder, tokens)
        m.assert_valid_token_state(tokens, tokenizer, start_pos)

    # Don't pass the last token of prompt here - we'll use it for generation
    past_key_values = m.forward_pass_no_sample(
        model=model,
        input_ids=tokens[: start_pos - 1].unsqueeze(0),
    )
    while num_correction_left > 0 and ret_prediction is None:
        processors = []
        if token_healing:
            healer = TokenHealingLogitsProcessor(
                guide_model,
                guide_model.model.config.vocab_size,
                tokens[tokens != tokenizer.pad_token_id],
            )
            healed_token_ids = healer.healed_token_ids
            if len(healed_token_ids) > 0:
                # Reset back, depending on length of healed tokens
                tokens[start_pos - len(healed_token_ids) :] = tokenizer.pad_token_id
                start_pos -= len(healed_token_ids)
                past_key_values = m.prune_kv_cache(
                    past_key_values=past_key_values,
                    up_to=start_pos - len(healed_token_ids),
                )
                for i in range(len(healed_token_ids)):
                    string_builder.pop(i, token_healing=True)
                string_builder.contiguous_pops = 0
                m.assert_valid_token_state(tokens, tokenizer, start_pos)
                processors.append(healer)
        tokens, start_pos, past_key_values, string_builder = m._gen_loop(
            model=model,
            tokenizer=tokenizer,
            tokens=tokens,
            past_key_values=past_key_values,
            processors=processors,
            string_builder=string_builder,
            start_pos=start_pos,
            total_len=total_len,
            stop_at_ids=stop_at_ids,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        m.assert_valid_string_state(string_builder, tokens)
        m.assert_valid_token_state(tokens, tokenizer, start_pos)
        generated_token_ids = tokens[prompt_ids_length:]
        program_prediction: str = tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )
        if validate_program(program_prediction, parser):
            ret_prediction = program_prediction
            continue
        prefix, str_candidates, re_candidates, pos_in_stream = obtain_correction_pairs(
            prediction=program_prediction,
            parser=parser,
            candidate_limit=64,
        )
        if all(len(x) == 0 for x in [str_candidates, re_candidates]):
            logger.debug("No candidates left")
            ret_prediction = prefix
            continue
        if len(str_candidates) == 1 and len(re_candidates) == 0:
            # If we only have 1 string candidate, no need to call draft_gen
            selected_candidate = str_candidates.pop()
            correction_type = "single_candidate"
        else:
            # Figure out `p`
            # I.e. - where in tokens is the index matching our `prefix`?
            # if prefix == program_prediction:
            #     pass
            # else:
            #     # Align the token id breakpoint with the stop position in the prefix
            #     # prefix = ' {\n "name": "Joseph Smith, 3"\n '
            #     # program_prediction =' {\n "name": "Joseph Smith, 3"\n "age": 32\n "occupation'
            #     prefix_ids = tokenizer(
            #         prefix, return_tensors="pt", add_special_tokens=False
            #     )["input_ids"].squeeze(0)
            #     predicted_ids = tokens[prompt_ids_length:start_pos]
            #     # TODO: the alignnment below breaks with token healing, since it changes
            #     #   the tokens in our runnng `tokens` array
            #     for p in range(max([predicted_ids.shape[-1], prefix_ids.shape[-1]])):
            #         if p >= prefix_ids.shape[-1] or p >= predicted_ids.shape[-1]:
            #             break
            #         prefix_id = prefix_ids[p]
            #         predicted_id = predicted_ids[p]
            #         if prefix_id != predicted_id:
            #             break
            #     assert tokenizer.decode(prefix_ids[:p]) == tokenizer.decode(
            #         predicted_ids[:p]
            #     )
            # # Cut off our kv cache up to the valid grammar prefix
            # past_key_values = m.prune_kv_cache(
            #     past_key_values=past_key_values,
            #     up_to=prompt_ids_length + p,
            # )
            # # Use these new past_key_values to select a candidate
            # _tokens = tokens[:]
            # _tokens[prompt_ids_length + p:] = tokenizer.pad_token_id
            # m.assert_valid_token_state(_tokens, tokenizer, p)
            # import re
            # processors = []
            # processors.append(
            #     RegexLogitsProcessor(
            #         pattern="|".join([re.escape(s) for s in str_candidates] + re_candidates),
            #         llm=guide_model,
            #         vocab_size=model.config.vocab_size,
            #         is_greedy=temperature==0,
            #         prefix_length=len(prefix),
            #         eos_token_id=tokenizer.eos_token_id
            #     )
            # )
            # _old_start_pos = p
            # tokens, start_pos, past_key_values = m._gen_loop(
            #     model=model,
            #     tokenizer=tokenizer,
            #     tokens=tokens,
            #     past_key_values=past_key_values,
            #     processors=processors,
            #     start_pos=p,
            #     total_len=total_len,
            #     stop_at_ids=stop_at_ids,
            #     max_new_tokens=max_new_tokens,
            #     top_p=top_p,
            #     temperature=temperature,
            # )
            # selected_candidate_ids = tokens[_old_start_pos:start_pos]
            # selected_candidate = tokenizer.decode(selected_candidate_ids)
            # selected_candidate = random.choice(str_candidates)
            if False:
                import guidance
                import re

                make_regex_pred = lambda pattern: (
                    partial_guidance_model
                    + prefix
                    + guidance.capture(
                        guidance.with_temperature(guidance.regex(pattern=pattern), 0.0),
                        "res",
                    )
                )["res"]

                selected_candidate = make_regex_pred(
                    "|".join([re.escape(s) for s in str_candidates] + re_candidates)
                )
            else:
                selected_candidate = random.choice(str_candidates)
            correction_type = "draft_gen"
        # Now, try to use our selected candidate in a few ways
        # 1) Insert our selection into the index where the error occurred, and add left/right context
        #   Example: SELECT a b FROM table -> SELECT a, b FROM table
        inserted_candidate_prediction = (
            (prefix + selected_candidate + program_prediction[pos_in_stream:])
            if pos_in_stream != -1
            else None
        )
        partial_program_prediction = prefix + selected_candidate
        if inserted_candidate_prediction is not None and validate_program(
            inserted_candidate_prediction, parser
        ):
            ret_prediction = inserted_candidate_prediction
            correction_type += "_middle_fill"
            corrections.append(
                Correction(
                    original_pred=program_prediction,
                    corrected_pred=ret_prediction,
                    selected_candidate=selected_candidate,
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
                    selected_candidate=selected_candidate,
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
                    selected_candidate=selected_candidate,
                    type=correction_type,
                )
            )
            selected_candidate_ids = tokenizer(
                selected_candidate, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            if (
                tokens.shape[-1]
                - tokens[tokens != tokenizer.pad_token_id].count_nonzero()
                < selected_candidate_ids.shape[-1]
            ):
                logger.debug(Fore.RED + "Exceeded max token array length" + Fore.RESET)
                ret_prediction = prefix
                continue
            if prefix == program_prediction:
                # Simple case: insert our candidate ids into the end of the running token array
                tokens[
                    start_pos : start_pos + selected_candidate_ids.shape[-1]
                ] = selected_candidate_ids
                # Forward pass new candidate tokens - only if length > 1
                if selected_candidate_ids.shape[-1] > 1:
                    past_key_values = m.forward_pass_no_sample(
                        model=model,
                        input_ids=selected_candidate_ids[:, :-1],
                        past_key_values=past_key_values,
                    )
                start_pos += selected_candidate_ids.shape[-1]
                string_builder.extend(
                    selected_candidate_ids.squeeze(0), StringType.CANDIDATE_SELECTION
                )
                m.assert_valid_string_state(string_builder, tokens)
                m.assert_valid_token_state(tokens, tokenizer, start_pos)
            else:
                # Align the token id breakpoint with the stop position in the prefix
                # prefix = ' {\n "name": "Joseph Smith, 3"\n '
                # program_prediction =' {\n "name": "Joseph Smith, 3"\n "age": 32\n "occupation'
                prefix_ids = tokenizer(
                    prefix, return_tensors="pt", add_special_tokens=False
                )["input_ids"].squeeze(0)
                predicted_ids = tokens[prompt_ids_length:start_pos]
                # TODO: the alignnment below breaks with token healing, since it changes
                #   the tokens in our runnng `tokens` array
                for p in range(max([predicted_ids.shape[-1], prefix_ids.shape[-1]])):
                    if p >= prefix_ids.shape[-1] or p >= predicted_ids.shape[-1]:
                        break
                    prefix_id = prefix_ids[p]
                    predicted_id = predicted_ids[p]
                    if prefix_id != predicted_id:
                        break
                assert tokenizer.decode(prefix_ids[:p]) == tokenizer.decode(
                    predicted_ids[:p]
                )
                # Cut off our kv cache up to the valid grammar prefix
                past_key_values = m.prune_kv_cache(
                    past_key_values=past_key_values,
                    up_to=prompt_ids_length + p,
                )
                # Clear tokens after valid_completion_id index
                tokens[prompt_ids_length + p :] = tokenizer.pad_token_id
                for i in range(predicted_ids.shape[-1] - p):
                    string_builder.pop(i)
                m.assert_valid_string_state(string_builder, tokens)
                diff = len(prefix_ids) - p
                if diff > 0:
                    tokens[
                        prompt_ids_length + p : prompt_ids_length + p + diff
                    ] = prefix_ids[-diff:]
                    past_key_values = m.forward_pass_no_sample(
                        model=model,
                        input_ids=prefix_ids[-diff:].reshape(1, -1),
                        past_key_values=past_key_values,
                    )
                    string_builder.extend(prefix_ids[-diff:], StringType.GENERATION)
                    m.assert_valid_string_state(string_builder, tokens)
                if (
                    selected_candidate_ids.shape[-1]
                    > tokens[prompt_ids_length + p :].shape[0]
                ):
                    raise ValueError(
                        f"Trying to insert candidate {selected_candidate}, but it is too long!"
                    )
                # Update tokens with our selected candidate id
                tokens[
                    prompt_ids_length
                    + p
                    + diff : prompt_ids_length
                    + p
                    + diff
                    + selected_candidate_ids.shape[-1]
                ] = selected_candidate_ids
                # Forward pass new candidate tokens - only if length > 1
                if selected_candidate_ids.shape[-1] > 1:
                    past_key_values = m.forward_pass_no_sample(
                        model=model,
                        input_ids=selected_candidate_ids[:, :-1],
                        past_key_values=past_key_values,
                    )
                start_pos = (
                    prompt_ids_length + p + diff + selected_candidate_ids.shape[-1]
                )
                string_builder.extend(
                    selected_candidate_ids.squeeze(0), StringType.CANDIDATE_SELECTION
                )
                m.assert_valid_string_state(string_builder, tokens)
                m.assert_valid_token_state(tokens, tokenizer, start_pos)
        num_correction_left -= 1
        logger.debug(
            Fore.YELLOW + f"Made a {corrections[-1].type} correction..." + Fore.RESET
        )
        if tokens[tokens != tokenizer.pad_token_id].count_nonzero() == 0:
            logger.debug(Fore.RED + "Exceeded max token array length" + Fore.RESET)
            ret_prediction = prefix
            continue
    if ret_prediction is None:
        logger.debug(
            Fore.RED
            + f"Cannot find a valid prediction after {max_grammar_corrections} retries"
            + Fore.RESET
        )
        ret_prediction = prefix
    display(
        {
            "text/html": "<div style='margin: 0px; padding: 0px; font-family: ColfaxAI, Arial; font-size: 20px;'"
            + "".join(string_builder.html)
            + "</div>"
        },
        display_id=0,
        raw=True,
        include=["text/html"],
    )
    return GrammarGuideOutput(
        response=ret_prediction,
        num_grammar_corrections=len(corrections),
        correction_log=corrections,
        process_time_seconds=time.time() - start,
    )
