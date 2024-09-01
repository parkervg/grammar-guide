from typing import Optional, Callable, Union
from collections.abc import Collection
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from colorama import Fore
from IPython.display import display
import logging
import guidance

from .minEarley.parser import EarleyParser
from .typedefs import GrammarGuideOutput, StringType
from .model import modeling_utils as m
from .model.logits_processors import TokenHealingLogitsProcessor
from .utils import is_interactive, prepare_initial_prefix, handle_program_prediction
from ._logger import logger


def load_parser(lark_grammar_str: Optional[str]) -> EarleyParser:
    return EarleyParser(
        grammar=lark_grammar_str,
        start="start",
        keep_all_tokens=True,
    )


def guide(
    draft_model: Union[AutoModelForCausalLM, Callable[[str], str]],
    parser: EarleyParser,
    prompt: str,
    target_model: guidance.models.Model,
    tokenizer: Optional[AutoTokenizer] = None,
    seed_str: Optional[str] = None,
    max_grammar_corrections: int = 3,
    stop_at: Optional[Collection[str]] = None,
    token_healing: Optional[bool] = True,
    top_p: float = 0.9,
    temperature: float = 0.0,
    token_lookahead: int = 20,
    save_html: bool = True,
    verbose: bool = True,
    debug: bool = False,
) -> GrammarGuideOutput:
    """
    Guides the generation process using a transformer model with grammar-based corrections.

    This function implements a guided text generation process using a transformer model,
    incorporating grammar-based corrections and other advanced features like token healing.

    Args:
        draft_model (Union[AutoModelForCausalLM, Callable[[str], str]]): A transformer model or callable to use for text generation.
        tokenizer (AutoTokenizer): Transformers only, the tokenizer associated with the model.
        parser (EarleyParser): The parser used for grammar checking.
        prompt (str): The initial prompt for text generation.
        target_model (guidance.models.Model): The guidance model to use for constrained grammar correction
            https://github.com/guidance-ai/guidance
        seed_str (Optional[str]): An optional seed string to start the generation.
        max_grammar_corrections (int): Maximum number of grammar corrections to attempt.
        stop_at (Collection[str]): Collection of strings to stop generation at.
        token_healing (Optional[bool]): Transformers only, whether to use token healing during generation.
        top_p (float): Transformers only, the cumulative probability for top-p sampling.
        temperature (float): Transformers only, the temperature for controlling randomness in generation.
        token_lookahead (int): Maximum number of new tokens to generate using draft model.
            Essentially the $K$ parameter in speculative decoding.
        save_html (bool): Whether to save the generation process as HTML.
        verbose (bool): Whether to print verbose output.
        debug (bool): Whether to run in debug mode with additional checks.

    Returns:
        GrammarGuideOutput: An object containing the generated text, correction logs,
                            processing time, and optionally HTML output.

    Raises:
        ValueError: If the selected candidate is too long to insert.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
    if hasattr(draft_model, "config"):
        return _transformers_guide(
            draft_model=draft_model,
            tokenizer=tokenizer,
            parser=parser,
            prompt=prompt,
            target_model=target_model,
            seed_str=seed_str,
            max_grammar_corrections=max_grammar_corrections,
            stop_at=stop_at,
            token_healing=token_healing,
            top_p=top_p,
            temperature=temperature,
            token_lookahead=token_lookahead,
            save_html=save_html,
            verbose=verbose,
            debug=debug,
        )
    assert isinstance(draft_model, Callable)
    return _generic_guide(
        draft_model=draft_model,
        parser=parser,
        prompt=prompt,
        target_model=target_model,
        seed_str=seed_str,
        max_grammar_corrections=max_grammar_corrections,
        token_lookahead=token_lookahead,
        verbose=verbose,
    )


# @profile
def _transformers_guide(
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    parser: EarleyParser,
    prompt: str,
    target_model: guidance.models.Model,
    seed_str: Optional[str],
    max_grammar_corrections: int,
    stop_at: Collection[str],
    token_healing: Optional[bool],
    top_p: float,
    temperature: float,
    token_lookahead: int,
    save_html: bool,
    verbose: bool,
    debug: bool,
):
    start = time.time()
    guide_model = None
    if token_healing:
        guide_model = m.Model(model=draft_model, tokenizer=tokenizer)
    # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_at_ids = None
    if stop_at:
        tokenizer.padding_side = "right"
        stop_at_tokenizer_out = tokenizer(
            stop_at, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(draft_model.device)
        # Get all tokens which include a 'stop_at' text as a prefix
        # stop_at_prefix_ids = sum([guide_model.prefix_matches(s) for s in stop_at], [])
        stop_at_ids = stop_at_tokenizer_out["input_ids"]
    prompt_input_ids = (
        tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"]
        .to(draft_model.device)
        .squeeze(0)
    )
    prompt_ids_length = prompt_input_ids.shape[0]
    total_len = draft_model.config.max_position_embeddings
    log_to_string_builder: bool = save_html or verbose or debug
    string_builder = None
    if log_to_string_builder:
        string_builder = m.TransformersStringBuilder(
            tokenizer,
            starting_ids=prompt_input_ids,
            log_changes=verbose,
            write_to_html=save_html,
        )
    tokens = m.initialize_tokens(total_len, tokenizer.pad_token_id, draft_model.device)
    tokens[: len(prompt_input_ids)] = prompt_input_ids
    target_model = target_model + prompt
    start_pos = len(prompt_input_ids)

    corrections = []
    num_correction_left = max_grammar_corrections
    ret_prediction, initial_prediction, selected_candidate, program_prediction = (
        None,
        None,
        None,
        None,
    )
    prefix: str = prepare_initial_prefix(parser=parser, seed_str=seed_str)

    if prefix:
        prefix_ids = (
            tokenizer(prefix, return_tensors="pt", padding=True)
            .to(draft_model.device)["input_ids"]
            .squeeze(0)
        )
        tokens[
            len(prompt_input_ids) : len(prompt_input_ids) + len(prefix_ids)
        ] = prefix_ids
        start_pos += len(prefix_ids)
        if log_to_string_builder:
            string_builder.extend(prefix_ids, StringType.PROMPT)
        if debug:
            m.assert_valid_string_state(string_builder, tokens)
            m.assert_valid_token_state(tokens, tokenizer, start_pos)
    # Don't pass the last token of prompt here - we'll use it for generation
    past_key_values = m.forward_pass_no_sample(
        model=draft_model,
        input_ids=tokens[: start_pos - 1].unsqueeze(0),
    )
    while num_correction_left > 0 and ret_prediction is None:
        _max_new_tokens = token_lookahead
        processors = []
        if token_healing:
            healer = TokenHealingLogitsProcessor(
                guide_model,
                guide_model.model.config.vocab_size,
                # IMPORTANT: if we don't clone below, then
                #   modifying the tokens array will result in
                #   manipulation of the healed_token_ids attribute
                tokens[:start_pos].clone().detach(),
            )
            healed_token_ids = healer.healed_token_ids
            # TODO: why do things break when len(healed_token_ids) > 1?
            if len(healed_token_ids) == 1:
                _max_new_tokens = token_lookahead + len(healed_token_ids)
                start_pos -= len(healed_token_ids)
                # Reset back, depending on length of healed tokens
                tokens[start_pos:] = tokenizer.pad_token_id
                # print("Healed tokens:")
                # print(repr(tokenizer.decode(healed_token_ids)))
                past_key_values = m.prune_kv_cache(
                    past_key_values=past_key_values,
                    up_to=start_pos - 1,
                )
                # print("Backed cache up to:")
                # print(repr(tokenizer.decode(tokens[:past_key_values.key_cache[0].shape[-2]])))
                if log_to_string_builder:
                    for i in range(len(healed_token_ids)):
                        string_builder.pop(i, token_healing=True)
                if debug:
                    m.assert_valid_token_state(tokens, tokenizer, start_pos)
                    m.assert_valid_kv_cache_state(past_key_values, start_pos)
                processors.append(healer)

        tokens, start_pos, past_key_values, string_builder = m._gen_loop(
            model=draft_model,
            tokenizer=tokenizer,
            tokens=tokens,
            past_key_values=past_key_values,
            processors=processors,
            string_builder=string_builder,
            start_pos=start_pos,
            total_len=total_len,
            stop_at_ids=stop_at_ids,
            max_new_tokens=_max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        if debug:
            m.assert_valid_string_state(string_builder, tokens)
            m.assert_valid_token_state(tokens, tokenizer, start_pos)
            m.assert_valid_kv_cache_state(past_key_values, start_pos)

        generated_token_ids = tokens[prompt_ids_length:start_pos]
        program_prediction: str = tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )

        prefix, ret_prediction, correction = handle_program_prediction(
            program_prediction=program_prediction,
            parser=parser,
            draft_model=target_model,
        )
        if correction is not None:
            corrections.append(correction)
        if ret_prediction is not None:
            continue
        selected_candidate = correction.selected_candidate

        selected_candidate_ids = tokenizer(
            selected_candidate, return_tensors="pt", add_special_tokens=False
        ).to(draft_model.device)["input_ids"]
        if (
            tokens.shape[-1] - tokens[tokens != tokenizer.pad_token_id].count_nonzero()
            < selected_candidate_ids.shape[-1]
        ):
            logger.debug(Fore.RED + "Exceeded max token array length" + Fore.RESET)
            ret_prediction = prefix
            continue
        if prefix == program_prediction:
            # Before we make modifications to tokens array - make sure the last generated token is passed to model
            past_key_values = m.forward_pass_no_sample(
                model=draft_model,
                input_ids=tokens[start_pos - 1].unsqueeze(0).unsqueeze(0),
                past_key_values=past_key_values,
            )
            # Simple case: insert our candidate ids into the end of the running token array
            tokens[
                start_pos : start_pos + selected_candidate_ids.shape[-1]
            ] = selected_candidate_ids
            # Forward pass new candidate tokens - only if length > 1
            if selected_candidate_ids.shape[-1] > 1:
                past_key_values = m.forward_pass_no_sample(
                    model=draft_model,
                    input_ids=selected_candidate_ids[:, :-1],
                    past_key_values=past_key_values,
                )
            start_pos += selected_candidate_ids.shape[-1]
            if log_to_string_builder:
                string_builder.extend(
                    selected_candidate_ids.squeeze(0), StringType.CANDIDATE_SELECTION
                )
            if debug:
                m.assert_valid_string_state(string_builder, tokens)
                m.assert_valid_token_state(tokens, tokenizer, start_pos)
                m.assert_valid_kv_cache_state(past_key_values, start_pos)
        else:
            # Align the token id breakpoint with the stop position in the prefix
            # prefix = ' {\n "name": "Joseph Smith, 3"\n '
            # program_prediction =' {\n "name": "Joseph Smith, 3"\n "age": 32\n "occupation'
            stripped_pred = program_prediction
            stripped_prefix = prefix
            for p in range(generated_token_ids.shape[-1] - 1, -1, -1):
                rstrip_s = tokenizer.decode(generated_token_ids[p])
                stripped_pred = stripped_pred.removesuffix(rstrip_s)
                if len(prefix) > len(stripped_pred):
                    while not prefix.endswith(rstrip_s):
                        rstrip_s = rstrip_s[:-1]
                    stripped_prefix = stripped_prefix.removesuffix(rstrip_s)
                    assert stripped_pred == stripped_prefix
                    break
                if stripped_pred == stripped_prefix:
                    break
            # Get the remaining bits of our prefix that didn't align well
            #   with the token representation we had thus far
            diff_str = prefix.removeprefix(stripped_prefix)
            assert prefix == tokenizer.decode(generated_token_ids[:p]) + diff_str
            # Cut off our kv cache up to the valid grammar prefix
            past_key_values = m.prune_kv_cache(
                past_key_values=past_key_values,
                up_to=prompt_ids_length + p,
            )
            start_pos = prompt_ids_length + p
            # print("Backed cache up to:")
            # print(repr(tokenizer.decode(tokens[prompt_ids_length:past_key_values.key_cache[0].shape[-2]])))
            # Clear tokens after valid_completion_id index
            tokens[prompt_ids_length + p :] = tokenizer.pad_token_id
            if log_to_string_builder:
                for i in range(generated_token_ids.shape[-1] - p):
                    string_builder.pop(i)
            if debug:
                m.assert_valid_string_state(string_builder, tokens)
                # m.assert_valid_kv_cache_state(past_key_values, start_pos)
            if diff_str:
                diff_token_ids = (
                    tokenizer(diff_str, return_tensors="pt", add_special_tokens=False)[
                        "input_ids"
                    ]
                    .to(draft_model.device)
                    .squeeze(0)
                )
                diff_l = diff_token_ids.shape[-1]
                tokens[
                    prompt_ids_length + p : prompt_ids_length + p + diff_l
                ] = diff_token_ids
                past_key_values = m.forward_pass_no_sample(
                    model=draft_model,
                    input_ids=diff_token_ids.reshape(1, -1),
                    past_key_values=past_key_values,
                )
                start_pos += diff_l
                if log_to_string_builder:
                    string_builder.extend(diff_token_ids, StringType.GENERATION)
                if debug:
                    m.assert_valid_string_state(string_builder, tokens)
                    # m.assert_valid_kv_cache_state(past_key_values, start_pos)
            if (
                selected_candidate_ids.shape[-1]
                > tokens[prompt_ids_length + p :].shape[0]
            ):
                raise ValueError(
                    f"Trying to insert candidate {selected_candidate}, but it is too long!"
                )
            # Update tokens with our selected candidate id
            tokens[
                start_pos : start_pos + selected_candidate_ids.shape[-1]
            ] = selected_candidate_ids
            # Forward pass new candidate tokens - only if length > 1
            if selected_candidate_ids.shape[-1] > 1:
                past_key_values = m.forward_pass_no_sample(
                    model=draft_model,
                    input_ids=selected_candidate_ids[:, :-1],
                    past_key_values=past_key_values,
                )
            start_pos += selected_candidate_ids.shape[-1]
            if log_to_string_builder:
                string_builder.extend(
                    selected_candidate_ids.squeeze(0), StringType.CANDIDATE_SELECTION
                )
            if debug:
                m.assert_valid_string_state(string_builder, tokens)
                m.assert_valid_token_state(tokens, tokenizer, start_pos)
                m.assert_valid_kv_cache_state(past_key_values, start_pos)
        num_correction_left -= 1
        # logger.debug(
        #     Fore.YELLOW + f"Made a {corrections[-1].type} correction..." + Fore.RESET
        # )
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

    html = None
    if save_html or (verbose and is_interactive()):
        html = (
            "<div style='margin: 0px; padding: 0px; font-family: ColfaxAI, Arial; font-size: 20px;'"
            + "".join(string_builder.html)
            + "</div>"
        )
    if verbose and is_interactive():
        display(
            {"text/html": html},
            display_id=0,
            raw=True,
            include=["text/html"],
        )
    process_time_seconds = time.time() - start
    return GrammarGuideOutput(
        response=ret_prediction,
        num_grammar_corrections=len(corrections),
        correction_log=corrections,
        process_time_seconds=process_time_seconds,
        html=html,
    )


def _generic_guide(
    draft_model: Callable[[str], str],
    parser: EarleyParser,
    prompt: str,
    target_model: guidance.models.Model,
    seed_str: Optional[str],
    max_grammar_corrections: int,
    token_lookahead: int,
    verbose: bool,
) -> GrammarGuideOutput:
    start = time.time()
    corrections = []
    num_correction_left = max_grammar_corrections
    ret_prediction, initial_prediction, selected_candidate = None, None, None
    prefix: str = prepare_initial_prefix(parser=parser, seed_str=seed_str)
    while num_correction_left > 0 and ret_prediction is None:
        # Some APIs (anthropic) don't allow trailing whitespace in final assistant content
        program_prediction: str = prefix + draft_model(
            prefix=prefix.rstrip(),
            prompt=prompt,
            max_new_tokens=token_lookahead,
        )
        prefix, ret_prediction, correction = handle_program_prediction(
            program_prediction=program_prediction,
            parser=parser,
            draft_model=target_model,
        )
        if correction is not None:
            corrections.append(correction)
        if ret_prediction is not None:
            continue
        prefix += correction.selected_candidate
        num_correction_left -= 1
        logger.debug(
            Fore.YELLOW + f"Made a {corrections[-1].type} correction..." + Fore.RESET
        )
    if ret_prediction is None:
        logger.debug(
            Fore.RED
            + f"Cannot find a valid prediction after {max_grammar_corrections} retries"
            + Fore.RESET
        )
        ret_prediction = prefix
    process_time_seconds = time.time() - start
    return GrammarGuideOutput(
        response=ret_prediction,
        num_grammar_corrections=len(corrections),
        correction_log=corrections,
        process_time_seconds=process_time_seconds,
    )
