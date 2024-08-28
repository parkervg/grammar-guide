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
    model: Union[AutoModelForCausalLM, Callable[[str], str]],
    parser: EarleyParser,
    prompt: str,
    draft_model: guidance.models.Model,
    tokenizer: Optional[AutoTokenizer] = None,
    seed_str: Optional[str] = None,
    max_grammar_corrections: int = 3,
    stop_at: Optional[Collection[str]] = None,
    token_healing: Optional[bool] = True,
    top_p: float = 0.9,
    temperature: float = 0.6,
    max_new_tokens: int = 32,
    save_html: bool = True,
    verbose: bool = True,
    debug: bool = False,
) -> GrammarGuideOutput:
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
    if hasattr(model, "config"):
        return _transformers_guide(
            model=model,
            tokenizer=tokenizer,
            parser=parser,
            prompt=prompt,
            draft_model=draft_model,
            seed_str=seed_str,
            max_grammar_corrections=max_grammar_corrections,
            stop_at=stop_at,
            token_healing=token_healing,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            save_html=save_html,
            verbose=verbose,
            debug=debug,
        )
    assert isinstance(model, Callable)
    return _generic_guide(
        model=model,
        parser=parser,
        prompt=prompt,
        draft_model=draft_model,
        seed_str=seed_str,
        max_grammar_corrections=max_grammar_corrections,
    )


# @profile
def _transformers_guide(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    parser: EarleyParser,
    prompt: str,
    draft_model: guidance.models.Model,
    seed_str: Optional[str],
    max_grammar_corrections: int,
    stop_at: Collection[str],
    token_healing: Optional[bool],
    top_p: float,
    temperature: float,
    max_new_tokens: int,
    save_html: bool,
    verbose: bool,
    debug: bool,
):
    start = time.time()
    guide_model = None
    if token_healing:
        guide_model = m.Model(model=model, tokenizer=tokenizer)
    # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_at_ids = None
    if stop_at:
        tokenizer.padding_side = "right"
        stop_at_tokenizer_out = tokenizer(
            stop_at, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        # Get all tokens which include a 'stop_at' text as a prefix
        # stop_at_prefix_ids = sum([guide_model.prefix_matches(s) for s in stop_at], [])
        stop_at_ids = stop_at_tokenizer_out["input_ids"]
    prompt_input_ids = (
        tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"]
        .to(model.device)
        .squeeze(0)
    )
    prompt_ids_length = prompt_input_ids.shape[0]
    total_len = model.config.max_position_embeddings
    string_builder = m.TransformersStringBuilder(
        tokenizer,
        starting_ids=prompt_input_ids,
        log_changes=verbose,
        write_to_html=save_html,
    )
    tokens = m.initialize_tokens(total_len, tokenizer.pad_token_id, model.device)
    tokens[: len(prompt_input_ids)] = prompt_input_ids
    draft_model = draft_model + prompt
    start_pos = len(prompt_input_ids)

    corrections = []
    num_correction_left = max_grammar_corrections
    ret_prediction, initial_prediction, selected_candidate = None, None, None
    prefix: str = prepare_initial_prefix(parser=parser, seed_str=seed_str)

    if prefix:
        prefix_ids = (
            tokenizer(prefix, return_tensors="pt", padding=True)
            .to(model.device)["input_ids"]
            .squeeze(0)
        )
        tokens[
            len(prompt_input_ids) : len(prompt_input_ids) + len(prefix_ids)
        ] = prefix_ids
        start_pos += len(prefix_ids)
        string_builder.extend(prefix_ids, StringType.PROMPT)
        if debug:
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
                if debug:
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
        if debug:
            m.assert_valid_string_state(string_builder, tokens)
            m.assert_valid_token_state(tokens, tokenizer, start_pos)

        generated_token_ids = tokens[prompt_ids_length:start_pos]
        program_prediction: str = tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )

        prefix, ret_prediction, correction = handle_program_prediction(
            program_prediction=program_prediction,
            parser=parser,
            draft_model=draft_model,
        )
        if correction is not None:
            corrections.append(correction)
        if ret_prediction is not None:
            continue
        selected_candidate = correction.selected_candidate

        selected_candidate_ids = tokenizer(
            selected_candidate, return_tensors="pt", add_special_tokens=False
        ).to(model.device)["input_ids"]
        if (
            tokens.shape[-1] - tokens[tokens != tokenizer.pad_token_id].count_nonzero()
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
            if debug:
                m.assert_valid_string_state(string_builder, tokens)
                m.assert_valid_token_state(tokens, tokenizer, start_pos)
        else:
            # Align the token id breakpoint with the stop position in the prefix
            # prefix = ' {\n "name": "Joseph Smith, 3"\n '
            # program_prediction =' {\n "name": "Joseph Smith, 3"\n "age": 32\n "occupation'
            stripped_pred = program_prediction
            stripped_prefix = prefix
            for p in range(generated_token_ids.shape[-1] - 1, -1, -1):
                rstrip_s = tokenizer.decode(generated_token_ids[p])
                stripped_pred = stripped_pred.rstrip(rstrip_s)
                if len(prefix) > len(stripped_pred):
                    while not prefix.endswith(rstrip_s):
                        rstrip_s = rstrip_s[:-1]
                    stripped_prefix = stripped_prefix.rstrip(rstrip_s)
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
            # Clear tokens after valid_completion_id index
            tokens[prompt_ids_length + p :] = tokenizer.pad_token_id
            for i in range(generated_token_ids.shape[-1] - p):
                string_builder.pop(i)
            if debug:
                m.assert_valid_string_state(string_builder, tokens)
            diff_l = 0
            if diff_str:
                diff_token_ids = (
                    tokenizer(diff_str, return_tensors="pt", add_special_tokens=False)[
                        "input_ids"
                    ]
                    .to(model.device)
                    .squeeze(0)
                )
                diff_l = diff_token_ids.shape[-1]
                tokens[
                    prompt_ids_length + p : prompt_ids_length + p + diff_l
                ] = diff_token_ids
                past_key_values = m.forward_pass_no_sample(
                    model=model,
                    input_ids=diff_token_ids.reshape(1, -1),
                    past_key_values=past_key_values,
                )
                string_builder.extend(diff_token_ids, StringType.GENERATION)
                if debug:
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
                + diff_l : prompt_ids_length
                + p
                + diff_l
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
                prompt_ids_length + p + diff_l + selected_candidate_ids.shape[-1]
            )
            string_builder.extend(
                selected_candidate_ids.squeeze(0), StringType.CANDIDATE_SELECTION
            )
            if debug:
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
    model: Callable[[str], str],
    parser: EarleyParser,
    prompt: str,
    draft_model: guidance.models.Model,
    seed_str: Optional[str],
    max_grammar_corrections: int,
) -> GrammarGuideOutput:
    start = time.time()
    corrections = []
    num_correction_left = max_grammar_corrections
    ret_prediction, initial_prediction, selected_candidate = None, None, None
    prefix: str = prepare_initial_prefix(parser=parser, seed_str=seed_str)
    while num_correction_left > 0 and ret_prediction is None:
        program_prediction: str = prefix + model(prompt + prefix)
        logger.debug(Fore.YELLOW + program_prediction + Fore.RESET)
        prefix, ret_prediction, correction = handle_program_prediction(
            program_prediction=program_prediction,
            parser=parser,
            draft_model=draft_model,
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
