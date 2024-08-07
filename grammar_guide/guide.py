from typing import Tuple, Set, Optional, Literal, Callable, List
from colorama import Fore
import logging
from dataclasses import dataclass
import time
from tabulate import tabulate
from functools import partial
from outlines.models import LogitsGenerator
from outlines import generate

from .minEarley.parser import EarleyParser
from ._logger import logger

tabulate = partial(
    tabulate, headers="keys", showindex="never", tablefmt="simple_outline"
)

CorrectionType = Literal[
    "single_candidate",
    "draft_gen",
    "single_candidate_middle_fill",
    "draft_gen_middle_fill",
]


@dataclass
class Correction:
    original_pred: str
    corrected_pred: str
    type: CorrectionType

    def to_dict(self):
        return {
            "Original": self.original_pred,
            "Corrected": self.corrected_pred,
            "Type": self.type,
        }


@dataclass
class GrammarGuideOutput:
    response: str
    num_grammar_corrections: int
    correction_log: List[Correction]
    process_time_seconds: float

    def to_list(self):
        return [i.to_dict() for i in self.correction_log]


def validate_program(prediction: str, parser: EarleyParser) -> bool:
    try:
        parser.parse(prediction)
        return True
    except Exception as runtime_e:
        logger.debug(Fore.LIGHTCYAN_EX + prediction + Fore.RESET)
        logger.debug(f"Error: {str(runtime_e)}")
        return False


def obtain_correction_pairs(
    prediction: str,
    parser: EarleyParser,
    candidate_limit: int,
    candidate_overflow_strategy: Literal["sample", "ignore"],
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
            candidate_overflow_strategy=candidate_overflow_strategy,
        )


def guide(
    prompt: str,
    target_gen: Callable[[str], str],
    draft_model: LogitsGenerator,
    cfg_grammar_str: Optional[str] = None,
    cfg_grammar_filepath: Optional[str] = None,
    max_grammar_corrections: int = 3,
    seed_str: Optional[str] = None,
    candidate_limit: Optional[int] = 64,
    candidate_overflow_strategy: Optional[Literal["sample", "ignore"]] = "sample",
    verbose: bool = False,
) -> GrammarGuideOutput:
    start = time.time()
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

    if all([x is None for x in {cfg_grammar_str, cfg_grammar_filepath}]):
        raise ValueError(
            "One of `cfg_grammar_str`, `cfg_grammar_filepath` must be specified!"
        )
    elif cfg_grammar_filepath:
        cfg_grammar_str = open(cfg_grammar_filepath).read()
    parser: EarleyParser = EarleyParser(
        grammar=cfg_grammar_str,
        start="start",
        keep_all_tokens=True,
    )

    # Just return a single generation
    if max_grammar_corrections == 0:
        return GrammarGuideOutput(
            response=target_gen(prompt),
            num_grammar_corrections=0,
            corrections=[],
            process_time_seconds=time.time() - start,
        )

    num_correction_left = max_grammar_corrections
    partial_program_prediction = seed_str or ""
    ret_prediction, initial_prediction = None, None
    corrections = []
    while num_correction_left > 0 and ret_prediction is None:
        residual_program_prediction = target_gen(prompt + partial_program_prediction)

        program_prediction = (
            partial_program_prediction + " " + residual_program_prediction
        )

        if validate_program(program_prediction, parser):
            ret_prediction = program_prediction
            continue

        # find the max score from a list of score
        prefix, candidates, has_re, pos_in_stream = obtain_correction_pairs(
            prediction=program_prediction,
            parser=parser,
            candidate_limit=candidate_limit,
            candidate_overflow_strategy=candidate_overflow_strategy,
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
            # Generate the continuation candidate with the highest probability
            selected_candidate = generate.regex(draft_model, "|".join(candidates))(
                prompt + prefix
            )
            correction_type = "draft_gen"

        # Try to use our selected candidate in a few ways
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
        # 2) If rest of our query is also broken, we just keep up to the prefix + candidate
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
            corrections.append(
                Correction(
                    original_pred=program_prediction,
                    corrected_pred=partial_program_prediction,
                    type=correction_type,
                )
            )
        num_correction_left -= 1

    if ret_prediction is None:
        logger.debug(
            Fore.RED
            + f"Cannot find a valid prediction after {max_grammar_corrections} retries"
            + Fore.RESET
        )
        ret_prediction = corrections[-1].corrected_pred
    logger.debug(Fore.GREEN + ret_prediction + Fore.RESET)
    return GrammarGuideOutput(
        response=ret_prediction,
        num_grammar_corrections=len(corrections),
        correction_log=corrections,
        process_time_seconds=time.time() - start,
    )
