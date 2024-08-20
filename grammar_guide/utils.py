from typing import Tuple, Set, Optional

import guidance.models

from .minEarley.parser import EarleyParser
from .typedefs import Correction


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def validate_program(prediction: str, parser: EarleyParser) -> bool:
    try:
        parser.parse(prediction)
        return True
    except Exception:
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


def prepare_initial_prefix(parser: EarleyParser, seed_str: Optional[str] = None) -> str:
    prefix = seed_str or ""
    # Check to see if our grammar gives us any freebies at the beginning of prediction
    # E.g. maybe in our SQL grammar, we can only begin with 'SELECT'
    if prefix == "":
        _, str_candidates, re_candidates, _ = obtain_correction_pairs(
            prediction=prefix,
            parser=parser,
            candidate_limit=64,
        )
        if len(re_candidates) == 0 and len(str_candidates) == 1:
            prefix = str_candidates.pop()
    return prefix


def handle_program_prediction(
    program_prediction: str, parser: EarleyParser, draft_model: guidance.models.Model
) -> tuple:
    """Checks the given program_prediction against our grammar.
    If the existing prediction is valid, we first select one of the possible candidates
        using our draft_model.
    Then, we perform a series of checks to see if we can naively fix the prediction
        using our selected candidate to satisfy the grammar constraints.

    Returns:
        Tuple of (prefix, ret_prediction, correction)
    """
    if validate_program(program_prediction, parser):
        return (None, program_prediction, None)

    prefix, str_candidates, re_candidates, pos_in_stream = obtain_correction_pairs(
        prediction=program_prediction,
        parser=parser,
        candidate_limit=64,
    )
    if all(len(x) == 0 for x in [str_candidates, re_candidates]):
        return (prefix, prefix, None)

    if len(str_candidates) == 1 and len(re_candidates) == 0:
        # If we only have 1 string candidate, no need to call draft_gen
        selected_candidate = str_candidates.pop()
        correction_type = "single_candidate"
    else:
        import guidance
        import re

        make_regex_pred = lambda pattern: (
            draft_model
            + prefix
            + guidance.capture(
                guidance.with_temperature(guidance.regex(pattern=pattern), 0.0),
                "res",
            )
        )["res"]

        selected_candidate = make_regex_pred(
            "|".join([re.escape(s) for s in str_candidates] + re_candidates)
        )
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
        correction_type += "_middle_fill"
        return (
            prefix,
            inserted_candidate_prediction,
            Correction(
                original_pred=program_prediction,
                corrected_pred=inserted_candidate_prediction,
                selected_candidate=selected_candidate,
                type=correction_type,
            ),
        )
    # 2) Just keep up to the prefix + selected_candidate
    # For example, if we just forgot a semicolon at the end of a JavaScript line
    elif validate_program(partial_program_prediction, parser):
        return (
            prefix,
            partial_program_prediction,
            Correction(
                original_pred=program_prediction,
                corrected_pred=partial_program_prediction,
                selected_candidate=selected_candidate,
                type=correction_type,
            ),
        )
    else:
        # 3) If rest of our query is also broken, we just keep up to the prefix + candidate
        # and re-generate a continuation again
        return (
            prefix,
            None,
            Correction(
                original_pred=program_prediction,
                corrected_pred=partial_program_prediction,
                selected_candidate=selected_candidate,
                type=correction_type,
            ),
        )
