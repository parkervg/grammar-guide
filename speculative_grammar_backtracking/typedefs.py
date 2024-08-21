from typing import Literal, List
from dataclasses import dataclass
from enum import Enum


class StringType(str, Enum):
    PROMPT = "prompt"
    DELETION = "deletion"
    GENERATION = "generation"
    CANDIDATE_SELECTION = "candidate_selection"


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
    selected_candidate: str
    type: CorrectionType

    def to_dict(self):
        return {
            "Original": self.original_pred,
            "Corrected": self.corrected_pred,
            "Selected Candidate": self.selected_candidate,
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
