from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from pathlib import Path
import torch
from string import Template
import pytest

set_seed(42)

PARENT_DIR = Path(__file__).parent
MODEL_NAME_OR_PATH = "HuggingFaceTB/SmolLM-135M"


def load_model(model_name_or_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="cuda" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return (model, tokenizer)


@pytest.fixture(scope="session")
def draft_model() -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH, device_map="cuda" if torch.cuda.is_available() else None
    )
    return model


@pytest.fixture(scope="session")
def tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    return tokenizer


@pytest.fixture(scope="session")
def grammar_template() -> Template:
    return Template(open(PARENT_DIR / "../examples/benchmarks/json.lark").read())


@pytest.fixture(scope="session")
def prompt() -> str:
    return "Here is a JSON."


@pytest.fixture(scope="session")
def stop_string_list() -> str:
    return ["```", "}"]
