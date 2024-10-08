from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import guidance
from textwrap import dedent

STOP_STRING_LIST = ["```", "}"]
PARENT_DIR = Path(__file__).parent


def load_model(model_name_or_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="cuda" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return (model, tokenizer)


if __name__ == "__main__":
    """
    reversed-string-alignment: 4.99
    main: 5.61 (6.65 when constructing guide_model var)

    Example of token healing:
        - tokens end in [216, 18] or ' ' + '"'
        - model chooses [476] or ' "'
    TODO:
        - Look into the 2 healed token situation
            - Is the generation after the 1st constrained token the same
                when doing normal (non-token healed) generation?
        - Look into huggingface token healing: https://github.com/huggingface/transformers/blob/c409cd81777fb27aadc043ed3d8339dbc020fb3b/src/transformers/generation/utils.py#L2217
    """
    from string import Template
    import grammar_guide as gg
    from transformers import set_seed

    set_seed(42)

    num_json_keys = 10

    prompt = dedent(
        f"""
        This is an introduction to a prompt. It is intended to mimick the lengthy few-shot prompts we tend to use.
        Anyways, now I will get to my real point.
        Here is a JSON object, with {num_json_keys} keys, using only string values:\n\n```json\n
        """
    )
    lark_grammar_str = Template(
        open(PARENT_DIR / "examples/benchmarks/json.lark").read()
    )
    lark_grammar_str = lark_grammar_str.safe_substitute(
        NUM_REPEATS=f"{num_json_keys - 1}"
    )

    model_name_or_path = "HuggingFaceTB/SmolLM-360M"
    model, tokenizer = load_model(model_name_or_path=model_name_or_path)
    res = gg.guide(
        model,
        # seed_str=" {",
        tokenizer=tokenizer,
        parser=gg.load_parser(lark_grammar_str),
        prompt=prompt,
        target_model=guidance.models.Transformers(model_name_or_path, echo=False),
        stop_at=STOP_STRING_LIST,
        max_grammar_corrections=10,
        token_lookahead=10,
        temperature=0.3,
        token_healing=True,
        verbose=True,
        debug=True,
    )
    import json

    with open("main.json", "w") as f:
        json.dump(res.to_list(), f, indent=4)
    print(res.num_grammar_corrections)
    print(res.process_time_seconds)
    print(res.response)
