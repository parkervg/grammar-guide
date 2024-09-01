import json
from functools import partial
import guidance

import grammar_guide as gg


def generate(
    prefix: str, prompt: str, model, tokenizer, max_new_tokens: int, stop_string_list
):
    model_inputs = tokenizer(prompt + prefix, return_tensors="pt").to(model.device)
    model_output = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        stop_strings=stop_string_list,
        tokenizer=tokenizer,
        do_sample=False,
    )
    return tokenizer.decode(
        model_output[:, model_inputs["input_ids"].shape[-1] :][0],
        skip_special_tokens=True,
    )


def test_simple_guide(
    draft_model, tokenizer, grammar_template, prompt, stop_string_list
):
    parser = gg.load_parser(grammar_template.safe_substitute(NUM_REPEATS=5))
    _generate = partial(
        generate,
        model=draft_model,
        tokenizer=tokenizer,
        stop_string_list=stop_string_list,
    )
    res = gg.guide(
        _generate,
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        target_model=guidance.models.Transformers(
            draft_model.config.name_or_path, echo=False
        ),
        stop_at=stop_string_list,
        max_grammar_corrections=20,
        token_lookahead=20,
        temperature=0.0,
        verbose=False,
        token_healing=False,
        debug=True,
    )
    print(json.dumps(json.loads(res.response), indent=4))
