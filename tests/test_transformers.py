import guidance
import guidance
import json

import grammar_guide as gg


def test_simple_guide(
    draft_model, tokenizer, grammar_template, prompt, stop_string_list
):
    parser = gg.load_parser(grammar_template.safe_substitute(NUM_REPEATS=5))
    res = gg.guide(
        draft_model,
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        target_model=guidance.models.Transformers(
            draft_model.config.name_or_path, echo=False
        ),
        stop_at=stop_string_list,
        max_grammar_corrections=5,
        token_lookahead=20,
        temperature=0.0,
        token_healing=True,
        verbose=True,
        debug=True,
    )
    print(json.dumps(json.loads(res.response), indent=4))
