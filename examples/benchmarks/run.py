from textwrap import dedent
import guidance
import outlines
from tabulate import tabulate
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def load_model(model_name_or_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return (model, tokenizer)


def transformers_cfg(model_name_or_path, grammar_str, prompt):
    from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
    from transformers_cfg.generation.logits_process import (
        GrammarConstrainedLogitsProcessor,
    )

    model, tokenizer = load_model(model_name_or_path)
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    start = time.time()
    input_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
        do_sample=False,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    elapsed_time_seconds = time.time() - start
    return elapsed_time_seconds, output.shape[-1] / elapsed_time_seconds


def speculative_grammar_backtracking(model_name_or_path, grammar_str, prompt):
    from speculative_grammar_backtracking import guide, load_parser

    model, tokenizer = load_model(model_name_or_path)
    parser = load_parser(grammar_str)
    start = time.time()
    res = guide(
        model,
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        draft_model=guidance.models.Transformers(model_name_or_path, echo=False),
        stop_at=["```", "}"],
        max_grammar_corrections=20,
        verbose=True,
        max_new_tokens=200,
        temperature=0.0,
    )
    print(f"SGB OUTPUT (with {res.num_grammar_corrections} corrections)")
    print(res.response)
    elapsed_time_seconds = time.time() - start
    return (
        elapsed_time_seconds,
        len(
            tokenizer(res.response, add_special_tokens=False, padding=True)["input_ids"]
        )
        / elapsed_time_seconds,
    )


def naive_speculative_grammar_backtracking(model_name_or_path, grammar_str, prompt):
    from transformers import pipeline
    from speculative_grammar_backtracking import guide, load_parser

    model, tokenizer = load_model(model_name_or_path)
    parser = load_parser(grammar_str)
    start = time.time()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=200,
        return_full_text=True,
    )
    res = guide(
        lambda x: pipe(x)[0]["generated_text"].lstrip(prompt),
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        draft_model=guidance.models.Transformers(model_name_or_path, echo=False),
        stop_at=["```", "}"],
        max_grammar_corrections=20,
        verbose=False,
        max_new_tokens=200,
        temperature=0.0,
    )
    print(f"NAIVE SGB OUTPUT (with {res.num_grammar_corrections} corrections)")
    print(res.response)
    elapsed_time_seconds = time.time() - start
    return (
        elapsed_time_seconds,
        len(
            tokenizer(res.response, add_special_tokens=False, padding=True)["input_ids"]
        )
        / elapsed_time_seconds,
    )


def outlines_cfg(model_name_or_path, grammar_str, prompt):
    _, tokenizer = load_model(model_name_or_path)
    model = outlines.models.transformers(model_name_or_path)
    start = time.time()
    generator = outlines.generate.cfg(model, grammar_str)
    sequence = generator(prompt)
    print("OUTLINES CFG OUTPUT:")
    print(sequence)
    elapsed_time_seconds = time.time() - start
    return (
        elapsed_time_seconds,
        len(tokenizer(sequence, add_special_tokens=False, padding=True)["input_ids"])
        / elapsed_time_seconds,
    )


if __name__ == "__main__":
    model_name_or_path = "HuggingFaceTB/SmolLM-360M"
    lark_grammar_str = open("./grammars/json.lark").read()
    ebnf_grammar_str = open("./grammars/json.ebnf").read()
    prompt = dedent(
        """
    Here is a really long JSON object, with 10 keys, using only string values:\n\n```json\n
    """
    )
    # Run benchmarks
    tabulate = partial(
        tabulate, headers="keys", showindex="never", tablefmt="simple_outline"
    )
    name_to_f = {
        "SGB": partial(
            speculative_grammar_backtracking,
            model_name_or_path,
            lark_grammar_str,
            prompt,
        ),
        "Naive SGB": partial(
            naive_speculative_grammar_backtracking,
            model_name_or_path,
            lark_grammar_str,
            prompt,
        ),
        "Transformers CFG": partial(
            transformers_cfg, model_name_or_path, ebnf_grammar_str, prompt
        ),
    }
    output = []
    for name, f in name_to_f.items():
        time_elapsed, tokens_per_second = f()
        output.append(
            {
                "Name": name,
                "Time Elapsed": time_elapsed,
                "Tokens Per Second": tokens_per_second,
            }
        )
    print(tabulate(output))
