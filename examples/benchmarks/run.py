from textwrap import dedent
import guidance
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from pathlib import Path
import subprocess
import sys
import pandas as pd
import importlib.util

GRAMMAR_GUIDE_MAX_NEW_TOKENS = 200
STOP_STRING_LIST = ["```", "}"]
PARENT_DIR = Path(__file__).parent


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def load_model(model_name_or_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="cuda" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return (model, tokenizer)


def run_transformers_cfg(model, tokenizer, grammar_str, prompt):
    _has_transformers_cfg = importlib.util.find_spec("transformers_cfg") is not None
    if not _has_transformers_cfg:
        install("transformers_cfg")
    from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
    from transformers_cfg.generation.logits_process import (
        GrammarConstrainedLogitsProcessor,
    )

    start = time.time()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    gen_start = time.time()
    input_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt", padding=True
    ).to(model.device)["input_ids"]
    output = model.generate(
        input_ids,
        max_new_tokens=200,
        logits_processor=[grammar_processor],
        repetition_penalty=1.1,
        num_return_sequences=1,
        do_sample=False,
    )
    print("TRANSFORMERS CFG OUTPUT")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    elapsed_time_seconds = time.time() - start
    elapsed_gen_time_seconds = time.time() - gen_start
    return (
        elapsed_time_seconds,
        elapsed_gen_time_seconds,
        output.shape[-1] / elapsed_gen_time_seconds,
    )


def run_grammar_guide(model, tokenizer, grammar_str, prompt):
    import grammar_guide as gg

    start = time.time()
    parser = gg.load_parser(grammar_str)
    gen_start = time.time()
    res = gg.guide(
        model,
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        draft_model=guidance.models.Transformers(model_name_or_path, echo=False),
        stop_at=STOP_STRING_LIST,
        max_grammar_corrections=20,
        verbose=False,
        max_new_tokens=GRAMMAR_GUIDE_MAX_NEW_TOKENS,
        temperature=0.0,
    )
    print(f"SGB OUTPUT (with {res.num_grammar_corrections} corrections)")
    print(res.response)
    elapsed_time_seconds = time.time() - start
    elapsed_gen_time_seconds = time.time() - gen_start
    return (
        elapsed_time_seconds,
        elapsed_gen_time_seconds,
        len(
            tokenizer(res.response, add_special_tokens=False, padding=True)["input_ids"]
        )
        / elapsed_gen_time_seconds,
    )


def run_naive_grammar_guide(model, tokenizer, grammar_str, prompt):
    import grammar_guide as gg

    start = time.time()
    parser = gg.load_parser(grammar_str)
    gen_start = time.time()

    def generate(text: str):
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
        model_output = model.generate(
            **model_inputs,
            max_new_tokens=200,
            stop_strings=STOP_STRING_LIST,
            tokenizer=tokenizer,
        )
        return tokenizer.decode(model_output[0], skip_special_tokens=True)

    res = gg.guide(
        lambda x: generate(x).lstrip(prompt),
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        draft_model=guidance.models.Transformers(model_name_or_path, echo=False),
        stop_at=STOP_STRING_LIST,
        max_grammar_corrections=20,
        verbose=True,
        max_new_tokens=GRAMMAR_GUIDE_MAX_NEW_TOKENS,
        temperature=0.0,
    )
    print(f"NAIVE SGB OUTPUT (with {res.num_grammar_corrections} corrections)")
    print(res.response)
    elapsed_time_seconds = time.time() - start
    elapsed_gen_time_seconds = time.time() - gen_start
    return (
        elapsed_time_seconds,
        elapsed_gen_time_seconds,
        len(
            tokenizer(res.response, add_special_tokens=False, padding=True)["input_ids"]
        )
        / elapsed_gen_time_seconds,
    )


def run_syncode(model, tokenizer, grammar_str, prompt):
    _has_syncode = importlib.util.find_spec("syncode") is not None
    if not _has_syncode:
        install("git+https://github.com/uiuc-focal-lab/syncode.git")
    from syncode import Syncode

    start = time.time()
    # Syncode only accepts the huggingface str id as input
    syn_llm = Syncode(
        model=model.config.name_or_path,
        grammar=grammar_str,
        parse_output_only=True,
        mode="grammar_strict",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    gen_start = time.time()
    output = syn_llm.infer(prompt)[0]
    print("SYNCODE OUTPUT:")
    print(output)
    elapsed_time_seconds = time.time() - start
    elapsed_gen_time_seconds = time.time() - gen_start
    return (
        elapsed_time_seconds,
        elapsed_gen_time_seconds,
        len(tokenizer(output, add_special_tokens=False, padding=True)["input_ids"])
        / elapsed_time_seconds,
    )


if __name__ == "__main__":
    model_name_or_path = "HuggingFaceTB/SmolLM-135M"
    model, tokenizer = load_model(model_name_or_path)

    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=20,
        return_full_text=False,
    )
    pipe("hello")
    lark_grammar_str = open(PARENT_DIR / "json.lark").read()
    ebnf_grammar_str = open(PARENT_DIR / "json.ebnf").read()
    prompt = dedent(
        """
    Here is a really long JSON object, with 10 keys, using only string values:\n\n```json\n
    """
    )
    # Run benchmarks
    name_to_f = {
        "Grammar Guide": partial(
            run_grammar_guide,
            model,
            tokenizer,
            lark_grammar_str,
            prompt,
        ),
        "Naive Grammar Guide": partial(
            run_naive_grammar_guide,
            model,
            tokenizer,
            lark_grammar_str,
            prompt,
        ),
        "Transformers CFG": partial(
            run_transformers_cfg, model, tokenizer, ebnf_grammar_str, prompt
        ),
        "Syncode": partial(run_syncode, model, tokenizer, lark_grammar_str, prompt),
    }
    output = []
    num_iters = 5
    for name, f in name_to_f.items():
        time_elapsed, gen_time_elapsed, tokens_per_second = 0, 0, 0
        for _ in range(num_iters):
            _time_elapsed, _gen_time_elapsed, _tokens_per_second = f()
            time_elapsed += _time_elapsed
            gen_time_elapsed += _gen_time_elapsed
            tokens_per_second += _tokens_per_second
        output.append(
            {
                "Name": name,
                "Time Elapsed": time_elapsed / num_iters,
                "Generation Time": gen_time_elapsed / num_iters,
                "Tokens Per Second": tokens_per_second / num_iters,
            }
        )
    print(pd.DataFrame(output).to_markdown(index=False))
