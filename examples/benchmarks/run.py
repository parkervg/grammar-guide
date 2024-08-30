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
from string import Template
import seaborn as sns
import matplotlib.pyplot as plt
import grammar_guide as gg

gg.modeling_utils.set_seed(42)

GRAMMAR_GUIDE_MAX_NEW_TOKENS = 50
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
        max_new_tokens=2000,
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


def run_grammar_guide(
    model,
    tokenizer,
    grammar_str,
    prompt,
    max_new_tokens,
    token_healing,
    draft_model_name_or_path,
):
    import grammar_guide as gg

    start = time.time()
    parser = gg.load_parser(grammar_str)
    gen_start = time.time()
    res = gg.guide(
        model,
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        draft_model=guidance.models.Transformers(draft_model_name_or_path, echo=False),
        stop_at=STOP_STRING_LIST,
        max_grammar_corrections=20,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        verbose=False,
        token_healing=token_healing,
        debug=False,
    )
    print(
        f"SGB OUTPUT (with {res.num_grammar_corrections} corrections, token_healing={token_healing})"
    )
    print(res.response)
    elapsed_time_seconds = time.time() - start
    elapsed_gen_time_seconds = time.time() - gen_start
    return (
        res,
        elapsed_time_seconds,
        elapsed_gen_time_seconds,
        len(
            tokenizer(res.response, add_special_tokens=False, padding=True)["input_ids"]
        )
        / elapsed_gen_time_seconds,
    )


def run_naive_grammar_guide(
    model, tokenizer, grammar_str, max_new_tokens, prompt, draft_model_name_or_path
):
    import grammar_guide as gg

    start = time.time()
    parser = gg.load_parser(grammar_str)
    gen_start = time.time()

    def generate(text: str):
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
        model_output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            stop_strings=STOP_STRING_LIST,
            tokenizer=tokenizer,
            do_sample=False,
        )
        return tokenizer.decode(
            model_output[:, model_inputs["input_ids"].shape[-1] :][0],
            skip_special_tokens=True,
        )

    res = gg.guide(
        lambda x: generate(x),
        tokenizer=tokenizer,
        parser=parser,
        prompt=prompt,
        draft_model=guidance.models.Transformers(draft_model_name_or_path, echo=False),
        stop_at=STOP_STRING_LIST,
        max_grammar_corrections=20,
        max_new_tokens=GRAMMAR_GUIDE_MAX_NEW_TOKENS,
        temperature=0.0,
        verbose=False,
        token_healing=False,
        debug=False,
    )
    print(f"NAIVE SGB OUTPUT (with {res.num_grammar_corrections} corrections)")
    print(res.response)
    elapsed_time_seconds = time.time() - start
    elapsed_gen_time_seconds = time.time() - gen_start
    return (
        res,
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

    lark_grammar_str = Template(open(PARENT_DIR / "json.lark").read())
    ebnf_grammar_str = Template(open(PARENT_DIR / "json.ebnf").read())

    num_iters_per_trial = 3
    json_key_trials = [10, 20, 30, 40]
    output = []
    for num_json_keys in json_key_trials:
        print(rf"Running eval with {num_json_keys} JSON keys...")
        prompt = dedent(
            f"""
        This is an introduction to a prompt. It is intended to mimick the lengthy few-shot prompts we tend to use.
        Anyways, now I will get to my real point.
        Here is a JSON object, with {num_json_keys} keys, using only string values:\n\n```json\n
        """
        )
        curr_ebnf_grammar_str = ebnf_grammar_str.safe_substitute(
            REPEATED_STRING_VALUES=' "," ws string ":" ws string ' * (num_json_keys - 1)
        )
        curr_lark_grammar_str = lark_grammar_str.safe_substitute(
            NUM_REPEATS=f"{num_json_keys-1}"
        )
        # Define benchmarks
        name_to_f = {
            "Naive Grammar Guide": partial(
                run_naive_grammar_guide,
                model,
                tokenizer,
                curr_lark_grammar_str,
                prompt,
            ),
            "Grammar Guide": partial(
                run_grammar_guide,
                model,
                tokenizer,
                curr_lark_grammar_str,
                prompt,
                token_healing=False,
            ),
            "Grammar Guide (with token healing)": partial(
                run_grammar_guide,
                model,
                tokenizer,
                curr_lark_grammar_str,
                prompt,
                token_healing=True,
            ),
            "Transformers CFG": partial(
                run_transformers_cfg, model, tokenizer, curr_ebnf_grammar_str, prompt
            ),
            "Syncode": partial(
                run_syncode, model, tokenizer, curr_lark_grammar_str, prompt
            ),
        }
        for name, f in name_to_f.items():
            for _ in range(num_iters_per_trial):
                time_elapsed, gen_time_elapsed, tokens_per_second = f()
                output.append(
                    {
                        "Name": name,
                        "Time Elapsed (s)": time_elapsed,
                        "Generation Time": gen_time_elapsed,
                        "Tokens Per Second": tokens_per_second,
                        "# JSON Keys": num_json_keys,
                    }
                )
    df = pd.DataFrame(output)
    df.to_csv(PARENT_DIR / "json_benchmark.csv", index=False)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.lineplot(data=df, x="# JSON Keys", y="Time Elapsed (s)", hue="Name")
    ax.set_xticks(json_key_trials, labels=json_key_trials)
    plt.title(
        f"Time to Generate JSON Using {model_name_or_path}, Averaged Across {num_iters_per_trial} Trials"
    )
    plt.savefig(
        PARENT_DIR / "runtime_lineplot.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="w",
    )
