from pathlib import Path
from string import Template
from textwrap import dedent
from functools import partial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .run import load_model, run_grammar_guide, run_naive_grammar_guide

STOP_STRING_LIST = ["```", "}"]
PARENT_DIR = Path(__file__).parent

if __name__ == "__main__":
    model_name_or_path = "HuggingFaceTB/SmolLM-135M"
    num_json_keys = 20
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

    lark_grammar_str = Template(open(PARENT_DIR / "json.lark").read()).safe_substitute(
        NUM_REPEATS=f"{num_json_keys-1}"
    )
    prompt = dedent(
        f"""
            This is an introduction to a prompt. It is intended to mimick the lengthy few-shot prompts we tend to use.
            Anyways, now I will get to my real point.
            Here is a JSON object, with {num_json_keys} keys, using only string values:\n\n```json\n
            """
    )
    num_iters_per_trial = 1
    max_new_token_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    output = []
    for max_new_tokens in max_new_token_list:
        name_to_f = {
            "Naive Grammar Guide": partial(
                run_naive_grammar_guide,
                model=model,
                tokenizer=tokenizer,
                grammar_str=lark_grammar_str,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                draft_model_name_or_path=model_name_or_path,
            ),
            "Grammar Guide (with token healing)": partial(
                run_grammar_guide,
                model=model,
                tokenizer=tokenizer,
                grammar_str=lark_grammar_str,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                token_healing=True,
                draft_model_name_or_path=model_name_or_path,
            ),
        }
        for name, f in name_to_f.items():
            print(max_new_tokens)
            for _ in range(num_iters_per_trial):
                res, time_elapsed, gen_time_elapsed, tokens_per_second = f()
                output.append(
                    {
                        "Name": name,
                        "Time Elapsed (s)": time_elapsed,
                        "Generation Time": gen_time_elapsed,
                        "Tokens Per Second": tokens_per_second,
                        "Max New Tokens": max_new_tokens,
                        "Num Corrections": res.num_grammar_corrections,
                    }
                )
    df = pd.DataFrame(output)
    df.to_csv(PARENT_DIR / "naive_vs_gg.csv", index=False)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.lineplot(data=df, x="Max New Tokens", y="Tokens Per Second", hue="Name")
    ax.set_xticks(max_new_token_list, labels=max_new_token_list)
    plt.title(
        f"Naive Backtracking vs. Optimized Grammar Guide for {num_json_keys} JSON Keys"
    )
    plt.savefig(
        PARENT_DIR / "naive_vs_gg.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="w",
    )
