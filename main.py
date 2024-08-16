if __name__ == "__main__":
    """
    Name ideas:
        - cfg-backspace
    Open questions:
        - How to align tokenizer representation with Lark lexer representation?
            - The token 'SE' matches to CNAME under the lexer, but we could complete with 'SELECT'

    Proposed flow:
        - Generate unconstrained
        - Run lark parser to fetch grammar prefix breakpoint
        - Using prefix breakpoint, modify the kv cache and re-run generate

    Ideas:
        - Add a parameter k
            - When a generated tokens' logprob deviates below k, intervene (even if we haven't hit a different eos_reached criteria)
        - Add to interactive parser in parallel to each forward pass
            - This way, we intervene right away when an error occurs

    """
    from textwrap import dedent
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import guidance
    import time
    import json

    from grammar_guide import guide

    model_name_or_path = "HuggingFaceTB/SmolLM-135M"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    examples = [
        {
            "prompt": dedent(
                """
        Here is a really long, nested JSON that extracts fields from this sentence:\n\nMy name is Joseph Smith, and I work at Apple. I'm 32 years old, and my interests include kayaking, skiing, snowboarding, and woodworking.\n\n```json\n
        """
            ),
            "lark_grammar_filepath": "./grammars/json.lark",
            "stop_at": ["```"],
            "seed_str": '{\n\t"name":',
        },
        {
            "prompt": dedent(
                "Hello, I am your teacher. Today I will write you a SQL query demonstrating `INNER JOIN` and `LIMIT`. It will translate the following question: Show me the top 5 students with a grade above 0.5\n\n"
            ),
            "lark_grammar_filepath": "./grammars/sql.lark",
            "seed_str": "SELECT",
            "stop_at": [";", "```"],
        },
    ]
    if True:
        max_new_tokens = 200
        max_grammar_corrections = 10
        # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
        start = time.time()
        for example in examples:
            res = guide(
                model,
                tokenizer,
                **example,
                token_healing=True,
                draft_model=guidance.models.Transformers(
                    model_name_or_path, echo=False
                ),
                max_grammar_corrections=max_grammar_corrections,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
            )
            for c in res.correction_log:
                print("Original:")
                print(c.original_pred)
                print(f"Corrected (using {c.type}):")
                print(c.corrected_pred)
                print("------------------------------------------------")
            print("\n\n\n")
            try:
                print(json.dumps(json.loads(res.response), indent=4))
            except:
                print(res.response)
            # print(json.dumps(res.to_list(), indent=4))
            print(time.time() - start)
            break
