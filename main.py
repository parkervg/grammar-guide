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

    from grammar_guide import guide, load_parser

    model_name_or_path = "HuggingFaceTB/SmolLM-135M"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    parser = load_parser(lark_grammar_filepath="examples/benchmarks/json.lark")

    # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
    res = guide(
        model,
        tokenizer=tokenizer,
        parser=parser,
        # seed_str="""{"name":""",
        prompt=dedent(
            """
        Here is a really long, nested JSON that extracts fields from this sentence:\n\nMy name is Joseph Smith, and I work at Apple. I'm 32 years old, and my interests include kayaking, skiing, snowboarding, and woodworking.\n\n```json\n
        """
        ),
        seed_str='{"name":',
        draft_model=guidance.models.Transformers(model_name_or_path, echo=False),
        max_grammar_corrections=10,
        max_new_tokens=50,
        temperature=0.3,
    )
    print(res.process_time_seconds)

    # from transformers import pipeline

    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device_map="auto",
    #     max_new_tokens=30,
    #     return_full_text=False,
    # )

    # prompt = dedent(
    #     """
    # Here is a really long, nested JSON that extracts fields from this sentence:\n\nMy name is Joseph Smith, and I work at Apple. I'm 32 years old, and my interests include kayaking, skiing, snowboarding, and woodworking.\n\n```json\n
    # """
    # )
    # res = guide(
    #     model=lambda x: pipe(x)[0]["generated_text"].rstrip(prompt),
    #     parser=parser,
    #     seed_str="""{"name":""",
    #     prompt=prompt,
    #     draft_model=guidance.models.Transformers(model_name_or_path, echo=False),
    # )

    # try:
    #     print(json.dumps(json.loads(res.response), indent=4))
    # except:
    #     print(res.response)
