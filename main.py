if __name__ == "__main__":
    """
    Open questions:
        - How to align tokenizer representation with Lark lexer representation?
            - The token 'SE' matches to CNAME under the lexer, but we could complete with 'SELECT'

    Proposed flow:
        - Generate unconstrained
        - Run lark parser to fetch grammar prefix breakpoint
        - Using prefix breakpoint, modify the kv cache and re-run generate
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import guidance
    import json
    import time

    from grammar_guide import guide

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    prompt = "Here's a very complex SQL statement (it's so long!!):\n\n"
    max_new_tokens = 30
    # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
    gen = lambda: guide(
        model,
        tokenizer,
        prompt,
        draft_model=guidance.models.Transformers(
            "HuggingFaceTB/SmolLM-135M", echo=False
        ),
        lark_grammar_filepath="./grammars/sql.lark",
        max_grammar_corrections=5,
        max_new_tokens=max_new_tokens,
        stop_at=["```", ";", "All done", "\n"],
        temperature=0.0,
    )
    for _ in range(1):
        start = time.time()
        res = gen()
        print(json.dumps(res.to_list(), indent=4))
        print(time.time() - start)

    # from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
    # from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
    #
    # start = time.time()
    # grammar = IncrementalGrammarConstraint(open("./grammars/json.ebnf").read(), "root", tokenizer)
    # grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
    # input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]
    #
    # output = model.generate(
    #     input_ids,
    #     max_length=30,
    #     logits_processor=[grammar_processor],
    # )
    # # decode output
    # generations = tokenizer.batch_decode(output, skip_special_tokens=True)
    # print(generations)
    # print(time.time() - start)
    # sys.exit()

    # from lark import Discard, Lark, Token
    # from lark.lexer import TerminalDef, PatternRE, PatternStr
    # from typing import List
    # text = ''
    # next_token = "]"
    # parser = Lark(open("./grammars/sql.lark").read(), start="start", parser="lalr")
    # p = parser.parse_interactive(text)
    # p.exhaust_lexer()
    # for t in parser.lex(next_token):
    #     if t.type in p.accepts():
    #         # I guess .feed_token() calls exhaust_lexer() behind the scenes?
    #         p.feed_token(t)
    #     else:
    #         raise ValueError(f"Token {next_token} is not in accept states")
    # accepted: List[TerminalDef] = [parser.get_terminal(a) for a in p.accepts()]
    # # Process states
    # for terminal_def in accepted:
    #     if isinstance(terminal_def.pattern, PatternStr):
    #         ...
    #     elif isinstance(terminal_def.pattern, PatternRE):
    #         ...
    #     else:
    #         raise ValueError(f"Not sure what to do with {terminal_def.pattern}")

    # start = time.time()
    # tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    # model_inputs = tokenizer(
    #     [
    #         prompt
    #     ], padding=True, return_tensors="pt"
    # ).to(device)
    # generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    # print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    # print(time.time() - start)
