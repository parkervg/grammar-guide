# Speculative Grammar Backtracking

This repo is an implementation of the decoding mechanism described in Section 3.2 of [Grammar Prompting for Domain-Specific Language
Generation with Large Language Models](https://arxiv.org/pdf/2305.19234) by [@berlino](https://github.com/berlino).

It is a form of (rather lenient) constrained decoding, and can be used to guide even proprietary, black-box LLM APIs according to some context-free grammar. 


### Features
- Compatible with *any* text generation function
  - OpenAI, Anthropic etc. - as long as you can provide some `generate(prompt: str) -> str` function!
- Efficient re-use of KV cache for all CausalLM Transformer models
  - Optimistic, speculative decoding = no need to manually update to support new tokenizers
- Visualization and logging of grammar corrections 
- Token healing to ensure high probability continuations

## Examples

### With Transformer Models
When using HuggingFace Transformer models, we get an extra speed boost by leveraging efficient caching and backtracking of the KV cache. When a grammar correction is made, we backtrack to the state of the KV cache aligned to the longest prefix that is valid under our Lark context-free grammar.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import guidance 

from speculative_grammar_backtracking import guide, load_parser

model_name_or_path = "HuggingFaceTB/SmolLM-135M"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
parser = load_parser(lark_grammar_filepath="../grammars/json.lark")

res = guide(
  model,
  tokenizer=tokenizer,
  parser=parser,
  prompt="Here is a really long, nested JSON that extracts fields from this sentence:\n\nMy name is Joseph Smith, and I work at Apple. I'm 32 years old, and my interests include kayaking, skiing, snowboarding, and woodworking.\n\n```json\n",
  draft_model=guidance.models.Transformers(
      model_name_or_path, echo=False
  ),
  stop_at=['```'],
  max_new_tokens=20,
  max_grammar_corrections=20,
  temperature=0.0
)
```
![jupyer-visualization](img/jupyter-example.png)

### With General API-based Providers
```python
import os
from openai import OpenAI
import guidance 

from speculative_grammar_backtracking import guide, load_parser

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define our core completion predict function
# This just needs to follow the `fn(s: str) -> str` contract
#   so we can use any black-box API provider.
def openai_generate(s: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "assistant",
                "content": s,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

res = guide(
    model=openai_generate,
    parser=load_parser('./grammars/sql.lark'),
    prompt="Here's a long, complex SQL query: ",
    draft_model=guidance.models.Transformers(
        "HuggingFaceTB/SmolLM-135M", echo=False
    ),
    max_grammar_corrections=20,
    verbose=True,
)
```

As described in the paper, one way many existing libraries achieve this goal is by enforcing some constraint at each decoding timestep. For local models, it is possible to pre-process the logit masks such that this is relatively efficient. However, for closed models (think OpenAI, Anthropic, etc.), this can be 'prohitively expensive', since it would require calling the API at each timestep with the full prompt and valid continuation tokens.

Instead, this library takes an optimistic approach to constrained decoding. Autoregressive language models are only going to get better, and often times the overhead of strict, mask-driven constrained decoding isn't worth it. 

For example, if we want gpt-4o to generate some SQLite query, chances are, it'll generate a valid query without any constraints. 

If there is a mistake, though, we use our grammar to parse the longest prefix that abides by our grammar definition. 

```python
prediction = "SELECT * FROM students WHERE name SIMILAR TO 'Dan%';"
# Oops! `SIMILAR TO` works in PostgreSQL, but not SQLite
prefix, candidates = obtain_correction_pairs(prediction, parser)
print(prefix)
# SELECT * FROM students WHERE name
print(candidates)
# ['IN', '>', '=', 'NOT', 'BETWEEN', 'LIKE', ...] 
```
Once we have a list of candidates, we can use our draft model to select a valid continuation. In the above example, our candidates are fairly simple strings. However, our grammar may define regular expression continuations as well (e.g. `(?:(?:[A-Z]|[a-z])|_)(?:(?:(?:[A-Z]|[a-z])|[0-9]|_))*`).
This is powered by the library [guidance](https://github.com/guidance-ai/guidance).

Once the draft model has selected a valid continuation, we are free to pass the new prefix back to the main lanugage model to complete the prediction.

```python
selected_candidate = choose_candidate(candidates, prefix, draft_model)
print(selected_candidate)
# 'LIKE'
# Now, pass back to the main model to continue its prediction from this new breakpoint
main_model.predict(prefix + selected_candidate)
```


### Benchmarks
The below benchmarks are done on a single A100. They measure the time it takes the respective methods to generate a JSON with exactly 10 string key-value pairs, using [HuggingFaceTB/SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M) and the below prompt.
> Here is a really long JSON object, with 10 keys, using only string values:\n\n```json\n

For most general usecases when using local Transformers models, I highly recommend the library (transformers-CFG)[https://github.com/epfl-dlab/transformers-CFG]! 

| Name                                           |   Time Elapsed |   Generation Time |   Tokens Per Second |
|:-----------------------------------------------|---------------:|------------------:|--------------------:|
| Speculative Grammar Backtracking (Optimized)   |        6.10101 |           5.93353 |            15.5051  |
| Speculative Grammar Backtracking (Naive) |       10.505   |          10.4947  |             9.43337 |
| Transformers CFG                               |        5.67131 |           4.90689 |            22.0099  |