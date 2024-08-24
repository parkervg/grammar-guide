# grammar-guide

| Name             |   Time Elapsed |   Generation Time |   Tokens Per Second |
|:-----------------|---------------:|------------------:|--------------------:|
| SGB              |        6.10101 |           5.93353 |            15.5051  |
| Naive SGB        |       10.505   |          10.4947  |             9.43337 |
| Transformers CFG |        5.67131 |           4.90689 |            22.0099  |

- Compatible with *any* text generation function
  - OpenAI, Anthropic etc. - as long as you can provide some `generate(prompt: str) -> str` function!
- Efficient re-use of KV cache for all CausalLM Transformer models
  - Optimistic, speculative decoding = no need to manually update to support new tokenizers


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
  prompt="Here's a long, complex SQL function:",
  draft_model=guidance.models.Transformers(
      model_name_or_path, echo=False
  ),
  stop_at=['```', ';'],
  max_grammar_corrections=20,
  temperature=0.0
)
```

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