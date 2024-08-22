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