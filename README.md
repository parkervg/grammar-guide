# grammar-guide

- Compatible with *any* text generation function
  - OpenAI, Anthropic etc. - as long as you can provide some `generate(prompt: str) -> str` function!
- Efficient re-use of KV cache for all CausalLM Transformer models
  - Optimistic, speculative decoding = no need to manually update to support new tokenizers