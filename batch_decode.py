import time
from typing import List, Union
from collections.abc import Collection
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

torch.manual_seed(42)
device = torch.device("cpu")


def sample_top_p(probs, p):
    # https://github.com/meta-llama/llama3/blob/main/llama/generation.py
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def check_for_stop_tokens(
    tokens: torch.tensor, stop_at_ids: torch.tensor, stop_at_mask: torch.tensor
):
    # print()
    return torch.tensor([0])


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    @torch.no_grad
    def generate(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: Union[str, List[str]],
        stop_at: Collection[str] = None,
        top_p: float = 0.9,
        temperature: float = 0.6,
        max_new_tokens: int = 32,
        logprobs: bool = False,
    ):
        # stop_at = ["\n", "---", "This is the end"]
        # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if stop_at:
            # tokenizer.padding_side = 'right'
            stop_at_tokenizer_out = tokenizer(
                stop_at, return_tensors="pt", padding=True, add_special_tokens=False
            )
            stop_at_ids = stop_at_tokenizer_out["input_ids"]
            stop_at_mask = stop_at_tokenizer_out["attention_mask"]
        if isinstance(prompt, str):
            prompt = [prompt]
        prompt_tokenizer_out = tokenizer(prompt, return_tensors="pt", padding=True)
        prompt_input_ids = prompt_tokenizer_out["input_ids"]
        prompt_mask = prompt_tokenizer_out["attention_mask"]
        batch_size, longest_prompt_seq = prompt_input_ids.shape
        total_len = min(
            model.config.max_position_embeddings, max_new_tokens + longest_prompt_seq
        )
        tokens = torch.full(
            (
                batch_size,
                total_len,
            ),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for k, t in enumerate(prompt_input_ids):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        past_key_values = DynamicCache()
        prev_pos = 0
        eos_reached = torch.tensor([False] * batch_size, device=device)
        input_text_mask = tokens != tokenizer.pad_token_id
        start = time.time()
        for cur_pos in range(1, total_len):
            # If past_key_values are used, the user can optionally input only the last input_ids
            # (those that don’t have their past key value states given to this model)
            # of shape (batch_size, 1) instead of all input_ids of shape (batch_size, sequence_length).
            # attention_mask = torch.ones_like(tokens[:, :cur_pos])
            # attention_mask[eos_reached, :] = 0
            model_output = model(
                input_ids=tokens[:, prev_pos:cur_pos],
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                # attention_mask=attention_mask
            )
            # len(cache) == model.config.num_hidden_layers
            # Each entry in cache is tuple of (key, value)
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            last_logits = logits[..., -1, :]
            if temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(last_logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            # Below is sort of like a ternary in javascript
            # If condition, then x, otherwise y
            # https://pytorch.org/docs/stable/generated/torch.where.html
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            next_token[eos_reached] = tokenizer.pad_token_id
            # Substitute where eos_reached with pad
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=tokenizer.pad_token_id,
                )
            is_at_stop_tokens = check_for_stop_tokens(tokens, stop_at_ids, stop_at_mask)
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.all(is_at_stop_tokens)
            )
            # Apply relu to get min(0, n)
            num_new_tokens = F.relu(cur_pos - prompt_mask.sum(dim=-1))
            eos_reached[torch.nonzero(num_new_tokens > max_new_tokens)] = True
            prev_pos = cur_pos
            if all(eos_reached):
                break
        decoded_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        print("\n\n---\n\n".join(decoded_text))
        print(f"Sampled tokens in {time.time() - start} seconds")

    # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html
    generate(
        model,
        tokenizer,
        [
            "12345",
            "678910",
            "Hi",
            "Here's some SQL that I will write for you now, it is very very very good SQL:\n```sql\n",
        ],
        max_new_tokens=5,
        stop_at=["```", "All done now."],
        temperature=0.0,
    )
    #
    start = time.time()
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # Most LLMs don't have a pad token by default
    model_inputs = tokenizer(
        [
            "12345",
            "678910",
            "Hi",
            "Here's some SQL that I will write for you now, it is very very very good SQL:\n```sql\n",
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=5)
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    print(time.time() - start)
