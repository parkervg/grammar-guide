class TokenHealingLogitsProcessor:
    # https://github.com/guidance-ai/guidance/blob/0.0.64/guidance/llms/_transformers.py
    """Token healing.

    When we tokenize the prompt the last token(s) we get are not the last token(s) we would
    have gotten if the prompt + generation was concatented and then tokenized. This
    is not good because it does not align with the pretraining of the model, so
    we "heal" this boundary by backing up as many tokens as needed and then forcing the first tokens
    generated to start with the prefix of the tokens we removed from the prompt. This could
    result in the same tokens at the end of the prompt, or some suffix of the tokens we removed
    could be replaced by a single longer one that crosses the prompt boundary.
    """

    def __init__(self, model, vocab_size, prompt_ids, bias_value=100.0):
        """Build a new TokenHealingLogitsProcessor.

        Note that bias_value is in score space (log-odds normally) and should be
        enough to ensure those tokens are the only ones used.
        """

        # loop backwards through the prompt tokens looking for places where there are possible
        # extensions that cross the prompt boundary
        prefix_str = ""
        self.extension_tokens = []
        for i in range(len(prompt_ids) - 1, max(len(prompt_ids) - 10, -1), -1):
            token_str = model.id_to_token(prompt_ids[i])
            prefix_str = token_str + prefix_str
            try:
                extensions = model.prefix_matches(prefix_str)
            except (
                KeyError
            ):  # this must be a special token outside the vocab, so we assume it does not have any valid extensions
                extensions = []
            self.extension_tokens.append(extensions)
            if i != len(prompt_ids) - 1:
                self.extension_tokens[-1].append(
                    prompt_ids[i]
                )  # add the token used in the input prompt to the list of possible extensions
        self.extension_tokens = self.extension_tokens[::-1]

        # prune off any extension token positions that don't have multiple possible extensions
        found_extensions = False
        for i in range(len(self.extension_tokens)):
            if len(self.extension_tokens[i]) > 1:
                self.extension_tokens = self.extension_tokens[i:]
                found_extensions = True
                break
        if found_extensions:
            self.healed_token_ids = prompt_ids[
                len(prompt_ids) - len(self.extension_tokens) :
            ]
        else:
            self.extension_tokens = []
            self.healed_token_ids = []

        # if we have multiple possible completions past the last token, then biasing is needed
        if len(self.extension_tokens) > 0:
            import torch

            # build a set of masks for each possible extension position
            self.token_masks = []
            for i in range(len(self.extension_tokens)):
                token_mask = torch.zeros(vocab_size)
                token_mask.scatter_(
                    0, torch.tensor(self.extension_tokens[i]), bias_value
                )
                if model.model.device is not None:
                    token_mask = token_mask.to(model.model.device)
                self.token_masks.append(token_mask)

        self.num_extensions = 0

    def __call__(self, input_ids, scores):
        # we only bias the first token generated
        if self.num_extensions >= len(self.extension_tokens):
            return scores
        self.num_extensions += 1

        # check if the last token was from the original prompt (if not then we have already "healed" by choosing a token that crosses the prompt boundary)
        if (
            self.num_extensions > 1
            and input_ids[0][-1] != self.healed_token_ids[self.num_extensions - 2]
        ):
            return scores

        # handle list inputs
        if isinstance(scores, list):
            import torch

            scores = torch.tensor(scores)

        # make only allowed tokens possible
        return scores + self.token_masks[self.num_extensions - 1]
