import re
import torch
import torch.nn.functional as F

from deepref.encoder.sentence_encoder import SentenceEncoder
from deepref.utils.model_registry import ModelRegistry

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = str(BASE_DIR / "tmp")


class LLMEncoder(SentenceEncoder):
    """Concrete sentence encoder backed by a HuggingFace transformer.

    Loads an ``AutoModel`` and ``AutoTokenizer`` from a pretrained checkpoint,
    registers entity special tokens (``[E1]``, ``[/E1]``, ``[E2]``, ``[/E2]``),
    and encodes text via average pooling with L2 normalisation.

    Args:
        model_name: HuggingFace model name or local path.
        max_length: maximum tokenized sequence length.
        padding_side: tokenizer padding side (``'left'`` or ``'right'``).
        device: PyTorch device string (e.g. ``'cpu'``, ``'cuda'``).
        attn_implementation: attention backend passed to ``AutoModel``
            (e.g. ``'eager'``, ``'flash_attention_2'``).
        trainable: if ``False`` (default) all model parameters are frozen.
    """

    def __init__(
        self,
        model_name,
        max_length=512,
        pooling="last_token",
        padding_side="left",
        device="cpu",
        attn_implementation="eager",
        trainable=False,
    ):
        super().__init__(
            model_name,
            max_length=max_length,
            device=device,
            attn_implementation=attn_implementation,
            trainable=trainable,
        )
        self.pooling = pooling
        self.padding_side = padding_side

        # Register the backbone as a proper nn.Module child so that
        # self.parameters() / the optimizer can reach its weights when
        # trainable=True.  Use the sanitized model_name as the attribute
        # key so that:
        #   - multiple trainable encoders in the same module tree each
        #     expose their backbone under a unique, model-specific name;
        #   - torch.save / load_state_dict work correctly across runs
        #     (the key is deterministic given model_name).
        # Frozen models are intentionally excluded so their parameters
        # are never passed to the optimizer.
        # With the backbone as a registered submodule, PyTorch's standard
        # .train() / .eval() propagation reaches it automatically — no
        # manual override is needed.
        if trainable:
            attr = "_backbone_" + re.sub(r"[^a-zA-Z0-9]", "_", model_name)
            self.add_module(attr, self.registry._models[model_name]["model"])

    def average_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average pooling over token embeddings, ignoring padding."""
        last_hidden_states = last_hidden_states.to(torch.float32)
        last_hidden_states_masked = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return F.normalize(embedding, dim=-1)

    def _build_prompt(self, item: dict) -> str:
        """Build the prompt string for one item without tokenizing."""
        token = item["token"]
        h1, h2 = item["h"]["pos"][0], item["h"]["pos"][1]
        t1, t2 = item["t"]["pos"][0], item["t"]["pos"][1]

        if h1 < t1:
            token = token[:t1] + ["[E2]"] + token[t1:t2] + ["[/E2]"] + token[t2:]
            token = token[:h1] + ["[E1]"] + token[h1:h2] + ["[/E1]"] + token[h2:]
        else:
            token = token[:h1] + ["[E1]"] + token[h1:h2] + ["[/E1]"] + token[h2:]
            token = token[:t1] + ["[E2]"] + token[t1:t2] + ["[/E2]"] + token[t2:]
        return (
            f"Instruct: Classify the semantic relation between the two marked entities."
            f"\nQuery: {' '.join(token)}"
        )

    def tokenize(self, item):
        """Tokenize a single item dict.

        Returns:
            Tuple of ``(input_ids, attention_mask)`` tensors of shape ``(1, max_length)``.
        """
        prompt = self._build_prompt(item)
        token_dict = self.registry.tokenize(
            self.model_name,
            prompt,
            max_length=self.max_length,
            padding="max_length",
        )
        return token_dict["input_ids"], token_dict["attention_mask"]

    def tokenize_batch(self, items: list[dict]):
        """Tokenize a list of items in a single tokenizer call.

        Pads to the longest sequence in the batch rather than to ``max_length``,
        which avoids computing attention over thousands of padding tokens when
        sentences are short relative to the configured ``max_length``.

        Returns:
            Tuple of ``(input_ids, attention_mask)`` tensors of shape ``(B, L)``
            where ``L`` is the length of the longest sequence in the batch.
        """
        prompts = [self._build_prompt(item) for item in items]
        token_dict = self.registry.tokenize(
            self.model_name,
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return token_dict["input_ids"], token_dict["attention_mask"]
    
    def last_token_pool(self, 
                        last_hidden_states: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def forward(self, 
                token: torch.Tensor,
                att_mask: torch.Tensor,) -> torch.Tensor:
        """Encode one or more texts and return L2-normalised average-pool embeddings.

        Args:
            texts: a single string or a list of strings to encode.

        Returns:
            Float32 tensor of shape ``(N, hidden_dim)``, L2-normalised.
        """

        outputs = self.registry.run_from_input_ids(self.model_name, token, attention_mask=att_mask)
        if self.pooling == "last_token":
            return self.last_token_pool(outputs.last_hidden_state, att_mask)
        elif self.pooling == "average":
            return self.average_pool(outputs.last_hidden_state, att_mask)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

