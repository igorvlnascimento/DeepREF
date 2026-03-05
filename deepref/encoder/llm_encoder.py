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
        padding_side="left",
        device="cpu",
        attn_implementation="eager",
        trainable=False,
    ):
        super().__init__()
        self.registry = ModelRegistry()
        self.registry.load(model_name,
                           device=device,
                           trainable=trainable,
                           attn_implementation=attn_implementation)

        self.model_name = model_name
        self.max_length = max_length
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

    def tokenize(self, item):
        """Tokenize a string or list of strings.

        Args:
            item: a single string or list of strings.

        Returns:
            dict with ``'input_ids'`` and ``'attention_mask'`` tensors.
        """
        return self.registry.tokenize(
            self.model_name,
            item,
            max_length=self.max_length,
            padding="max_length",
        )

    def forward(self, texts: str | list[str]) -> torch.Tensor:
        """Encode one or more texts and return L2-normalised average-pool embeddings.

        Args:
            texts: a single string or a list of strings to encode.

        Returns:
            Float32 tensor of shape ``(N, hidden_dim)``, L2-normalised.
        """

        model_outputs, batch = self.registry.run(self.model_name, texts)
        return self.average_pool(model_outputs.last_hidden_state, batch["attention_mask"])

