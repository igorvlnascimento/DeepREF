import pytest

from deepref.encoder.sentence_encoder import SentenceEncoder
from deepref.module.nn.mlp import MLP

MODEL_NAMES = [
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google-bert/bert-base-uncased"
]

def get_instruction(task_instruction: str, query: str) -> str:
    return f"Instruct: {task_instruction}\nQuery: {query}"

def build_text():
    task = "Given a question, retrieve passages that answer the question"
    queries = [
        get_instruction(task, "How do neural networks learn patterns from examples?"),
    ]

    # No instruction is required for documents corpus
    documents = [
        "Deep learning models adjust their weights through backpropagation, using gradient descent to minimize error on training data and improve predictions over time.",
        "Market prices are determined by the relationship between how much people want to buy a product and how much is available for sale, with scarcity driving prices up and abundance driving them down.",
    ]
    input_texts = queries + documents

    return input_texts


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_mlp(model_name):
    sentence_encoder = SentenceEncoder(model_name)
    mlp = MLP(sentence_encoder)
    
    input_texts = build_text()

    embeddings = sentence_encoder(input_texts)
    x = mlp(embeddings)

    assert x.shape == (len(input_texts), sentence_encoder.model.config.hidden_size//2**mlp.num_layers)