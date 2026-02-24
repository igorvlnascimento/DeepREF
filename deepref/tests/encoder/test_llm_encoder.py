import torch

from deepref.encoder.llm_encoder import LLMEncoder

def get_instruction(task_instruction: str, query: str) -> str:
    return f"Instruct: {task_instruction}\nQuery: {query}"

def test_smollm_llm_encoder():
    llm_encoder = LLMEncoder("HuggingFaceTB/SmolLM-135M-Instruct")
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

    embeddings = llm_encoder(input_texts)

    assert embeddings.shape == (len(input_texts), 576)