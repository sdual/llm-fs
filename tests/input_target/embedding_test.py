import os

import torch

from llmfs import PROJECT_ROOT
from llmfs.input_target import create_dataloader_v1


def test_embedding():
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)

    print(embedding_layer(torch.tensor([3])))

    print(embedding_layer(input_ids))


def test_more_realistic_embedding():
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    file_path = os.path.join(PROJECT_ROOT, "resources", "texts", "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_lenght = max_length
    pos_embedding_layer = torch.nn.Embedding(context_lenght, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_lenght))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
