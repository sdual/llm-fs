import os

import tiktoken
import torch
from torch import nn

from llmfs import PROJECT_ROOT
from llmfs.attention import CausalAttention, SelfAttentionV2
from llmfs.embedding import embed
from llmfs.input_target import create_dataloader_v1
from llmfs.model import DummyGPTModel, DummyLayerNorm, DummyTransformerBlock


def encoding():
    file_path = os.path.join(PROJECT_ROOT, "resources",
                             "texts", "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

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

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange((context_length)))
    print(pos_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)


def embedding():
    embed()


def attention():
    vocab_size = 50257
    output_dim = 3
    text = "Your journey starts with one step"

    max_length = 6
    dataloader = create_dataloader_v1(
        text, batch_size=1, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)

    inputs, targets = next(data_iter)
    print(inputs)

    vocab_size = 50257
    output_dim = 3
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings)

    query = token_embeddings[0][1]
    print(query)
    attn_scores_2 = torch.empty(token_embeddings[0].shape[0])
    for i, x_i in enumerate(token_embeddings[0]):
        attn_scores_2[i] = torch.dot(x_i, query)

    print(torch.softmax(attn_scores_2, dim=0))

    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(token_embeddings[0]):
        context_vec_2 += attn_scores_2[i] * x_i

    print(context_vec_2)


def all_attetions():
    vocab_size = 50257
    output_dim = 3
    text = "Your journey starts with one step"

    max_length = 6
    dataloader = create_dataloader_v1(
        text, batch_size=1, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)

    inputs, targets = next(data_iter)

    vocab_size = 50257
    output_dim = 3
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings)

    #    attn_scores = torch.empty(6, 6)
    #    for i, x_i in enumerate(token_embeddings[0]):
    #        for j, x_j in enumerate(token_embeddings[0]):
    #            attn_scores[i, j] = torch.dot(x_i, x_j)
    #
    attn_scores = token_embeddings[0] @ token_embeddings[0].T
    print(attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)

    all_context_vecs = attn_weights @ token_embeddings[0]
    print(all_context_vecs)


def weight_attentions():
    vocab_size = 50257
    output_dim = 3
    text = "Your journey starts with one step"

    max_length = 6
    dataloader = create_dataloader_v1(
        text, batch_size=1, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)

    raw_inputs, targets = next(data_iter)

    vocab_size = 50257
    output_dim = 3
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(raw_inputs)
    inputs = token_embeddings[0]

    x_2 = inputs[1]
    print(x_2)

    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)

    keys = inputs @ W_key
    values = inputs @ W_value
    print(keys)
    print("keys.shape:", keys.shape)
    print("value.shape:", values.shape)

    attn_score_2 = query_2 @ keys.T
    print(attn_score_2)

    d_k = keys.shape[-1]
    attn_weights_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
    print(attn_weights_2)


def self_attention():
    vocab_size = 50257
    output_dim = 3
    text = "Your journey starts with one step"

    max_length = 6
    dataloader = create_dataloader_v1(
        text, batch_size=1, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)

    raw_inputs, targets = next(data_iter)

    vocab_size = 50257
    output_dim = 3
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(raw_inputs)
    inputs = token_embeddings[0]
    print(inputs)

    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    print(sa_v2(inputs))


def causal_attention():
    vocab_size = 50257
    output_dim = 3
    text = "Your journey starts with one step"

    max_length = 6
    dataloader = create_dataloader_v1(
        text, batch_size=1, max_length=max_length, stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)

    raw_inputs, targets = next(data_iter)

    vocab_size = 50257
    output_dim = 3
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(raw_inputs)
    inputs = token_embeddings[0]
    print(inputs)

    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(789)
    context_length = 6
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    print(ca(token_embeddings))


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def gpt():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    batch = torch.stack(batch, dim=0)
    print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape: ", logits.shape)
    print(logits)


def normalize():
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    print("Normalzed layer outputs:\n", out_norm)


if __name__ == "__main__":
    normalize()
