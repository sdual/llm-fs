import os

import numpy as np
import tiktoken
import torch
from torch import mode, nn

from chap5 import calc_loss_batch, calc_loss_loader, train_val_dataset
from llmfs import PROJECT_ROOT
from llmfs.attention import CausalAttention, SelfAttentionV2
from llmfs.embedding import embed
from llmfs.gpt_download import download_and_load_gpt2
from llmfs.input_target import create_dataloader_v1
from llmfs.model import (DummyGPTModel, DummyLayerNorm, DummyTransformerBlock,
                         ExampleDeepNeuralNetwork, GPTModel)
from llmfs.model.gpt import generate, generate_text_simple


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
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def dummy_gptmy_gpt():
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


def example_deep_neural_net():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    torch.manual_seed(123)

    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True)

    print_gradients(model_without_shortcut, sample_input)


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.0]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {
                  param.grad.abs().mean().item()}")


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
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape: ", logits.shape)
    print(logits)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


def chat05_gpt():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokeinizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokeinizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokeinizer))


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): ",
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}",
                )
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac の GPU
    else:
        device = torch.device("cpu")

    torch.manual_seed(123)
    print(f"Using device: {device}")
    train_loader, val_loader = train_val_dataset()
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1,
    )
    num_epochs = 100
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )
    return train_losses, val_losses, tokens_seen


def random_control():
    model = GPTModel(GPT_CONFIG_124M)
    model.to("cpu")
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def download():
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2")
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {
                         left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, dtype=left.dtype, device=left.device))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        block = gpt.trf_blocks[b]
        block_params = params["blocks"][b]

        q_w, k_w, v_w = np.split(
            block_params["attn"]["c_attn"]["w"], 3, axis=-1)
        block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            block_params["attn"]["c_attn"]["b"], 3, axis=-1)
        block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

        block.att.out_proj.weight = assign(
            block.att.out_proj.weight, block_params["attn"]["c_proj"]["w"].T
        )
        block.att.out_proj.bias = assign(
            block.att.out_proj.bias, block_params["attn"]["c_proj"]["b"]
        )

        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight, block_params["mlp"]["c_fc"]["w"].T
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias, block_params["mlp"]["c_fc"]["b"])
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight, block_params["mlp"]["c_proj"]["w"].T
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias, block_params["mlp"]["c_proj"]["b"]
        )

        block.norm1.scale = assign(
            block.norm1.scale, block_params["ln_1"]["g"])
        block.norm1.shift = assign(
            block.norm1.shift, block_params["ln_1"]["b"])
        block.norm2.scale = assign(
            block.norm2.scale, block_params["ln_2"]["g"])
        block.norm2.shift = assign(
            block.norm2.shift, block_params["ln_2"]["b"])

    gpt.filan_norm.scale = assign(gpt.filan_norm.scale, params["g"])
    gpt.filan_norm.shift = assign(gpt.filan_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def load_model():

    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac の GPU
    else:
        device = torch.device("cpu")

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])

    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2")

    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    load_model()
