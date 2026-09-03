import torch

import loralib as lora


def test_embedding_lora_parameter_initialization():
    layer = lora.Embedding(
        num_embeddings=100,
        embedding_dim=32,
        r=8,
    )

    assert not torch.all(layer.lora_A == 0)
    assert torch.all(layer.lora_B == 0)


def test_linear_lora_parameter_initialization():
    layer = lora.Linear(
        in_features=100,
        out_features=32,
        r=8,
    )

    assert not torch.all(layer.lora_A == 0)
    assert torch.all(layer.lora_B == 0)