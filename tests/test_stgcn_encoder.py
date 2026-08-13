import torch

from src.models.stgcn_encoder import STGCNEncoder, ntu_adjacency


def test_ntu_adjacency_shape_and_self_links():
    a = ntu_adjacency()
    assert a.shape == (3, 25, 25)
    assert torch.allclose(a[0], torch.eye(25))


def test_stgcn_encoder_forward_shape_and_norm():
    model = STGCNEncoder(embed_dim=128)
    x = torch.randn(2, 64, 25, 3)
    with torch.no_grad():
        z = model(x)
    assert z.shape == (2, 128)
    norms = torch.linalg.norm(z, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
