"""Torch-dependent tests: U-Net shapes, diffusion loss/sampling, tag
conditioner, checkpoint round-trip. The whole module skips when torch
isn't installed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# a partially-installed torch raises OSError, not ImportError, so plain
# importorskip isn't enough
try:
    import torch
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"torch not usable: {exc}", allow_module_level=True)


def test_unet_shapes():
    from models.layout.unet import UNet
    net = UNet(in_channels=9, base=32, mults=(1, 2), emb_dim=64)
    x = torch.randn(2, 9, 32, 32)
    t = torch.randint(0, 1000, (2,))
    cond = torch.randn(2, 64)
    assert net(x, t, cond).shape == x.shape


def test_diffusion_loss_and_sample():
    from models.layout.diffusion import GaussianDiffusion
    from models.layout.unet import UNet
    net = UNet(in_channels=9, base=32, mults=(1, 2), emb_dim=64)
    diff = GaussianDiffusion(net, timesteps=100)
    x0 = torch.rand(2, 9, 32, 32) * 2 - 1
    loss = diff.loss(x0, torch.zeros(2, 64))
    assert torch.isfinite(loss)
    sample = diff.ddim_sample((1, 9, 32, 32), torch.zeros(1, 64),
                              torch.zeros(1, 64), steps=3, guidance=1.0)
    assert sample.shape == (1, 9, 32, 32)
    assert torch.isfinite(sample).all()


def test_conditioner_registers_and_encodes():
    from models.conditioning import TagConditioner
    cond = TagConditioner(embed_dim=32, max_tags=8)
    v = cond.encode([{"techbase": 1.0, "hell": 0.5}], device="cpu", register_new=True)
    assert v.shape == (1, 32) and not torch.all(v == 0)
    assert cond.tag_to_index == {"techbase": 0, "hell": 1}
    with pytest.warns(UserWarning):
        unknown = cond.encode([{"never-seen": 1.0}], device="cpu")
    assert torch.all(unknown == 0)


def test_conditioner_cfg_dropout_gives_null():
    from models.conditioning import TagConditioner
    cond = TagConditioner(embed_dim=16, max_tags=4)
    torch.manual_seed(0)
    v = cond.encode([{"a": 1.0}] * 32, device="cpu", cfg_dropout=1.0, register_new=True)
    assert torch.all(v == 0)  # dropout=1 -> always the null condition


def test_checkpoint_round_trip(tmp_path):
    from models.checkpoint import load_checkpoint, save_checkpoint
    from models.conditioning import TagConditioner
    from models.layout.diffusion import GaussianDiffusion
    from models.layout.unet import UNet
    config = {"in_channels": 9, "base": 32, "mults": [1, 2], "emb_dim": 64,
              "timesteps": 100, "max_tags": 8, "resolution": 32}
    diff = GaussianDiffusion(UNet(9, 32, (1, 2), 64), timesteps=100)
    cond = TagConditioner(64, 8)
    cond.encode([{"techbase": 1.0}], device="cpu", register_new=True)
    save_checkpoint(tmp_path / "c.pt", diff, cond, config, step=1)
    _, diff2, cond2, config2 = load_checkpoint(tmp_path / "c.pt")
    assert cond2.tag_to_index == {"techbase": 0}
    assert config2 == config