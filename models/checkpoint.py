"""Checkpoint save/load: model weights + tag vocabulary + architecture
config travel together, because the tag table's row assignment is
meaningless without its tag->index mapping (open vocabulary: the mapping
grows during training and is data, not architecture).
"""
from __future__ import annotations

from pathlib import Path

import torch

from models.conditioning import TagConditioner
from models.layout.diffusion import GaussianDiffusion
from models.layout.unet import UNet


def save_checkpoint(path: Path | str, diffusion: GaussianDiffusion,
                    conditioner: TagConditioner, config: dict, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": diffusion.state_dict(),
        "conditioner": conditioner.state_dict(),
        "tag_to_index": conditioner.tag_to_index,
        "config": config,
        "guidance_convention": diffusion.guidance_convention,
        "step": step,
    }, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path | str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = UNet(in_channels=config["in_channels"], base=config["base"],
                 mults=tuple(config["mults"]), emb_dim=config["emb_dim"])
    diffusion = GaussianDiffusion(model, timesteps=config["timesteps"]).to(device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()
    conditioner = TagConditioner(embed_dim=config["emb_dim"],
                                 max_tags=config["max_tags"], numeric_dim=config.get("numeric_dim", 0)).to(device)
    conditioner.load_checkpoint_state(ckpt)
    return model, diffusion, conditioner, config
