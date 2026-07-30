#!/usr/bin/env python3
"""Generate input/recon/transformed wandb image grids for an already-trained ops checkpoint,
and append them to its existing (possibly "finished") wandb run.

Needed for checkpoints trained with --latents-path (train_ops.py skips this logging live,
since the default load_vae(dataset) would decode with the wrong VAE for custom latents).

Usage:
  python scripts/viz_from_checkpoint.py checkpoints/fashion_ph_8_reuse_clsw10.pt \
      --vae-path ~/datasets/hoplas_vae/fashion_vae_clsw10.pt \
      --wandb-run-id x5h2v5sj --project ring-fashion --epoch 1000
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.utils import make_grid
import wandb

from hoplas.inference import load_for_inference, make_class_ordered_images, make_viz_grids
from hoplas.vae import _load_cifar_vae, _pick_device


def _wimg(t):
    arr = make_grid(t, nrow=10).permute(1, 2, 0).numpy()
    return wandb.Image(arr[:, :, 0] if arr.shape[2] == 1 else arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--vae-path", required=True)
    p.add_argument("--wandb-run-id", required=True)
    p.add_argument("--project", default="ring-fashion")
    p.add_argument("--entity", default="drscotthawley")
    p.add_argument("--epoch", type=int, required=True, help="epoch value to log against (for x-axis alignment)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    proj, trans_op, inv_proj, device, dataset = load_for_inference(args.checkpoint, args.device)
    vae = _load_cifar_vae(os.path.expanduser(args.vae_path)).to(device).eval()
    viz_imgs = make_class_ordered_images(dataset=dataset).to(device)

    for m in (proj, trans_op, inv_proj):
        m.eval()
    imgs_in, imgs_recon, imgs_xform, _ = make_viz_grids(vae, proj, trans_op, inv_proj, viz_imgs)

    wandb.init(project=args.project, entity=args.entity, id=args.wandb_run_id, resume="must")
    wandb.log({f"{dataset}_input": _wimg(imgs_in), f"{dataset}_recon": _wimg(imgs_recon),
               f"{dataset}_transformed": _wimg(imgs_xform), "epoch": args.epoch})
    wandb.finish()
    print(f"logged viz grids to run {args.wandb_run_id} (epoch={args.epoch})")


if __name__ == "__main__":
    main()
