#!/usr/bin/env python3
"""Train a convolutional β-VAE on CIFAR-10.
Saves best checkpoint (by val recon loss) to ~/datasets/hoplas_vae/cifar_vae.pt.
Load with: from hoplas.vae import load_cifar_vae; vae = load_cifar_vae()
"""

import argparse
import os
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from hoplas.vae import CIFARVAE, WEIGHTS_DIR


def vae_loss(recon, x, mu, logvar, beta, free_bits=0.0):
    """β-VAE loss: MSE (sum over pixels, mean over batch) + β·KLD with optional free-bits per dim.
    Use reduction='sum'/B (not 'mean') so beta values stay meaningful across latent_dim choices."""
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(0)  # (D,), batch-avg per dim
    if free_bits > 0:
        kld_per_dim = kld_per_dim.clamp(min=free_bits)   # floor per dim: prevents posterior collapse
    return recon_loss + beta * kld_per_dim.sum(), recon_loss, kld_per_dim.sum()


class EMA:
    """Exponential moving average of model weights (stored as float32 regardless of model dtype)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def state_dict(self): return self.shadow


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu") if not args.cpu else torch.device("cpu")
    print(f"device={device}  latent_dim={args.latent_dim}  base_channels={args.base_channels}  beta={args.beta}")

    root = os.path.expanduser("~/datasets/cifar10")
    train_tfms = ([transforms.RandomHorizontalFlip()] if args.hflip else []) + [transforms.ToTensor()]
    train_ds = CIFAR10(root=root, train=True,  download=True, transform=transforms.Compose(train_tfms))
    val_ds   = CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    spec = dict(latent_dim=args.latent_dim, base_channels=args.base_channels,
                image_size=32, in_channels=3, num_groups=32, attn_heads=4)
    model = CIFARVAE(**spec).to(device)
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None

    # weight-decay only on >=2D weights (biases/norms excluded)
    split = lambda ps: ([p for p in ps if p.ndim >= 2], [p for p in ps if p.ndim < 2])
    decay, no_decay = split(list(model.parameters()))
    optimizer = torch.optim.AdamW([{"params": decay,    "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    save_path = args.save_path or os.path.join(WEIGHTS_DIR, "cifar_vae.pt")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_val_recon = float("inf")

    if not args.no_wandb:
        wandb.init(project="hoplas-vae", name=f"cifar10_beta{args.beta}_d{args.latent_dim}", config=vars(args))

    # fixed val batch for reconstruction logging
    val_imgs_fixed = next(iter(val_loader))[0][:16].to(device)

    try:
        for epoch in range(1, args.epochs + 1):
            beta = args.beta * min(1.0, epoch / max(1, args.beta_warmup_epochs))  # linear warmup

            model.train()
            tot_loss = tot_recon = tot_kld = 0.0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
            for x, _ in pbar:
                x = x.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x)
                loss, recon_loss, kld = vae_loss(recon, x, mu, logvar, beta, args.free_bits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema: ema.update(model)
                bs = x.size(0)
                tot_loss += loss.item() * bs; tot_recon += recon_loss.item() * bs; tot_kld += kld.item() * bs
                pbar.set_postfix(recon=f"{recon_loss.item():.3f}", kld=f"{kld.item():.2f}")
            n = len(train_ds)
            avg_loss, avg_recon, avg_kld = tot_loss/n, tot_recon/n, tot_kld/n
            print(f"epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  recon={avg_recon:.4f}  kld={avg_kld:.4f}  beta={beta:.3f}")

            model.eval()
            val_tot = val_recon_tot = val_kld_tot = 0.0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    recon, mu, logvar = model(x)
                    loss, recon_loss, kld = vae_loss(recon, x, mu, logvar, beta, args.free_bits)
                    bs = x.size(0)
                    val_tot += loss.item() * bs; val_recon_tot += recon_loss.item() * bs; val_kld_tot += kld.item() * bs
            nv = len(val_ds)
            val_loss, val_recon, val_kld = val_tot/nv, val_recon_tot/nv, val_kld_tot/nv
            print(f"       val  loss={val_loss:.4f}  recon={val_recon:.4f}  kld={val_kld:.4f}")

            if val_recon < best_val_recon:
                best_val_recon = val_recon
                sd = ema.state_dict() if ema else model.state_dict()
                torch.save({"spec": spec, "state_dict": sd, "epoch": epoch, "val_recon": val_recon}, save_path)
                print(f"  → saved checkpoint (val_recon={val_recon:.4f})")

            if wandb.run is not None:
                log = {"epoch": epoch, "loss": avg_loss, "recon_loss": avg_recon, "kld": avg_kld, "beta": beta,
                       "val_loss": val_loss, "val_recon": val_recon, "val_kld": val_kld}
                if epoch % args.img_every == 0 or epoch == 1:
                    with torch.no_grad():
                        recon_fixed, _, _ = model(val_imgs_fixed)
                        prior_samples = model.decoder(torch.randn(16, args.latent_dim, device=device))
                    # top row: originals; bottom row: reconstructions
                    grid_recon = make_grid(torch.cat([val_imgs_fixed.cpu(), recon_fixed.cpu()]), nrow=8)
                    grid_prior = make_grid(prior_samples.cpu(), nrow=4)
                    log["recon_grid"]    = wandb.Image(grid_recon.permute(1,2,0).numpy())
                    log["prior_samples"] = wandb.Image(grid_prior.permute(1,2,0).numpy())
                wandb.log(log)
    except KeyboardInterrupt:
        print("\ninterrupted — latest best checkpoint preserved")
    finally:
        if wandb.run is not None:
            wandb.finish()


def main():
    p = argparse.ArgumentParser(description="Train a β-VAE on CIFAR-10.")
    p.add_argument("--base-channels",      type=int,   default=128)
    p.add_argument("--batch-size",         type=int,   default=128)
    p.add_argument("--beta",               type=float, default=1.0,   help="KLD weight (β in β-VAE; sweep [0.25, 4])")
    p.add_argument("--beta-warmup-epochs", type=int,   default=10,    help="Linearly ramp β from 0 to target over N epochs")
    p.add_argument("--cpu",                action="store_true")
    p.add_argument("--ema",                type=float, default=0.999, help="EMA decay for weight averaging (0 disables)")
    p.add_argument("--epochs",             type=int,   default=150)
    p.add_argument("--free-bits",          type=float, default=0.5,   help="KLD free-bits floor per dim (0 disables)")
    p.add_argument("--hflip",              action=argparse.BooleanOptionalAction, default=True,
                                           help="Random horizontal flip augmentation")
    p.add_argument("--img-every",          type=int,   default=10,    help="Log recon/prior grids to W&B every N epochs")
    p.add_argument("--latent-dim",         type=int,   default=128)
    p.add_argument("--lr",                 type=float, default=2e-4)
    p.add_argument("--no-wandb",           action="store_true")
    p.add_argument("--save-path",          type=str,   default=None,  help="Override default save path")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--weight-decay",       type=float, default=1e-4)
    args = p.parse_args()
    print("Arguments:", vars(args))
    train(args)


if __name__ == "__main__":
    main()
