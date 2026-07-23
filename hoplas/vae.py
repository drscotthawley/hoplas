"""Loader for Marco Cassar's pretrained MNIST VAE (third-party, fetched on first use).

Centralizes acquiring the third-party code + weights so both the encode script
and any decode/display script share one loader:

    from hoplas.vae import load_mnist_vae
    vae = load_mnist_vae()      # has .encoder and .decoder; latent dim = 16

This loader is MNIST-specific (Marco's VAE). A CIFAR VAE, when available, will
get its own load_cifar_vae() alongside it.

The VAE class (InspoResNetVAE) lives in marco_submission.py, which we download
into <repo>/third_party/ (gitignored). The weights are pulled from Google Drive
into ~/datasets/hoplas_vae/ (gitignored). We build InspoResNetVAE directly
rather than via SubmissionInterface, so we skip the unused flow model and the
gdown `fuzzy` kwarg that newer gdown removed.

Original VAE code: Marco (Ocramaru/dl_experimentation).
"""

import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(REPO_ROOT, "third_party")
WEIGHTS_DIR = os.path.expanduser("~/datasets/hoplas_vae")

_MARCO_URL = ("https://raw.githubusercontent.com/Ocramaru/dl_experimentation/"
              "33621a796421d9cf82d5cf7d1e49eb48f13f2f68/submissions/marco_submission.py")
_MARCO_PATH = os.path.join(THIRD_PARTY, "marco_submission.py")

_VAE_WEIGHTS_ID = "1rP6yP5yixCI1M7LOrv9v9vJkeYkXnptG"   # Google Drive file id
_VAE_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "downloaded_vae.safetensors")

MNIST_LATENT_DIM = 16
# VAE architecture spec, copied verbatim from marco's SubmissionInterface
# (he pins a commit, so this stays in sync). `act` added at call time.
_VAE_SPEC = dict(latent_shape=(MNIST_LATENT_DIM,), base_channels=16, blocks_per_level=2,
                 groups=4, dropout=0.3, use_skips=True, use_bn=True)


def _ensure_marco():
    """Download third_party/marco_submission.py if missing; put it on sys.path."""
    os.makedirs(THIRD_PARTY, exist_ok=True)
    if not os.path.exists(_MARCO_PATH):
        print(f"Downloading marco_submission.py -> {_MARCO_PATH}")
        # download to a temp path first so a failed fetch leaves nothing behind
        tmp = _MARCO_PATH + ".part"
        subprocess.run(["wget", "-q", "-O", tmp, _MARCO_URL], check=True)
        os.replace(tmp, _MARCO_PATH)
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)


def _pick_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _load_mnist_vae(device=None):
    import torch.nn as nn
    import gdown
    from safetensors.torch import load_file
    _ensure_marco()
    from marco_submission import InspoResNetVAE
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(_VAE_WEIGHTS_PATH):
        gdown.download(id=_VAE_WEIGHTS_ID, output=_VAE_WEIGHTS_PATH, quiet=False)
    vae = InspoResNetVAE(act=nn.GELU, **_VAE_SPEC)
    vae.load_state_dict(load_file(_VAE_WEIGHTS_PATH))
    return vae


# ─── CIFAR-10 β-VAE ───────────────────────────────────────────────────────────

import torch
import torch.nn as nn

CIFAR_LATENT_DIM = 128
_CIFAR_VAE_SPEC = dict(latent_dim=128, base_channels=128, image_size=32, in_channels=3, num_groups=32, attn_heads=4)
_CIFAR_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cifar_vae.pt")
# Same conv β-VAE (arch spec read from each checkpoint), trained by scripts/train_vae.py.
_FASHION_WEIGHTS_PATH   = os.path.join(WEIGHTS_DIR, "fashion_vae.pt")
_MNIST_OURS_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "mnist_vae.pt")   # our trained MNIST VAE (opt-in via "mnist_ours")


class _ResBlock2d(nn.Module):
    """Pre-activation residual block: (GN→SiLU→Conv)×2, last conv zero-init → starts as identity."""
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x): return x + self.net(x)


class _SelfAttn2d(nn.Module):
    """Multi-head self-attention over H×W spatial tokens."""
    def __init__(self, channels, num_heads=4, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H*W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class _CIFAREncoder(nn.Module):
    def __init__(self, in_ch, C, latent_dim, G, heads, sp):
        super().__init__()
        self.stem  = nn.Conv2d(in_ch, C, 3, padding=1)
        self.down1 = nn.Sequential(_ResBlock2d(C, G),   nn.Conv2d(C,   2*C, 3, stride=2, padding=1))
        self.down2 = nn.Sequential(_ResBlock2d(2*C, G), nn.Conv2d(2*C, 2*C, 3, stride=2, padding=1))
        self.mid   = nn.Sequential(_ResBlock2d(2*C, G), _SelfAttn2d(2*C, heads, G), _ResBlock2d(2*C, G))
        self.head  = nn.Linear(2*C * sp * sp, 2 * latent_dim)

    def forward(self, x):
        h = self.mid(self.down2(self.down1(self.stem(x))))
        mu, logvar = self.head(h.flatten(1)).chunk(2, dim=-1)
        return mu, logvar


class _CIFARDecoder(nn.Module):
    def __init__(self, out_ch, C, latent_dim, G, heads, sp):
        super().__init__()
        self.C, self.sp = C, sp
        self.in_proj = nn.Linear(latent_dim, 2*C * sp * sp)
        self.mid  = nn.Sequential(_ResBlock2d(2*C, G), _SelfAttn2d(2*C, heads, G), _ResBlock2d(2*C, G))
        self.up1  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv2d(2*C, 2*C, 3, padding=1), _ResBlock2d(2*C, G))
        self.up2  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv2d(2*C, C, 3, padding=1),   _ResBlock2d(C, G))
        self.out  = nn.Sequential(nn.GroupNorm(G, C), nn.SiLU(), nn.Conv2d(C, out_ch, 3, padding=1), nn.Sigmoid())

    def forward(self, z):
        h = self.in_proj(z).reshape(z.size(0), 2*self.C, self.sp, self.sp)
        return self.out(self.up2(self.up1(self.mid(h))))


class CIFARVAE(nn.Module):
    """Convolutional β-VAE for CIFAR-10 with a global latent vector.

    Interface matches the MNIST VAE:
        mu, logvar = vae.encoder(x)   # x: (B,3,32,32) in [0,1]
        img        = vae.decoder(z)   # z: (B, latent_dim) -> (B,3,32,32) in [0,1]
    """
    def __init__(self, latent_dim=128, base_channels=128, image_size=32,
                 in_channels=3, num_groups=32, attn_heads=4):
        super().__init__()
        sp = image_size // 4   # spatial size at bottleneck (32 → 8)
        C, G = base_channels, num_groups
        self.encoder = _CIFAREncoder(in_channels, C, latent_dim, G, attn_heads, sp)
        self.decoder = _CIFARDecoder(in_channels, C, latent_dim, G, attn_heads, sp)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar.clamp(-30, 20)).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar


def _load_cifar_vae(weights_path=None):
    weights_path = weights_path or _CIFAR_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} not found — run scripts/train_vae.py first")
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    vae = CIFARVAE(**ck.get("spec", _CIFAR_VAE_SPEC))
    vae.load_state_dict(ck["state_dict"])
    return vae


def load_vae(dataset, device=None):
    """Load the pretrained VAE for the given dataset, in eval mode.

    "mnist"      — the borrowed InspoResNetVAE (default; unchanged pipeline)
    "cifar"      — our conv β-VAE (cifar_vae.pt)
    "fashion"    — our conv β-VAE (fashion_vae.pt)
    "mnist_ours" — our conv β-VAE trained on MNIST (mnist_vae.pt); opt-in, does NOT
                   replace "mnist" so existing results stay on the borrowed VAE.
    """
    if dataset == "mnist":
        vae = _load_mnist_vae()
    elif dataset == "cifar":
        vae = _load_cifar_vae()
    elif dataset == "fashion":
        vae = _load_cifar_vae(_FASHION_WEIGHTS_PATH)
    elif dataset == "mnist_ours":
        vae = _load_cifar_vae(_MNIST_OURS_WEIGHTS_PATH)
    else:
        raise ValueError(f"no VAE for dataset: {dataset!r}")
    return vae.to(device or _pick_device()).eval()
