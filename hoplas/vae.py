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
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")


def load_mnist_vae(device=None):
    """Return Marco's pretrained MNIST VAE (eval mode, on `device`).

    Has `.encoder` (image -> (mu, logvar)) and `.decoder` (latent -> image).
    Fetches code/weights on first call and caches them.
    """
    import torch
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
    device = device or _pick_device()
    return vae.to(device).eval()
