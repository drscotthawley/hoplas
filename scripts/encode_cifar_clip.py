#! /usr/bin/env python3
# Original: precompute_cifar10_embeddings_with_labels.py
# Author: Manu, rucci.emanuele.work@gmail.com
# Used with permission, renamed to encode_cifar10.py by SHH and modified 

import os
import argparse
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import open_clip
from torchvision.datasets import CIFAR10


# -----------------------------------
# ------ DATASET UTILS FUNCTIONS ----
# -----------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _save_npz(
    out_path: str,
    text_emb: np.ndarray,         # (N, D)
    vision_emb: np.ndarray,       # (N, D)
    fns: List[str],               # len N
    label_ids: np.ndarray,        # (N,)
    label_names: List[str],       # len N
    meta: Dict[str, Any],
) -> None:
    fns_arr = np.array(fns, dtype=object)
    label_names_arr = np.array(label_names, dtype=object)

    np.savez_compressed(
        out_path,
        text_emb=text_emb,
        vision_emb=vision_emb,
        fns=fns_arr,
        label_ids=label_ids.astype(np.int64),
        label_names=label_names_arr,
        meta=np.array([meta], dtype=object),
    )


# -----------------------------------
# -------- MODEL BUILDERS -----------
# -----------------------------------

def build_clip(device: torch.device, model_name: str, pretrained: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


# -----------------------------------
# --------- MAIN PRECOMPUTE ---------
# -----------------------------------

@torch.no_grad()
def precompute_cifar10_split(
    *,
    split_name: str,                 # "train" or "test"
    dataset: CIFAR10,
    clip_model,
    preprocess,
    tokenizer,
    device: torch.device,
    out_dir: str,
    normalize: bool,
    clip_model_name: str,            # recorded into meta
    shard_size: Optional[int] = None,   # images per shard; None -> single file
    class_text_template: str = "{label}",  # e.g. "{label}" OR "a photo of a {label}"
) -> None:
    """
    For each CIFAR-10 image:
      - vision_emb[i] = CLIP image embedding (D,)
      - text_emb[i]   = CLIP text embedding of the class name (D,)
      - label_ids[i]  = ground-truth class id (0..9)
      - label_names[i]= class name (string)

    Output shard NPZ keys:
      text_emb:   (N, D)
      vision_emb: (N, D)
      fns:        (N,)
      label_ids:  (N,)
      label_names:(N,)
      meta:       dict
    """

    _ensure_dir(out_dir)
    class_names = list(dataset.classes)

    # Encode the (few) class-name texts once, then index by label_id in the loop.
    text_strs = [class_text_template.format(label=name) for name in class_names]
    tokens = tokenizer(text_strs).to(device)                 # (num_classes, ...)
    class_text_embs = clip_model.encode_text(tokens)         # (num_classes, D)
    if normalize:
        class_text_embs = F.normalize(class_text_embs, dim=-1)
    class_text_embs = _to_numpy(class_text_embs)             # (num_classes, D)

    shard_fns: List[str] = []
    shard_vis: List[np.ndarray] = []
    shard_text: List[np.ndarray] = []
    shard_label_ids: List[int] = []
    shard_label_names: List[str] = []

    shard_idx = 0
    num_in_shard = 0

    def flush_shard():
        nonlocal shard_idx, shard_fns, shard_vis, shard_text
        nonlocal shard_label_ids, shard_label_names, num_in_shard

        if len(shard_fns) == 0:
            return

        vis_np = np.stack(shard_vis, axis=0)               # (N, D)
        text_np = np.stack(shard_text, axis=0)             # (N, D)
        label_ids_np = np.array(shard_label_ids, np.int64) # (N,)

        meta = {
            "dataset_name": "cifar10",
            "split": split_name,
            "num_items": int(vis_np.shape[0]),
            "embedding_dim": int(vis_np.shape[1]),
            "text_per_image": 1,
            "text_template": class_text_template,
            "normalization": bool(normalize),
            "label_space": "CIFAR-10",
            "clip_model": clip_model_name,
        }

        if shard_size is None:
            out_path = os.path.join(out_dir, f"{split_name}.npz")
        else:
            out_path = os.path.join(out_dir, f"{split_name}_shard{shard_idx:03d}.npz")

        _save_npz(
            out_path=out_path,
            text_emb=text_np,
            vision_emb=vis_np,
            fns=shard_fns,
            label_ids=label_ids_np,
            label_names=shard_label_names,
            meta=meta,
        )

        print(f"[SAVED] {out_path} | N={vis_np.shape[0]} | vision={vis_np.shape} | text={text_np.shape}")

        shard_idx += 1
        shard_fns = []
        shard_vis = []
        shard_text = []
        shard_label_ids = []
        shard_label_names = []
        num_in_shard = 0

    print(f"\n=== Precomputing CIFAR-10 CLIP embeddings (image + class-text): {split_name} ===")

    for i in tqdm(range(len(dataset)), desc=f"{split_name} images", total=len(dataset)):
        img, label_id = dataset[i]  # img is PIL.Image, label_id is int
        label_id = int(label_id)
        label_name = class_names[label_id]

        # ---- image emb ----
        img_tensor = preprocess(img).unsqueeze(0).to(device)     # (1,3,H,W)
        vision_emb = clip_model.encode_image(img_tensor)         # (1, D)
        if normalize:
            vision_emb = F.normalize(vision_emb, dim=-1)

        shard_vis.append(_to_numpy(vision_emb)[0])       # (D,)
        shard_text.append(class_text_embs[label_id])     # (D,) precomputed
        shard_label_ids.append(label_id)
        shard_label_names.append(label_name)
        shard_fns.append(f"{split_name}_{i:08d}")

        num_in_shard += 1
        if shard_size is not None and num_in_shard >= shard_size:
            flush_shard()

    flush_shard()

    print(f"\n✅ Done. Saved under: {out_dir}")
    print("NPZ keys: text_emb, vision_emb, fns, label_ids, label_names, meta")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar_root", type=str, default="~/datasets/cifar10", help="Where CIFAR-10 will be stored.")
    parser.add_argument("--out_dir", type=str, default="~/datasets/precomputed_cifar10_text_vision_labels", help="Output dir.")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing.")
    parser.add_argument("--device", type=str, default="auto", help="auto / cuda:0 / mps / cpu / etc.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize CLIP embeddings.")
    parser.add_argument("--shard_size", type=int, default=4096, help="Images per shard. Use 0 to disable sharding.")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="open_clip model name.")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k", help="open_clip pretrained tag.")
    parser.add_argument(
        "--class_text_template",
        type=str,
        default="{label}",
        help="Template for class text. Examples: '{label}' or 'a photo of a {label}'."
    )
    args = parser.parse_args()

    if args.device == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev_str = args.device
    device = torch.device(dev_str)

    cifar_root = os.path.expanduser(args.cifar_root)
    out_root = os.path.expanduser(args.out_dir)
    _ensure_dir(out_root)

    # dataset
    train = (args.split == "train")
    dataset = CIFAR10(root=cifar_root, train=train, download=args.download)

    # clip
    clip_model, preprocess, tokenizer = build_clip(
        device=device,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
    )

    shard_size = None if args.shard_size == 0 else int(args.shard_size)

    run_name = f"cifar10_{args.split}_{args.clip_model}_{args.clip_pretrained}".replace("/", "_")
    out_dir = os.path.join(out_root, run_name)
    _ensure_dir(out_dir)

    precompute_cifar10_split(
        split_name=args.split,
        dataset=dataset,
        clip_model=clip_model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        out_dir=out_dir,
        normalize=bool(args.normalize),
        clip_model_name=args.clip_model,
        shard_size=shard_size,
        class_text_template=args.class_text_template,
    )


if __name__ == "__main__":
    main()