"""Dataset / dataloader utilities for hoplas.

CIFAR-10 precomputed-embedding loading (EmbeddingsDatasetWithLabels and
make_loaders_cifar10) was written by Manu (Emanuele Rucci,
rucci.emanuele.work@gmail.com) and is included here with permission, lightly
edited by SHH. MNIST routines to be added below.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LineDataset(Dataset):
    """ Dataset of along line in N-dim space (+ optional noise), 
        with the target being the next point on the line (+ optional noise), 
        with a 'wrap around' boundary condition on the end. 
    """
    def __init__(self, nd=3, npoints=12, noise=0.00, debug=True, len=60_000):
        super().__init__()
        self.nd, self.npoints, self.noise, self.len = nd, npoints, noise, len
        self.line_vals = torch.linspace(-1.0, 1.0, npoints) # points in a line 
        if debug:
            print(f"LineDataset: nd={nd}  npoints={npoints}  noise={noise}")
            print(f"line_vals: {self.line_vals}")

    def __len__(self): # a big to constitute and "epoch"
        return self.len
    
    def __getitem__(self, idx):
        inp = self.noise * torch.randn(self.nd)
        tgt = self.noise * torch.randn(self.nd)
        i = torch.randint(self.npoints, (1,)).item()
        j = (i + 1) % self.npoints     # next, roll/wrap at the end
        inp[0] += self.line_vals[i]
        tgt[0] += self.line_vals[j]
        return {'data': inp, 'label': i}, {'data': tgt, 'label': j}



class EmbeddingsDatasetWithLabels(Dataset):
    """Loads precomputed CLIP embeddings (vision + class-text) from .npz shards.

    Original author: Manu (rucci.emanuele.work@gmail.com).
    """
    def __init__(
        self,
        precomputed_dir,
        split_name="train",   # filename prefix to match: "train" or "test"
        return_label_name=False,
        allow_legacy_flickr_format=False,  # True to also load (N,5,D) text_emb
        text_index=0,                      # which caption to use (legacy only)
    ):
        self.text_embeddings = []
        self.vision_embeddings = []
        self.label_ids = []
        self.label_names = []

        self.return_label_name = return_label_name
        self.allow_legacy_flickr_format = allow_legacy_flickr_format
        self.text_index = text_index

        found = 0
        # Match both sharded (train_shard000.npz) and single-file (train.npz) outputs.
        for fn in os.listdir(precomputed_dir):
            if fn.endswith(".npz") and fn.startswith(split_name):
                found += 1
                path = os.path.join(precomputed_dir, fn)
                data = np.load(path, allow_pickle=True)

                for k in ["vision_emb", "text_emb", "label_ids", "label_names"]:
                    if k not in data:
                        raise RuntimeError(f"Missing key '{k}' in {path}")

                v = data["vision_emb"]   # (N, D)
                t = data["text_emb"]     # (N, D) new OR (N,5,D) legacy
                y = data["label_ids"]    # (N,)
                n = data["label_names"]  # (N,)

                if t.ndim == 2:
                    pass  # new format (N, D)
                elif t.ndim == 3 and self.allow_legacy_flickr_format:
                    pass  # legacy (N, 5, D)
                else:
                    raise RuntimeError(
                        f"Unexpected text_emb shape {t.shape} in {path}. "
                        f"Expected (N,D) or legacy (N,5,D) with allow_legacy_flickr_format=True."
                    )

                self.vision_embeddings.append(v)
                self.text_embeddings.append(t)
                self.label_ids.append(y)
                self.label_names.append(n)

        if found == 0:
            raise RuntimeError(f"No .npz files found in {precomputed_dir} for split '{split_name}'")

        self.vision_embeddings = np.concatenate(self.vision_embeddings, axis=0)  # (N, D)
        self.label_ids = np.concatenate(self.label_ids, axis=0)                  # (N,)
        self.label_names = np.concatenate(self.label_names, axis=0)              # (N,)
        # If shards mix (N,D) and (N,5,D), concatenate errors here -- intentional.
        self.text_embeddings = np.concatenate(self.text_embeddings, axis=0)

        assert len(self.vision_embeddings) == len(self.label_ids) == len(self.label_names)
        assert self.text_embeddings.shape[0] == len(self.vision_embeddings)

        print(f"[Loaded] {len(self)} samples from {precomputed_dir} | text_emb shape={self.text_embeddings.shape}")

    def __len__(self):
        return len(self.vision_embeddings)

    def __getitem__(self, idx):
        t = self.text_embeddings[idx]
        text_emb = t if t.ndim == 1 else t[self.text_index]  # legacy (5,D) -> one caption

        text_emb = torch.as_tensor(text_emb).float()
        vision_emb = torch.as_tensor(self.vision_embeddings[idx]).float()
        label_id = torch.as_tensor(self.label_ids[idx]).long()

        if self.return_label_name:
            return text_emb, vision_emb, label_id, self.label_names[idx]
        return text_emb, vision_emb, label_id


def make_loaders_cifar10(batch_size=256, precomputed_train_dir=None,
                         precomputed_test_dir=None, num_workers=0):
    """Train/test DataLoaders over precomputed CIFAR-10 embeddings.

    Original author: Manu (rucci.emanuele.work@gmail.com).
    """
    ds_train = EmbeddingsDatasetWithLabels(precomputed_train_dir, split_name="train")
    ds_test = EmbeddingsDatasetWithLabels(precomputed_test_dir, split_name="test")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
