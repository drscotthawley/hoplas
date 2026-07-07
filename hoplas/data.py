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
        with the target being a transform of the input point (+ optional noise),
        with a 'wrap around' boundary condition on the end.

        target:
          'ring' (default) — next point on the line: j = (i + 1) % npoints (cyclic transposition T)
          'reflect'        — pitch-class inversion: j = (-i) % npoints (dihedral reflection I, fixes 0)
    """
    def __init__(self, nd=3, npoints=12, noise=0.00, debug=True, len=60_000, target='ring'):
        super().__init__()
        assert target in ('ring', 'reflect'), f"target must be 'ring' or 'reflect', got {target}"
        self.nd, self.npoints, self.noise, self.len, self.target = nd, npoints, noise, len, target
        self.line_vals = torch.linspace(-1.0, 1.0, npoints) # points in a line
        if debug:
            print(f"LineDataset: nd={nd}  npoints={npoints}  noise={noise}  target={target}")
            print(f"line_vals: {self.line_vals}")

    def __len__(self): # a big to constitute and "epoch"
        return self.len

    def __getitem__(self, idx):
        inp = self.noise * torch.randn(self.nd)
        tgt = self.noise * torch.randn(self.nd)
        i = torch.randint(self.npoints, (1,)).item()
        if self.target == 'reflect':
            j = (-i) % self.npoints    # pitch-class inversion x -> -x mod n (fixes 0)
        else:
            j = (i + 1) % self.npoints # 'ring': next, roll/wrap at the end
        inp[0] += self.line_vals[i]
        tgt[0] += self.line_vals[j]
        return {'data': inp, 'label': i}, {'data': tgt, 'label': j}

    def sample_target(self, labels, target_name):
        """Batched target data + labels for a named operation (for multi-head training).
        labels: (B,) source classes i. Returns (data (B,nd), target_labels j (B,)) on labels.device."""
        n = self.npoints
        if target_name == 'ring':
            j = (labels + 1) % n
        elif target_name == 'reflect':
            j = (-labels) % n
        else:
            raise ValueError(f"unknown target_name {target_name!r}")
        out = self.noise * torch.randn(labels.size(0), self.nd, device=labels.device)
        out[:, 0] += self.line_vals.to(labels.device)[j]
        return out, j


class EncodingsDataset(Dataset):
    """Pairs of precomputed VAE encodings, paired by `target` with wraparound:
    'ring' -> (class i, class i+1); 'reflect' -> (class i, class -i mod n).

    Drop-in replacement for LineDataset: returns the same
    ({'data', 'label'}, {'data', 'label'}) pair, where the target is a *random*
    encoding of the paired class. The .pt file is small enough to hold in memory.
    """
    def __init__(self, pt_path, split="train", debug=True, target="ring"):
        super().__init__()
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
        assert target in ("ring", "reflect"), f"target must be 'ring' or 'reflect', got {target}"
        self.target = target
        pt_path = os.path.expanduser(pt_path)
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"{pt_path} not found — run the appropriate encode script first")
        blob = torch.load(pt_path, map_location="cpu")
        self.z = blob[f"{split}_z"].float()           # (N, nd)
        self.labels = blob[f"{split}_labels"].long()  # (N,)
        self.nd = self.z.shape[1]
        self.n_classes = int(self.labels.max().item()) + 1
        self.class_indices = [torch.nonzero(self.labels == c, as_tuple=False).flatten()
                              for c in range(self.n_classes)]
        if debug:
            print(f"EncodingsDataset: {pt_path}  split={split}  N={len(self.z)}  nd={self.nd}  n_classes={self.n_classes}")

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        i = int(self.labels[idx])
        # 'ring': next class i+1 (wrap 9->0); 'reflect': class inversion -i (fixes 0)
        j = (-i) % self.n_classes if self.target == "reflect" else (i + 1) % self.n_classes
        pool = self.class_indices[j]
        tgt_idx = pool[torch.randint(len(pool), (1,)).item()]  # random sample of class j
        return ({'data': self.z[idx], 'label': i},
                {'data': self.z[tgt_idx], 'label': j})

    def sample_target(self, labels, target_name):
        """Batched target data + labels for a named operation (for multi-head training).
        labels: (B,) source classes i. Returns (data (B,nd), target_labels j (B,)) on labels.device;
        data is a random encoding of each target class."""
        n = self.n_classes
        if target_name == 'ring':
            j = (labels + 1) % n
        elif target_name == 'reflect':
            j = (-labels) % n
        else:
            raise ValueError(f"unknown target_name {target_name!r}")
        idx = torch.empty(labels.size(0), dtype=torch.long)
        for b, jj in enumerate(j.tolist()):
            pool = self.class_indices[jj]
            idx[b] = pool[torch.randint(len(pool), (1,)).item()]
        return self.z[idx].to(labels.device), j.to(labels.device)



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


class KGTripleDataset(Dataset):
    """Knowledge-graph triples (head, relation, tail) as integer IDs, for the
    "relation as a learnable group action" view of the ring task.

    Each item is a triple ``(h, r, t)`` (LongTensor of 3): the relation ``r``
    transforms head entity ``h`` into tail entity ``t``. Entities are *not* given
    as coordinates here (unlike LineDataset); the trainer owns a learnable entity
    embedding table indexed by these IDs, and a per-relation operator. SIGReg on the
    entity cloud replaces negative sampling as the anti-collapse force.

    Canonical entity/relation IDs and the standard train/valid/test splits are
    sourced once from PyKEEN's bundled dataset (download + caching only — no PyKEEN
    training/eval machinery). ``create_inverse=True`` appends, for every (h, r, t),
    the inverse triple (t, r + num_base_relations, h) so head prediction is just tail
    prediction under the inverse relation.

    Shared vocabulary: build the *train* split first, then pass its
    ``.entity_to_id`` / ``.relation_to_id`` (via ``from_dataset``) so valid/test use
    identical IDs.
    """

    @staticmethod
    def _load_openke(local_dir, split):
        """Load an OpenKE-format dataset (integer-ID triples) from local_dir. Files:
        entity2id.txt / relation2id.txt (line 1 = count), and {train,valid,test}2id.txt
        (line 1 = count, then `head tail rel` IDs). Returns (num_entities, num_relations,
        base_triples) with base_triples columns reordered to (head, relation, tail)."""
        def _count(fn):
            with open(os.path.join(local_dir, fn)) as f:
                return int(f.readline().split()[0])
        ne, nr = _count("entity2id.txt"), _count("relation2id.txt")
        fn = {"train": "train2id.txt", "valid": "valid2id.txt", "test": "test2id.txt"}[split]
        arr = np.loadtxt(os.path.join(local_dir, fn), skiprows=1, dtype=np.int64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        base = torch.from_numpy(arr[:, [0, 2, 1]]).long()  # OpenKE (h, t, r) -> (h, r, t)
        return ne, nr, base

    def __init__(self, name="WN18RR", split="train", create_inverse=True, debug=True):
        super().__init__()
        self.name, self.split, self.create_inverse = name, split, create_inverse
        # Local OpenKE dataset (for datasets whose pykeen download is broken, e.g. WN18/FB15K).
        data_root = os.path.expanduser(os.environ.get("HOPLAS_DATA", "~/github/hoplas/data"))
        local_dir = os.path.join(data_root, name)
        if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "train2id.txt")):
            self.num_entities, self.num_base_relations, base = self._load_openke(local_dir, split)
            self.entity_to_id = self.relation_to_id = None  # ids already integer; unused downstream
            self.source = f"openke:{local_dir}"
        else:
            from pykeen.datasets import get_dataset  # data sourcing only
            ds = get_dataset(dataset=name)
            factory = {"train": ds.training, "valid": ds.validation, "test": ds.testing}[split]
            self.entity_to_id = ds.training.entity_to_id
            self.relation_to_id = ds.training.relation_to_id
            self.num_entities = ds.training.num_entities
            self.num_base_relations = ds.training.num_relations
            base = factory.mapped_triples.long()  # (T, 3) columns: head, relation, tail
            self.source = "pykeen"
        self.num_relations = self.num_base_relations * (2 if create_inverse else 1)

        if create_inverse:
            inv = base[:, [2, 1, 0]].clone()
            inv[:, 1] += self.num_base_relations
            self.triples = torch.cat([base, inv], dim=0)
        else:
            self.triples = base
        if debug:
            print(f"KGTripleDataset[{name}/{split}]: {len(self.triples)} triples "
                  f"(inverse={create_inverse})  num_entities={self.num_entities}  "
                  f"num_relations={self.num_relations}")

    def __len__(self):
        return self.triples.size(0)

    def __getitem__(self, idx):
        return self.triples[idx]  # LongTensor (3,): (h, r, t)

    def true_tails(self, all_splits):
        """Map (h, r) -> set of true tails, pooled over the given datasets, for
        filtered ranking at eval time. `all_splits`: iterable of KGTripleDataset."""
        from collections import defaultdict
        hr2t = defaultdict(set)
        for ds in all_splits:
            for h, r, t in ds.triples.tolist():
                hr2t[(h, r)].add(t)
        return hr2t
