"""Random-access loading of prepared, one-sequence-per-line RNA text files."""

import hashlib
import json
import os
import random
from array import array
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RNATokenizer:
    # The order is part of the checkpoint format. No start/end tokens are added.
    tokens = ["<pad>", "<unk>", "<mask>", "<cls>", "<eos>"] + list("ACGURYSWKMBDHVN")
    pad_token_id = 0
    mask_token_id = 2
    base_token_ids = [5, 6, 7, 8]

    def __init__(self):
        self.vocab = {token: i for i, token in enumerate(self.tokens)}

    def encode(self, sequence):
        if not sequence:
            raise ValueError("Empty RNA sequence")
        try:
            return [self.vocab[base] for base in sequence]
        except KeyError as exc:
            raise ValueError("Expected prepared uppercase RNA (ACGU and IUPAC codes); invalid character %r" % exc.args[0]) from exc

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        Path(directory, "vocab.json").write_text(json.dumps(self.vocab, indent=2) + "\n", encoding="utf-8")


def sample_seed(seed, epoch, index, purpose):
    key = "%s:%s:%s:%s" % (seed, epoch, index, purpose)
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def corpus_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def index_paths(path, index_dir):
    identity = corpus_identity(path)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return Path(index_dir, key + ".offsets"), Path(index_dir, key + ".json")


def build_line_index(path, index_dir):
    """Cache byte offsets only; never alter, filter, cluster, or copy the corpus."""
    offsets_path, metadata_path = index_paths(path, index_dir)
    identity = corpus_identity(path)
    if offsets_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source"] == identity and offsets_path.stat().st_size == metadata["count"] * 8:
            return offsets_path
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    temporary = offsets_path.with_suffix(".tmp-%s" % os.getpid())
    count, offset = 0, 0
    with open(path, "rb") as source, open(temporary, "wb") as destination:
        while True:
            lines = source.readlines(8 * 1024 * 1024)
            if not lines:
                break
            offsets = array("Q")
            for line in lines:
                offsets.append(offset)
                offset += len(line)
            offsets.tofile(destination)
            count += len(offsets)
    if count == 0:
        temporary.unlink()
        raise ValueError("Training corpus is empty: %s" % path)
    if corpus_identity(path) != identity:
        temporary.unlink()
        raise ValueError("Corpus changed while building the line index")
    os.replace(temporary, offsets_path)
    metadata_path.write_text(json.dumps({"source": identity, "count": count}), encoding="utf-8")
    return offsets_path


class PreparedRNADataset(Dataset):
    def __init__(self, path, index_dir, max_length=1024, seed=42):
        self.path = str(Path(path).resolve())
        self.offsets_path, metadata_path = index_paths(path, index_dir)
        self.count = json.loads(metadata_path.read_text())["count"]
        self.max_length, self.seed, self.epoch = max_length, seed, 0
        self._offsets = self._source = self._pid = None
        self.tokenizer = RNATokenizer()

    def __len__(self):
        return self.count

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_offsets=None, _source=None, _pid=None)
        return state

    def __getitem__(self, index):
        if self._pid != os.getpid():
            if self._source is not None:
                self._source.close()
            self._source = open(self.path, "rb")
            self._offsets = np.memmap(self.offsets_path, dtype=np.uint64, mode="r")
            self._pid = os.getpid()
        self._source.seek(int(self._offsets[index]))
        sequence = self._source.readline().rstrip(b"\r\n").decode("ascii")
        # Crop before tokenization so long transcripts do not create large tensors.
        if len(sequence) > self.max_length:
            rng = random.Random(sample_seed(self.seed, self.epoch, index, "crop"))
            start = rng.randrange(len(sequence) - self.max_length + 1)
            sequence = sequence[start:start + self.max_length]
        try:
            tokens = self.tokenizer.encode(sequence)
        except ValueError as exc:
            raise ValueError("%s, line %s: %s" % (self.path, index + 1, exc)) from exc
        return {"input_ids": tokens, "mask_seed": sample_seed(self.seed, self.epoch, index, "mask")}


class MLMCollator:
    """Independent 30% selection; by default all selected bases become <mask>."""

    def __init__(self, mask_probability=0.30, mask_replace_probability=1.0, random_replace_probability=0.0):
        if not 0 < mask_probability <= 1:
            raise ValueError("mask_probability must be in (0, 1]")
        if min(mask_replace_probability, random_replace_probability) < 0 or mask_replace_probability + random_replace_probability > 1:
            raise ValueError("Replacement probabilities must be nonnegative and sum to at most one")
        self.mask_probability = mask_probability
        self.mask_replace_probability = mask_replace_probability
        self.random_replace_probability = random_replace_probability

    def __call__(self, examples):
        length = max(len(example["input_ids"]) for example in examples)
        input_ids = torch.full((len(examples), length), RNATokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, example in enumerate(examples):
            generator = torch.Generator().manual_seed(example["mask_seed"])
            original = torch.tensor(example["input_ids"], dtype=torch.long)
            size = len(original)
            selected = torch.rand(size, generator=generator) < self.mask_probability
            # Make even very short RNA sequences contribute a finite MLM loss.
            if not selected.any():
                selected[torch.randint(size, (1,), generator=generator)] = True
            corrupted = original.clone()
            replacement = torch.rand(size, generator=generator)
            masked = selected & (replacement < self.mask_replace_probability)
            random_positions = selected & (replacement >= self.mask_replace_probability) & (replacement < self.mask_replace_probability + self.random_replace_probability)
            corrupted[masked] = RNATokenizer.mask_token_id
            random_bases = torch.tensor(RNATokenizer.base_token_ids)[torch.randint(4, (size,), generator=generator)]
            corrupted[random_positions] = random_bases[random_positions]
            input_ids[row, :size] = corrupted
            labels[row, :size] = torch.where(selected, original, -100)
            attention_mask[row, :size] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
