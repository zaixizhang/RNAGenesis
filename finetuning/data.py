"""Load user-supplied sequence-level classification CSVs without splitting them."""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pretraining.data import RNATokenizer


class ClassificationDataset(Dataset):
    def __init__(self, path, max_length, num_classes=13, require_labels=True):
        self.path = str(Path(path).resolve())
        self.records = []
        tokenizer = RNATokenizer()
        with open(path, newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            required = {"sequence", "label"} if require_labels else {"sequence"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("%s must have CSV columns %s" % (path, sorted(required)))
            for line, row in enumerate(reader, start=2):
                sequence = row["sequence"]
                if sequence is None or len(sequence) > max_length:
                    raise ValueError("%s:%d sequence exceeds encoder context %d or is missing" % (path, line, max_length))
                try:
                    token_ids = tokenizer.encode(sequence)
                    label = int(row["label"]) if require_labels else None
                    if label is not None and not 0 <= label < num_classes:
                        raise ValueError("label must be in [0, %d]" % (num_classes - 1))
                except (ValueError, TypeError) as error:
                    raise ValueError("%s:%d: %s" % (path, line, error)) from error
                self.records.append({"input_ids": token_ids, "label": label,
                                     "id": row.get("id") or str(line - 2)})
        if not self.records:
            raise ValueError("Empty classification dataset: %s" % path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def classification_collator(examples):
    length = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), length), RNATokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, example in enumerate(examples):
        size = len(example["input_ids"])
        input_ids[index, :size] = torch.tensor(example["input_ids"])
        mask[index, :size] = True
    batch = {"input_ids": input_ids, "attention_mask": mask, "ids": [example["id"] for example in examples]}
    if all(example["label"] is not None for example in examples):
        batch["labels"] = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    return batch
