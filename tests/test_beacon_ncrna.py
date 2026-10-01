"""Small CPU checks of pretrained loading, LoRA, and the supervised pipeline."""

import copy
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from finetuning.beacon_ncrna import RNAGenesisForNcRNAClassification, load_pretrained_encoder
from finetuning.data import ClassificationDataset, classification_collator
from pretraining.model import EncoderConfig, RNAEncoderForMaskedLM
from train_beacon_ncrna import validate_training_config


ROOT = Path(__file__).resolve().parents[1]


def task_config():
    return {"name": "beacon_ncrna", "num_classes": 13, "readout": "mean", "adaptation": "lora",
            "head": {"hidden_sizes": [12], "activation": "gelu", "dropout": 0.1},
            "lora": {"rank": 2, "alpha": 4, "dropout": 0.1, "targets": ["query", "value"]}}


def tiny_encoder():
    return RNAEncoderForMaskedLM(EncoderConfig(hidden_size=16, num_layers=1, num_heads=2,
        intermediate_size=24, max_length=16, hidden_dropout=0.1, attention_dropout=0.1))


def example_batch():
    return classification_collator([
        {"input_ids": [5, 6, 7, 8, 5], "label": 0, "id": "first"},
        {"input_ids": [7, 8], "label": 12, "id": "second"},
    ])


class BeaconTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_pretrained_loading_and_zero_initialized_lora(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = tiny_encoder().eval()
            encoder.save_pretrained(directory)
            restored = load_pretrained_encoder(directory).eval()
            ids = torch.tensor([[5, 6, 7]])
            reference = encoder.encode(ids)
            torch.testing.assert_close(restored.encode(ids), reference)
            full_path = Path(directory, "pretrain-checkpoint.pt")
            torch.save({"config": {"model": asdict(encoder.config)}, "model": encoder.state_dict()}, full_path)
            torch.testing.assert_close(load_pretrained_encoder(full_path).eval().encode(ids), reference)
            model = RNAGenesisForNcRNAClassification(restored, task_config()).eval()
            torch.testing.assert_close(model.encoder.encode(ids), reference)
            torch.testing.assert_close(model.encoder.layers[0].attention.qkv.base.weight,
                                       encoder.layers[0].attention.qkv.weight)

    def test_updates_only_lora_and_prediction_head(self):
        model = RNAGenesisForNcRNAClassification(tiny_encoder(), task_config(), gradient_checkpointing=True)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        trainable = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all(".updates." in name or name.startswith("classifier.") for name in trainable))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        batch = example_batch()
        batch.pop("ids")
        model(**batch)["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optimizer.step()
        after = dict(model.named_parameters())
        for name, expected in frozen.items():
            torch.testing.assert_close(after[name], expected, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(after[name], value) for name, value in trainable.items() if ".updates." in name))
        self.assertTrue(any(not torch.equal(after[name], value) for name, value in trainable.items() if name.startswith("classifier.")))

    def test_pooling_ignores_padding_and_loader_preserves_sequence(self):
        model = RNAGenesisForNcRNAClassification(tiny_encoder(), task_config()).eval()
        with torch.no_grad():
            short = model(torch.tensor([[5, 6, 7]]))["logits"]
            padded = model(torch.tensor([[5, 6, 7, 0, 0, 0]]))["logits"]
        torch.testing.assert_close(short, padded, atol=1e-6, rtol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "data.csv")
            path.write_text("id,sequence,label\nx,ACGU,12\ny,RYN,0\n")
            dataset = ClassificationDataset(path, 16)
            batch = classification_collator([dataset[0], dataset[1]])
            self.assertEqual(batch["input_ids"][0].tolist(), [5, 6, 7, 8])
            self.assertEqual(batch["labels"].tolist(), [12, 0])
            self.assertFalse(batch["input_ids"].eq(2).any())
            with self.assertRaisesRegex(ValueError, "exceeds encoder context"):
                ClassificationDataset(path, 3)

    def test_no_silent_hyperparameter_defaults(self):
        config = json.loads((ROOT / "configs/finetuning/beacon_ncrna.json").read_text())
        self.assertEqual(config["data"], {"train": "", "validation": "", "test": ""})
        with self.assertRaisesRegex(ValueError, "head.hidden_sizes"):
            validate_training_config(config)

    def test_train_resume_and_standalone_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            encoder_dir = directory / "encoder"
            encoder = tiny_encoder()
            encoder.save_pretrained(encoder_dir)
            for name, offset in (("train", 0), ("validation", 1), ("test", 2)):
                with open(directory / (name + ".csv"), "w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["id", "sequence", "label"])
                    # Nine examples exercise unequal microbatches and a short final
                    # accumulation window, without imposing benchmark sample counts.
                    for i in range(9):
                        writer.writerow([name + str(i), ["ACGU", "GGGAAAC", "RYN"][i % 3], (i + offset) % 13])
            config = json.loads((ROOT / "configs/finetuning/beacon_ncrna.json").read_text())
            config["pretrained_encoder"] = str(encoder_dir)
            config["data"] = {name: str(directory / (name + ".csv")) for name in ("train", "validation", "test")}
            config["task"] = task_config()
            config["training"].update(epochs=2, batch_size=2, learning_rate=0.001, weight_decay=0.01,
                optimizer="adamw", scheduler="constant", warmup_ratio=0.0, gradient_accumulation_steps=2)
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "1"
            def run(arguments):
                result = subprocess.run([sys.executable, str(ROOT / "train_beacon_ncrna.py")] + arguments,
                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ("full", "resumed"):
                current = copy.deepcopy(config)
                current["output_dir"] = str(directory / name)
                path = directory / (name + ".json")
                path.write_text(json.dumps(current))
                extra = ["--stop_after_epochs", "1"] if name == "resumed" else []
                run(["--config", str(path), "--device", "cpu"] + extra)
            self.assertFalse((directory / "resumed/test_metrics.json").exists())
            run(["--config", str(directory / "resumed.json"), "--device", "cpu",
                 "--resume", str(directory / "resumed/last.pt")])
            full = torch.load(directory / "full/last.pt", weights_only=True)
            resumed = torch.load(directory / "resumed/last.pt", weights_only=True)
            for name, value in full["model_state"].items():
                torch.testing.assert_close(resumed["model_state"][name], value, rtol=0, atol=0)
            self.assertEqual(full["update"], 6)
            torch.testing.assert_close(full["model_state"]["encoder.embedding.weight"], encoder.embedding.weight, rtol=0, atol=0)
            test_metrics = json.loads((directory / "full/test_metrics.json").read_text())
            self.assertEqual(test_metrics["samples"], 9)
            self.assertEqual(sum(map(sum, test_metrics["confusion_matrix"])), 9)
            best = torch.load(directory / "full/best.pt", weights_only=True)
            self.assertEqual(best["precision"], "fp32")
            # A CUDA/FlashAttention export must remain loadable for CPU inference.
            best["encoder_config"]["attention_backend"] = "flash_attention_2"
            best["precision"] = "bf16"
            torch.save(best, directory / "gpu-export.pt")
            (directory / "predict.csv").write_text("id,sequence\nnew,ACGURYN\n")
            run(["--mode", "predict", "--checkpoint", str(directory / "gpu-export.pt"),
                 "--data", str(directory / "predict.csv"), "--output_dir", str(directory / "predictions"), "--device", "cpu"])
            with open(directory / "predictions/predictions.csv") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(records[0]["id"], "new")
            self.assertAlmostEqual(sum(float(records[0]["probability_%d" % c]) for c in range(13)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
