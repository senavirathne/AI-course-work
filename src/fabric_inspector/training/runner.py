"""Train/validation-only ANN experiments with inspectable real checkpoints."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fabric_inspector.common import code_hash, environment, now, sha256, write_json
from fabric_inspector.data.images import FabricDataset, preprocessing_spec
from fabric_inspector.data.manifest import load_manifest
from fabric_inspector.evaluation.metrics import select_threshold
from fabric_inspector.models.neural import build_model, choose_device, architecture_table


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def ann_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    scores, labels, ids = [], [], []
    with torch.inference_mode():
        for images, targets, patch_ids in loader:
            logits = model(images.to(device))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.int().tolist())
            ids.extend(patch_ids)
    return np.asarray(scores), np.asarray(labels), ids


def train_run(
    manifest: Path,
    output: Path,
    config: dict,
    project: Path,
    device_name="auto",
    run_id="ann-initial",
    verify_files=True,
) -> dict:
    record_path = output / "experiments" / f"{run_id}.json"
    if record_path.exists():
        raise FileExistsError(f"Run ID already exists: {run_id}")
    split, rows, root = load_manifest(manifest, verify_files=verify_files)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "val"]
    # No test image is loaded anywhere in this function.
    seed_everything(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = choose_device(device_name)
    spec = preprocessing_spec(config["model"]["name"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        FabricDataset(training, root, spec, config.get("augment", True)),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        FabricDataset(validation, root, spec), batch_size=config["batch_size"], num_workers=0
    )
    model = build_model(config["model"], initialise_pretrained=config.get("pretrained", False))
    architecture = architecture_table(model, config["model"])
    model.to(device)
    if config.get("memory_format") == "channels_last":
        model.to(memory_format=torch.channels_last)
    positives = sum(int(r["label"]) for r in training)
    pos_weight = (
        (len(training) - positives) / positives if config.get("weighted_loss", True) else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    if config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4),
        )
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=config.get("weight_decay", 1e-4),
        )
    else:
        raise ValueError("Use adamw or sgd")
    record = {
        "run_id": run_id,
        "started_at": now(),
        "status": "RUNNING",
        "code_sha256": code_hash(project),
        "environment": environment(),
        "device": str(device),
        "split_sha256": split["split_sha256"],
        "provenance": split["provenance"],
        "config": config,
        "hypothesis": config["hypothesis"],
        "pos_weight_from_train_only": pos_weight,
        "architecture": architecture,
        "determinism": "Fixed seeds; deterministic algorithms requested with warnings for unsupported kernels",
    }
    write_json(record_path, record)
    started, best_key, best_state, best_epoch, history, stale = (
        time.perf_counter(),
        None,
        None,
        0,
        [],
        0,
    )
    try:
        for epoch in range(config["epochs"]):
            model.train()
            train_sum = 0.0
            for images, targets, _ in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                train_sum += float(loss.detach()) * len(targets)
            model.eval()
            val_sum, scores, labels = 0.0, [], []
            with torch.inference_mode():
                for images, targets, _ in val_loader:
                    logits = model(images.to(device))
                    val_sum += float(criterion(logits, targets.to(device))) * len(targets)
                    scores.extend(torch.sigmoid(logits).cpu().tolist())
                    labels.extend(targets.int().tolist())
            threshold, measured = select_threshold(labels, scores)
            val_loss = val_sum / len(validation)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_sum / len(training),
                    "val_loss": val_loss,
                    "val_f2": measured["f2"],
                    "threshold": threshold,
                    "val_metrics": measured,
                }
            )
            print(
                f"{run_id} epoch {epoch + 1}/{config['epochs']}: train loss={train_sum / len(training):.4f}, val loss={val_loss:.4f}, val F2={measured['f2']:.4f}",
                flush=True,
            )
            key = (measured["f2"], -val_loss)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= config.get("patience", 6):
                break
        synchronize(device)
        seconds = time.perf_counter() - started
        best = history[best_epoch - 1]
        checkpoint = output / "artifacts" / f"{run_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 1,
            "trained_steps": sum(len(train_loader) for _ in history),
            "run_id": run_id,
            "model": config["model"],
            "preprocessing": spec,
            "split_sha256": split["split_sha256"],
            "provenance": split["provenance"],
            "code_sha256": record["code_sha256"],
            "pretrained_initialisation": config.get("pretrained", False),
        }
        torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint)
        record.update(
            status="COMPLETE",
            finished_at=now(),
            epochs_completed=len(history),
            selected_epoch=best_epoch,
            early_stopped=len(history) < config["epochs"],
            training_seconds=seconds,
            validation_threshold=best["threshold"],
            validation_metrics=best["val_metrics"],
            history=history,
            checkpoint=str(checkpoint.relative_to(output)),
            checkpoint_sha256=sha256(checkpoint),
            outcome="Measured; compare with control before describing an improvement or failure",
            lesson="PENDING evidence-based student interpretation",
        )
        write_json(record_path, record)
        print(
            f"{run_id}: val F2={best['val_f2']:.4f}, {seconds:.1f}s, {split['provenance']}",
            flush=True,
        )
        return record
    except Exception as error:
        record.update(
            status="FAILED_EXECUTION",
            finished_at=now(),
            error=f"{type(error).__name__}: {error}",
            history=history,
            elapsed_seconds=time.perf_counter() - started,
            outcome="Execution failure; not automatically a substantive scientific failed approach",
        )
        write_json(record_path, record)
        raise


def load_checkpoint(path: Path, device_name="cpu", expected_sha=None):
    if expected_sha and sha256(path) != expected_sha:
        raise ValueError("Checkpoint checksum mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata.get("schema_version") != 1 or metadata.get("trained_steps", 0) <= 0:
        raise ValueError("A supported, actually trained checkpoint is required")
    if metadata["preprocessing"] != preprocessing_spec(metadata["model"]["name"]):
        raise ValueError("Checkpoint preprocessing contract mismatch")
    model = build_model(metadata["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = choose_device(device_name)
    return model.to(device).eval(), metadata, device


def default_config() -> dict:
    return {
        "model": {"name": "custom", "depth": 3, "width": 8, "dropout": 0.2},
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 8,
        "patience": 3,
        "cpu_threads": 8,
        "memory_format": "channels_last",
        "seed": 42,
        "augment": True,
        "weighted_loss": True,
        "pretrained": False,
        "hypothesis": "A small spatial CNN can learn local weave discontinuities without excessive capacity",
    }


def search_configs() -> list[dict]:
    """20 predeclared ANN configurations, including optimiser-specific LR ranges."""
    base = default_config()
    configs = [base]
    changes = [
        ({"learning_rate": value}, f"AdamW learning-rate comparison at {value}")
        for value in [1e-4, 3e-4, 3e-3]
    ]
    changes += [({"batch_size": value}, f"Batch-size comparison at {value}") for value in [16, 64]]
    changes += [
        ({"model": dict(base["model"], depth=value)}, f"Actual custom network depth {value} blocks")
        for value in [2, 4]
    ]
    changes += [
        (
            {"optimizer": "sgd", "learning_rate": value},
            f"SGD momentum=.9 with appropriate LR {value}",
        )
        for value in [0.001, 0.01, 0.03, 0.1]
    ]
    changes += [
        (
            {"weighted_loss": False},
            "Unweighted loss may be preferable if weighting causes excess holds",
        ),
        ({"augment": False}, "Orientation augmentation may obscure an acquisition-specific cue"),
        (
            {"model": dict(base["model"], width=16)},
            "More channels may recover subtle local features",
        ),
        (
            {"model": dict(base["model"], dropout=0)},
            "Removing dropout may improve an underfit small model",
        ),
    ]
    for change, hypothesis in changes:
        config = copy.deepcopy(base)
        config.update(change, hypothesis=hypothesis)
        configs.append(config)
    for rate in [1e-4, 3e-4]:
        for augment in [False, True]:
            config = copy.deepcopy(base)
            config.update(
                model={"name": "mobilenet"},
                pretrained=True,
                learning_rate=rate,
                augment=augment,
                hypothesis="Transfer learning may reduce limited-data overfitting; verify tiny-defect resolution",
            )
            configs.append(config)
    return configs
