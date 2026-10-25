"""
Minibatch training and full-dev evaluation for either language model

This file was written with assistance from generative AI.
All AI-generated code was read over and checked for correctness.
"""

import copy
import csv
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch import nn
from tqdm import tqdm

from evaluator import PAD_ID, evaluate
from part1_data import ByteBlockDataset, make_loader
from part1_model import LSTMModel



def build_optimizer(model, config):
    opt = config["optimizer"]
    return torch.optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        betas=tuple(opt["betas"]),
        eps=opt["epsilon"],
        weight_decay=opt["weight_decay"],
        amsgrad=opt["amsgrad"],
        foreach=opt["foreach"],
        fused=opt["fused"],
    )


def evaluate_dev(model, documents, batch_size, device):
    """Dev loss and perplexity from evaluator.py, with dropout and grads off."""
    model.eval()

    def predict_logits(input_ids, attention_mask):
        del attention_mask
        ids = torch.from_numpy(np.ascontiguousarray(input_ids)).to(device)
        with torch.no_grad():
            logits = model(ids)
        return logits.detach().float().cpu().numpy()

    return evaluate(predict_logits, documents, batch_size=batch_size)


def relative_to(path, start):
    return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()


def export_config(config, paths):
    """Write config.yaml beside the checkpoint, with paths relative to that file."""
    exported = copy.deepcopy(config)
    base = paths["artifact_dir"]
    exported["output"]["artifact_dir"] = "."
    exported["output"]["metrics_file"] = relative_to(paths["metrics_file"], base)
    exported["output"]["summary_file"] = relative_to(paths["summary_file"], base)
    exported["output"]["plot_file"] = relative_to(paths["plot_file"], base)
    exported["wandb"]["dir"] = relative_to(paths["wandb_dir"], base)
    base.mkdir(parents=True, exist_ok=True)
    destination = base / "config.yaml"
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(exported, handle, sort_keys=False)
    return destination


def plot_losses(train_steps, train_losses, dev_epochs, dev_losses, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(train_steps, train_losses)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("train loss")
    axes[1].plot(dev_epochs, dev_losses, marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation loss")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def save_checkpoint(model, config, path, best):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model": dict(config["model"]),
        "epoch": best["epoch"],
        "global_step": best["global_step"],
        "dev_loss": best["dev_loss"],
        "dev_ppl": best["dev_ppl"],
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, train_documents, dev_documents, device, paths, model_cls=LSTMModel):
    training_cfg = config["training"]

    # set seed
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model_cls.from_config(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logging.info("parameter count: %d", parameter_count)

    optimizer = build_optimizer(model, config)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    train_dataset = ByteBlockDataset(train_documents)
    loader = make_loader(train_dataset, config, device)

    for path in (
        paths["artifact_dir"],
        paths["metrics_file"].parent,
        paths["summary_file"].parent,
        paths["plot_file"].parent,
        paths["wandb_dir"],
    ):
        path.mkdir(parents=True, exist_ok=True)
    export_config(config, paths)

    clip_norm = training_cfg["gradient_clip_norm"]
    epochs = int(training_cfg["epochs"])
    eval_batch_size = int(config["evaluation"]["batch_size"])

    wandb_cfg = config["wandb"]
    init_kwargs = {
        "project": wandb_cfg["project"],
        "name": wandb_cfg["name"],
        "mode": wandb_cfg["mode"],
        "dir": str(paths["wandb_dir"]),
        "config": copy.deepcopy(config),
    }
    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    wandb.init(**init_kwargs)
    wandb.define_metric("global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/loss", step_metric="global_step")
    wandb.define_metric("dev/loss", step_metric="epoch")
    wandb.define_metric("dev/ppl", step_metric="epoch")

    train_steps = []
    train_losses = []
    dev_epochs = []
    dev_losses = []
    best_ppl = math.inf
    best = None
    global_step = 0
    checkpoint_path = paths["artifact_dir"] / "checkpoint" / "model.pt"

    try:
        with paths["metrics_file"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kind", "step", "epoch", "train_loss", "dev_loss", "dev_ppl"])
            for epoch_idx in range(epochs):
                epoch = epoch_idx + 1
                model.train()
                token_nll = 0.0
                token_count = 0
                for input_ids, targets in tqdm(loader, desc=f"epoch {epoch}"):
                    input_ids = input_ids.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    logits = model(input_ids)

                    loss = loss_fn(logits.transpose(1, 2), targets)
                    optimizer.zero_grad()
                    loss.backward()
                    if clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()

                    global_step += 1
                    n_tokens = int((targets != PAD_ID).sum().item())
                    token_nll += loss.item() * n_tokens
                    token_count += n_tokens
                    train_steps.append(global_step)
                    train_losses.append(loss.item())
                    wandb.log({"train/loss": loss.item(), "global_step": global_step})
                    writer.writerow(["train", global_step, epoch, loss.item(), "", ""])
                    handle.flush()

                epoch_loss = token_nll / token_count
                metrics = evaluate_dev(model, dev_documents, eval_batch_size, device)
                dev_epochs.append(epoch)
                dev_losses.append(metrics["loss"])
                wandb.log(
                    {
                        "epoch": epoch,
                        "dev/loss": metrics["loss"],
                        "dev/ppl": metrics["ppl"],
                    }
                )
                writer.writerow(
                    ["dev", global_step, epoch, "", metrics["loss"], metrics["ppl"]]
                )
                handle.flush()
                logging.info(
                    "completed epoch: %d, train loss: %.4f, dev loss: %.4f, dev ppl: %.4f",
                    epoch,
                    epoch_loss,
                    metrics["loss"],
                    metrics["ppl"],
                )
                if metrics["ppl"] < best_ppl:
                    best_ppl = metrics["ppl"]
                    best = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "dev_loss": metrics["loss"],
                        "dev_ppl": metrics["ppl"],
                        "dev_nll": metrics["nll"],
                        "dev_num_bytes": metrics["num_bytes"],
                    }
                    save_checkpoint(model, config, checkpoint_path, best)
                    logging.info("saved checkpoint %s", checkpoint_path)
    finally:
        wandb.finish()

    summary = {
        "best_epoch": best["epoch"],
        "best_dev_loss": best["dev_loss"],
        "best_dev_ppl": best["dev_ppl"],
        "best_dev_nll": best["dev_nll"],
        "best_dev_num_bytes": best["dev_num_bytes"],
        "global_step": best["global_step"],
        "parameter_count": parameter_count,
        "checkpoint": "checkpoint/model.pt",
    }
    with paths["summary_file"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    plot_losses(train_steps, train_losses, dev_epochs, dev_losses, paths["plot_file"])
    logging.info("best dev ppl: %.4f at epoch %d", best["dev_ppl"], best["epoch"])
    return model, summary
