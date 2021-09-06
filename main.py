"""
Train and eval functions used in main.py
"""
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import math
import sys
import time
import datetime
from typing import Iterable
from pathlib import Path

import json
import random
import numpy as np
import torch
import wandb

from dataset.evaluator import SmoothedValue, MetricLogger
from model.detr import build_model
from dataset.construction_dataset import build_dataset
from dataset.evaluator import collate_fn, evaluate, save_on_master

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)  # missing some deterministic impl

device = torch.device("cuda:0")


class Args:
    pass


args = Args()
# Postitional encoding
args.position_embedding = "sine"
# CNN Backbone
args.backbone = "resnet50"
args.dilation = None
# Hungarian matcher
args.set_cost_class = 1
args.set_cost_bbox = 5
args.set_cost_giou = 2
# Transformer
args.hidden_dim = 256
args.dropout = 0.1
args.nheads = 8
args.dim_feedforward = 2048
args.enc_layers = 6
args.dec_layers = 6
args.pre_norm = None
# DETR
args.num_queries = 100
args.aux_loss = True  # calculate loss at eache decoder layer
args.masks = True
args.frozen_weights = None
args.bbox_loss_coef = 5
args.mask_loss_coef = 1
args.dice_loss_coef = 1
args.giou_loss_coef = 2
args.eos_coef = 0.1
# Dataset
args.dataset_file = "coco_panoptic"  # construction
args.coco_path = "./data"
args.coco_panoptic_path = "./data"
# Training
args.lr = 1e-4
args.weight_decay = 1e-4
args.lr_backbone = 0  # 0 means frozen backbone
args.batch_size = 1
args.epochs = 2
args.lr_drop = 200
args.clip_max_norm = 0.1

args.output_dir = "out_dir"
args.eval = False


# !mkdir out_dir/panoptic_eval -p
try:
    os.mkdir("out_dir/panoptic_eval")
except Exception as e:
    pass


# set if you plan to log on wandb
ENABLE_WANDB = True
# if set not train from scratch (detre pretrained on COCO)
used_artifact = None # "2_2_attentionfreeze_aux:latest"
# set if starting a new run
wandb_experiment_name = "2_2_1_transf_unfreeze_aux"
# set to None if starting a new run
run_id = None

if ENABLE_WANDB:
    import wandb

    if run_id is not None:
        wandb.init(project="detr", id=run_id, resume="allow")
    else:
        wandb.init(project="detr", name=wandb_experiment_name)

    wandb.config.position_embedding = args.position_embedding
    wandb.config.backbone = args.backbone
    wandb.config.dilation = args.dilation
    wandb.config.set_cost_class = args.set_cost_class
    wandb.config.set_cost_bbox = args.set_cost_bbox
    wandb.config.set_cost_giou = args.set_cost_giou
    wandb.config.hidden_dim = args.hidden_dim
    wandb.config.dropout = args.dropout
    wandb.config.nheads = args.nheads
    wandb.config.dim_feedforward = args.dim_feedforward
    wandb.config.enc_layers = args.enc_layers
    wandb.config.dec_layers = args.dec_layers
    wandb.config.pre_norm = args.pre_norm
    wandb.config.num_queries = args.num_queries
    wandb.config.aux_loss = args.aux_loss
    wandb.config.masks = args.masks
    wandb.config.frozen_weights = args.frozen_weights
    wandb.config.bbox_loss_coef = args.bbox_loss_coef
    wandb.config.mask_loss_coef = args.mask_loss_coef
    wandb.config.dice_loss_coef = args.dice_loss_coef
    wandb.config.giou_loss_coef = args.giou_loss_coef
    wandb.config.eos_coef = args.eos_coef
    wandb.config.lr = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.lr_backbone = args.lr_backbone
    wandb.config.batch_size = args.batch_size
    wandb.config.epochs = args.epochs
    wandb.config.lr_drop = args.lr_drop
    wandb.config.clip_max_norm = args.clip_max_norm


def freeze_attn(model, args):
    for i in range(args.dec_layers):
        for param in model.detr.transformer.decoder.layers[i].self_attn.parameters():
            param.requires_grad = False
        for param in model.detr.transformer.decoder.layers[
            i
        ].multihead_attn.parameters():
            param.requires_grad = False

    for i in range(args.enc_layers):
        for param in model.detr.transformer.encoder.layers[i].self_attn.parameters():
            param.requires_grad = False


def freeze_decoder(model, args):
    for param in model.detr.transformer.decoder.parameters():
        param.requires_grad = False


def freeze_first_layers(model, args):
    for i in range(args.enc_layers // 2):
        for param in model.detr.transformer.encoder.layers[i].parameters():
            param.requires_grad = False

    for i in range(args.dec_layers // 2):
        for param in model.detr.transformer.decoder.layers[i].parameters():
            param.requires_grad = False


def build_pretrained_model(args):
    pre_trained = torch.hub.load(
        "facebookresearch/detr",
        "detr_resnet50_panoptic",
        pretrained=True,
        return_postprocessor=False,
        num_classes=250,
    )
    model, criterion, postprocessors = build_model(args)

    model.detr.backbone.load_state_dict(pre_trained.detr.backbone.state_dict())
    model.detr.bbox_embed.load_state_dict(pre_trained.detr.bbox_embed.state_dict())
    model.detr.query_embed.load_state_dict(pre_trained.detr.query_embed.state_dict())
    model.detr.input_proj.load_state_dict(pre_trained.detr.input_proj.state_dict())
    model.detr.transformer.load_state_dict(pre_trained.detr.transformer.state_dict())

    model.bbox_attention.load_state_dict(pre_trained.bbox_attention.state_dict())
    model.mask_head.load_state_dict(pre_trained.mask_head.state_dict())

    freeze_attn(model, args)

    return model, criterion, postprocessors


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ENABLE_WANDB:
            wandb.log(loss_dict_reduced)
            wandb.log({"loss": loss_value})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train():
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    model, criterion, postprocessors = build_pretrained_model(args)
    model.to(device)

    if ENABLE_WANDB:
        wandb.watch(model)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if ENABLE_WANDB and used_artifact is not None:
        artifact = wandb.use_artifact(used_artifact)
        artifact_dir = artifact.download()
        checkpoint = torch.load(artifact_dir + "/checkpoint.pth")

        model.load_state_dict(checkpoint["model"])
        if run_id is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=1,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=1,
    )

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)

    if args.eval:
        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir
        )
        print(test_stats)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch + 1, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )

        lr_scheduler.step()

        if args.output_dir:
            checkpoint_path = output_dir / "checkpoint.pth"
            save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )
            if ENABLE_WANDB:
                artifact = wandb.Artifact(wandb_experiment_name, type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)

        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        if ENABLE_WANDB:
            wandb.log(test_stats)

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
