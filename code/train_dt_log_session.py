#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_dt_log_session.py

Offline training for DecisionTransformerRec on log_session using KRMBSeqReaderDT.
"""

import argparse
import os
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

EVAL_TOPK = [5, 10, 20]
MAIN_METRIC_K = 5

from tqdm import tqdm

import utils
from reader.KRMBSeqReaderDT import KRMBSeqReaderDT
from model.DecisionTransformerRec import DecisionTransformerRec


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id; <0 for CPU')
    parser.add_argument('--epoch', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Train batch size')
    parser.add_argument('--val_batch_size', type=int, default=512, help='Validation batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=5,
        help='main metric epoch ; <=0  early stopping'
    )

    parser = DecisionTransformerRec.parse_model_args(parser)
    parser = KRMBSeqReaderDT.parse_data_args(parser)

    args = parser.parse_args()
    return args


def build_device(args):
    """ device,  args.device"""
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    print(f"[Info] Using device: {device}")
    return device


def do_eval(model, reader, args):
    """
 current reader.phase  eval: 
 : 
 avg_loss: float
 metrics: dict, 
 - 'top1_acc': float
 - 'recall': {k: float}
 - 'ndcg': {k: float}
 - 'eval_k_list': [k1, k2, ...]
 """
    model.eval()

    eval_loader = DataLoader(
        reader,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=getattr(reader, "n_worker", 0),
    )

    losses = []
    correct_top1 = 0
    total = 0

    k_list = getattr(args, "eval_topk", None)
    if k_list is None:
        k_list = EVAL_TOPK
    k_list = sorted(int(k) for k in k_list)
    max_k = max(k_list)

    sum_recall = {k: 0.0 for k in k_list}
    sum_ndcg = {k: 0.0 for k in k_list}

    with torch.no_grad():
        pbar = tqdm(total=len(reader), desc=f"Validating ({reader.phase})")
        for batch_data in eval_loader:
            wrapped_batch = utils.wrap_batch(batch_data, device=args.device)

            if wrapped_batch['user_id'].shape[0] == 0:
                pbar.update(1)
                continue

            out_dict = model.do_forward_and_loss(wrapped_batch)
            loss = out_dict['loss']
            losses.append(loss.item())

            preds = out_dict['preds']          # (B, V)
            target_items = wrapped_batch['item_id'].view(-1).long()  # (B,)
            B = target_items.shape[0]

            pred_items = preds.argmax(dim=-1)  # (B,)
            correct_top1 += (pred_items == target_items).sum().item()
            total += B

            topk_scores, topk_indices = preds.topk(max_k, dim=-1)  # [B, max_k]

            target_expanded = target_items.view(-1, 1).expand_as(topk_indices)
            hit_matrix = (topk_indices == target_expanded)  # [B, max_k] bool

            positions = torch.arange(max_k, device=preds.device).view(1, -1).expand_as(topk_indices)
            pos_hit = torch.where(hit_matrix, positions, torch.full_like(positions, max_k))
            min_pos, _ = pos_hit.min(dim=1)  # [B]

            for k in k_list:
                hit_in_k = (min_pos < k)  # [B],  top-k  True
                sum_recall[k] += hit_in_k.float().sum().item()

                if hit_in_k.any():
                    ranks = min_pos[hit_in_k]
                    dcg = 1.0 / torch.log2(ranks.float() + 2.0)
                    sum_ndcg[k] += dcg.sum().item()

            pbar.update(wrapped_batch['user_id'].shape[0])

        pbar.close()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    if total > 0:
        top1_acc = correct_top1 / total
        recall = {k: (sum_recall[k] / total) for k in k_list}
        ndcg = {k: (sum_ndcg[k] / total) for k in k_list}
    else:
        top1_acc = 0.0
        recall = {k: 0.0 for k in k_list}
        ndcg = {k: 0.0 for k in k_list}

    metrics = {
        "top1_acc": top1_acc,
        "recall": recall,
        "ndcg": ndcg,
        "eval_k_list": k_list,
    }
    return avg_loss, metrics



def print_rtg_percentiles(reader, args, phase="train", max_batches=200):
    """
  phase(default train),  batch, 
  RTG (raw & / rtg_scale). 

  Reader ,  key  'rtg' , 
  attention mask filter padding. 
 """
    if hasattr(reader, "set_phase"):
        reader.set_phase(phase)

    data_loader = DataLoader(
        reader,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=getattr(reader, "n_worker", 0),
    )

    all_rtg_raw = []

    for b_idx, batch_data in enumerate(data_loader):
        rtg_keys = [k for k in batch_data.keys() if "rtg" in k.lower()]
        if not rtg_keys:
            continue

        attn_keys = [k for k in batch_data.keys()
                     if "attn" in k.lower() and "mask" in k.lower()]
        attn = None
        if attn_keys:
            attn = batch_data[attn_keys[0]].numpy().astype(np.float32)

        for k in rtg_keys:
            v = batch_data[k].numpy().astype(np.float32)
            if attn is not None and attn.shape == v.shape:
                mask = attn > 0
                v = v[mask]
            all_rtg_raw.append(v.reshape(-1))

        if b_idx + 1 >= max_batches:
            break

    if not all_rtg_raw:
        print("[Warn]  batch_data  'rtg' ,  RTG . ")
        return

    rtg_raw = np.concatenate(all_rtg_raw, axis=0)
    if rtg_raw.size == 0:
        print("[Warn]  RTG , . ")
        return

    percentiles = [1, 25, 50, 75, 90, 95, 99]

    print("====== RTG raw  ======")
    for p in percentiles:
        val = float(np.percentile(rtg_raw, p))
        print(f"  p{p:>3} = {val:.4f}")
    print(f"  mean = {rtg_raw.mean():.4f}")



def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args()

    utils.set_random_seed(args.seed)

    device = build_device(args)

    print("[Info] Initializing reader (KRMBSeqReaderDT)...")
    reader = KRMBSeqReaderDT(args)
    stats = reader.get_statistics()
    print("[Info] Data statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    try:
        print("\n[Info] Inspecting RTG distribution on TRAIN phase...")
        print_rtg_percentiles(reader, args, phase="train", max_batches=200)
    except Exception as e:
        print(f"[Warn] RTG , : {e}")

    print("[Info] Initializing DecisionTransformerRec...")
    model = DecisionTransformerRec(args, stats, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer  #  BaseModel.save_checkpoint()  optimizer

    print(f"[Info] Model will be saved to: {args.model_path}")

    print("\n[Eval] Validation before training...")
    reader.set_phase("val")
    val_loss, val_metrics = do_eval(model, reader, args)
    eval_k_list = val_metrics.get("eval_k_list", EVAL_TOPK)
    recall_dict = val_metrics["recall"]
    ndcg_dict = val_metrics["ndcg"]
    top1_acc = val_metrics.get("top1_acc", 0.0)

    recall_str = ", ".join([f"Recall@{k}={recall_dict.get(k, 0.0):.4f}" for k in eval_k_list])
    ndcg_str = ", ".join([f"NDCG@{k}={ndcg_dict.get(k, 0.0):.4f}" for k in eval_k_list])
    print(
        f"[Eval] Before training: loss={val_loss:.4f}, "
        f"{recall_str}, {ndcg_str}, top1_acc={top1_acc:.4f}"
    )

    best_val_loss = val_loss
    best_recall_main = recall_dict.get(MAIN_METRIC_K, 0.0)
    best_epoch = 0
    no_improve = 0

    for epo in range(1, args.epoch + 1):
        print(f"\n===== Epoch {epo}/{args.epoch} =====")
        model.train()
        reader.set_phase("train")

        train_loader = DataLoader(
            reader,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=getattr(reader, "n_worker", 0),
        )

        t0 = time()
        pbar = tqdm(total=len(reader), desc=f"Training (epoch {epo})")
        step_losses = []

        for batch_data in train_loader:
            wrapped_batch = utils.wrap_batch(batch_data, device=device)

            if wrapped_batch['user_id'].shape[0] == 0:
                pbar.update(1)
                continue

            optimizer.zero_grad()

            out_dict = model.do_forward_and_loss(wrapped_batch)
            loss = out_dict['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            step_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            pbar.update(wrapped_batch['user_id'].shape[0])

        pbar.close()
        avg_train_loss = float(np.mean(step_losses)) if step_losses else 0.0
        used_time = time() - t0
        print(f"[Train] Epoch {epo}: avg_loss={avg_train_loss:.4f}, time={used_time:.2f}s")

        reader.set_phase("val")
        val_loss, val_metrics = do_eval(model, reader, args)

        eval_k_list = val_metrics.get("eval_k_list", EVAL_TOPK)
        recall_dict = val_metrics["recall"]
        ndcg_dict = val_metrics["ndcg"]
        top1_acc = val_metrics.get("top1_acc", 0.0)

        recall_str = ", ".join(
            [f"Recall@{k}={recall_dict.get(k, 0.0):.4f}" for k in eval_k_list]
        )
        ndcg_str = ", ".join(
            [f"NDCG@{k}={ndcg_dict.get(k, 0.0):.4f}" for k in eval_k_list]
        )
        print(
            f"[Eval] Epoch {epo}: val_loss={val_loss:.4f}, "
            f"{recall_str}, {ndcg_str}, top1_acc={top1_acc:.4f}"
        )

        cur_recall_main = recall_dict.get(MAIN_METRIC_K, 0.0)
        cur_ndcg_main = ndcg_dict.get(MAIN_METRIC_K, 0.0)
        improved = cur_recall_main > best_recall_main + 1e-6

        if improved:
            best_val_loss = val_loss
            best_recall_main = cur_recall_main
            best_epoch = epo
            no_improve = 0

            try:
                model.save_checkpoint()
                print(
                    f"[Save] New best model at epoch {epo}, "
                    f"val_loss={val_loss:.4f}, "
                    f"Recall@{MAIN_METRIC_K}={cur_recall_main:.4f}, "
                    f"NDCG@{MAIN_METRIC_K}={cur_ndcg_main:.4f} -> {args.model_path}"
                )
            except Exception as e:
                print(f"[Warn] model.save_checkpoint() failed: {e}")
                try:
                    ckpt_path = args.model_path
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "stats": stats,
                    }
                    if not (
                        ckpt_path.endswith(".pt")
                        or ckpt_path.endswith(".pth")
                        or ckpt_path.endswith(".checkpoint")
                        or ckpt_path.endswith(".model")
                    ):
                        ckpt_path = ckpt_path + ".pt"
                    torch.save(ckpt, ckpt_path)
                    print(f"[Save] Fallback checkpoint saved to {ckpt_path}")
                except Exception as e2:
                    print(f"[Error] Fallback torch.save also failed: {e2}")
        else:
            no_improve += 1
            print(
                f"[Info] No val Recall@{MAIN_METRIC_K} improvement for {no_improve} epoch(s). "
                f"Best so far: epoch={best_epoch}, "
                f"val_loss={best_val_loss:.4f}, "
                f"Recall@{MAIN_METRIC_K}={best_recall_main:.4f}"
            )

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(
                f"[EarlyStop] No improvement on Recall@{MAIN_METRIC_K} for {no_improve} epochs "
                f"(patience={args.early_stop_patience}), stop training."
            )
            break

    print(
        f"\n[Done] Training finished. "
        f"Best epoch={best_epoch}, "
        f"val_loss={best_val_loss:.4f}, "
        f"best_Recall@{MAIN_METRIC_K}={best_recall_main:.4f}"
    )


if __name__ == '__main__':
    main()
