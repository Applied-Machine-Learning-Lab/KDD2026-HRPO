# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import T5Config, T5ForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            vocab_size=config["vocab_size"],
            pad_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],
            decoder_start_token_id=config["pad_token_id"],
            feed_forward_proj=config["feed_forward_proj"],
        )
        self.model = T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


class KuaiRandSIDTigerDataset(Dataset):
    """
    Build Tiger training samples from log CSV and SID mapping.
    Each sample is (history_sid_tokens, target_sid_tokens):
    - item SID is [sid_1, ..., sid_D]
    - token id uses sid_raw + 1 (0 is PAD/EOS)
    - history packs up to H items then flattens to length H * D
    - target is the current clicked item SID sequence (length D)
    """

    def __init__(
        self,
        log_paths: List[str],
        sid_mapping_path: str,
        sep: str = ",",
        user_col: str = "user_id",
        item_col: str = "video_id",
        label_col: str = "is_click",
        only_positive: bool = True,
        min_hist_items: int = 1,
        max_hist_items: int = 50,
    ):
        super().__init__()
        self.max_hist_items = max_hist_items
        self.min_hist_items = min_hist_items
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.only_positive = only_positive

        dfs = []
        for p in log_paths:
            p = p.strip()
            if not p:
                continue
            if not os.path.exists(p):
                raise FileNotFoundError(f"Log file not found: {p}")
            df = pd.read_csv(p, sep=sep)
            dfs.append(df)

        if not dfs:
            raise ValueError("No valid log_paths provided.")

        df_log = pd.concat(dfs, ignore_index=True)

        if only_positive and label_col in df_log.columns:
            df_log = df_log[df_log[label_col] > 0].copy()

        if "time_ms" in df_log.columns:
            df_log = df_log.sort_values([user_col, "time_ms"])
        elif "date" in df_log.columns and "hourmin" in df_log.columns:
            df_log = df_log.sort_values([user_col, "date", "hourmin"])

        df_log = df_log[[user_col, item_col]]

        if not os.path.exists(sid_mapping_path):
            raise FileNotFoundError(f"SID mapping not found: {sid_mapping_path}")
        df_sid = pd.read_csv(sid_mapping_path)

        sid_cols = [c for c in df_sid.columns if c.startswith("sid_")]
        if not sid_cols:
            raise ValueError(f"No sid_* columns found in {sid_mapping_path}")
        self.sid_cols = sid_cols
        self.sid_depth = len(sid_cols)

        codes_raw = df_sid[sid_cols].values.astype(int)
        self.codebook_size = int(codes_raw.max() + 1)

        df = df_log.merge(df_sid, on=item_col, how="inner")

        from collections import defaultdict

        user2_items: Dict[Any, List[List[int]]] = defaultdict(list)
        for _, row in df.iterrows():
            uid = row[user_col]
            sid_seq = [int(row[c]) + 1 for c in sid_cols]  # [sid_1+1,...]
            user2_items[uid].append(sid_seq)

        self.pad_token_id = 0
        self.eos_token_id = 0
        self.vocab_size = self.codebook_size + 1  # 0..K -> vocab_size=K+1

        self.samples: List[Dict[str, torch.Tensor]] = []
        sample_uid_list: List[int] = []

        max_hist_tokens = max_hist_items * self.sid_depth

        for uid, seq in user2_items.items():
            T = len(seq)
            if T <= 1:
                continue

            for t in range(1, T):
                hist_items = seq[max(0, t - max_hist_items): t]
                if len(hist_items) < min_hist_items:
                    continue

                hist_tokens: List[int] = []
                for codes_item in hist_items:
                    hist_tokens.extend(codes_item)

                if len(hist_tokens) > max_hist_tokens:
                    hist_tokens = hist_tokens[-max_hist_tokens:]

                pad_len = max_hist_tokens - len(hist_tokens)
                history = [self.pad_token_id] * pad_len + hist_tokens
                attention_mask = [0] * pad_len + [1] * len(hist_tokens)
                target = seq[t]  # length = sid_depth

                self.samples.append(
                    {
                        "history": torch.tensor(history, dtype=torch.long),
                        "attention_mask": torch.tensor(
                            attention_mask, dtype=torch.long
                        ),
                        "target": torch.tensor(target, dtype=torch.long),
                    }
                )
                sample_uid_list.append(int(uid))

        if not self.samples:
            raise ValueError(
                "No Tiger-SID training samples constructed from logs + SID mapping."
            )

        self.max_seq_len = max_hist_tokens  # inputlength(history)
        self.sample_user_ids = np.array(sample_uid_list, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
 preds: (B, maxk, L) ; labels: (B, L)
  (B, maxk)  bool,  j  beam 
 """
    return (preds == labels.unsqueeze(1)).all(dim=-1)


def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    return pos_index[:, :k].any(dim=1).float()


def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    B, M = pos_index.shape
    ranks = torch.arange(1, M + 1, device=pos_index.device, dtype=torch.float)
    dcg_weights = 1.0 / torch.log2(ranks + 1.0)  # (M,)
    dcg = pos_index.float() * dcg_weights
    return dcg[:, :k].sum(dim=1)


def train_one_epoch(
    model: TIGER,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)  # (B, sid_depth)

        optimizer.zero_grad()
        loss, _ = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.inference_mode()
def evaluate(
    model: TIGER,
    eval_loader: DataLoader,
    topk_list: List[int],
    beam_size: int,
    device: torch.device,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
 Beam Search evaluation: 
 - sample beam_size candidate SID ; 
 -  SID token  beam ; 
 -  item  Recall@K / NDCG@K. 
 """
    model.eval()
    max_k = max(topk_list)
    effective_beams = max(beam_size, max_k)  #  K

    sum_recalls = {k: 0.0 for k in topk_list}
    sum_ndcgs = {k: 0.0 for k in topk_list}
    n_batches = 0

    for batch in eval_loader:
        input_ids = batch["history"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)  # (B, sid_depth)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=effective_beams,
            num_return_sequences=effective_beams,
            max_length=labels.shape[1] + 1,  # +1  decoder_start_token
            early_stopping=True,
            do_sample=False,
        )
        gen = gen[:, 1:]  # (B * beams, L)
        B, L = labels.shape
        gen = gen.view(B, effective_beams, L)

        pos_index = calculate_pos_index(gen, labels)  # (B, beams)

        for k in topk_list:
            k_eff = min(k, effective_beams)
            r = recall_at_k(pos_index, k_eff).mean().item()
            n = ndcg_at_k(pos_index, k_eff).mean().item()
            sum_recalls[k] += r
            sum_ndcgs[k] += n

        n_batches += 1

    avg_recalls = {k: (sum_recalls[k] / max(1, n_batches)) for k in topk_list}
    avg_ndcgs = {k: (sum_ndcgs[k] / max(1, n_batches)) for k in topk_list}
    return avg_recalls, avg_ndcgs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Tiger baseline (SID version) on KuaiRand-Pure"
    )

    parser.add_argument(
        "--log_paths",
        type=str,
        default=str(PROJECT_ROOT / "dataset/kuairand/kuairand-Pure/data/log_standard_4_08_to_4_21_pure.csv")
        + ","
        + str(PROJECT_ROOT / "dataset/kuairand/kuairand-Pure/data/log_standard_4_22_to_5_08_pure.csv"),
        help="Comma-separated input log CSV paths.",
    )
    parser.add_argument(
        "--sid_mapping_path",
        type=str,
        default=str(PROJECT_ROOT / "code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"),
        help="Path to video_sid_mapping.csv (with sid_1, sid_2, ... columns).",
    )
    parser.add_argument("--data_separator", type=str, default=",")
    parser.add_argument("--label_col", type=str, default="is_click")
    parser.add_argument("--min_hist_items", type=int, default=1)
    parser.add_argument("--max_hist_items", type=int, default=50)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument(
        "--save_path", type=str, default="./ckpt/tiger_sid_krpure.pth"
    )
    parser.add_argument(
        "--log_path", type=str, default="./logs/tiger_sid_krpure.log"
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["mini", "medium", "large"],
        help="Tiger model size preset.",
    )
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")

    parser.add_argument(
        "--topk_list",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Top-K values used in evaluation, e.g. --topk_list 5 10 20.",
    )
    parser.add_argument("--beam_size", type=int, default=30)

    args = parser.parse_args()

    if args.model_size == "mini":
        args.num_layers = 3
        args.num_decoder_layers = 3
        args.d_model = 128
        args.d_ff = 512
        args.num_heads = 4
        args.d_kv = 16
    elif args.model_size == "medium":
        args.num_layers = 4
        args.num_decoder_layers = 4
        args.d_model = 128
        args.d_ff = 1024
        args.num_heads = 6
        args.d_kv = 64
    elif args.model_size == "large":
        args.num_layers = 6
        args.num_decoder_layers = 6
        args.d_model = 192
        args.d_ff = 1536
        args.num_heads = 8
        args.d_kv = 24

    config: Dict[str, Any] = vars(args)

    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["log_path"]), exist_ok=True)

    logging.basicConfig(
        filename=config["log_path"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    log_paths = [p.strip() for p in config["log_paths"].split(",") if p.strip()]
    dataset = KuaiRandSIDTigerDataset(
        log_paths=log_paths,
        sid_mapping_path=config["sid_mapping_path"],
        sep=config["data_separator"],
        label_col=config["label_col"],
        only_positive=True,
        min_hist_items=config["min_hist_items"],
        max_hist_items=config["max_hist_items"],
    )

    config["vocab_size"] = dataset.vocab_size
    config["pad_token_id"] = dataset.pad_token_id
    config["eos_token_id"] = dataset.eos_token_id
    config["max_len"] = dataset.max_seq_len

    total_len = len(dataset)
    sample_uids = dataset.sample_user_ids  # [N]

    uid_last_idx: Dict[int, int] = {}
    for idx, uid in enumerate(sample_uids):
        uid_last_idx[int(uid)] = idx  #  -> sample

    val_indices = sorted(uid_last_idx.values())
    val_set = set(val_indices)
    train_indices = [i for i in range(total_len) if i not in val_set]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_len = len(train_dataset)
    val_len = len(val_dataset)

    logging.info(
        f"Dataset constructed. Total={total_len}, Train={train_len}, Val={val_len}, "
        f"Users={len(uid_last_idx)}, SID depth={dataset.sid_depth}, "
        f"codebook_size={dataset.codebook_size}, vocab_size={dataset.vocab_size}"
    )
    print(
        f"[Data] Total={total_len}, Train={train_len} ({train_len/total_len:.4f}), "
        f"Val={val_len} ({val_len/total_len:.4f}), Users={len(uid_last_idx)}\n"
        f"       SID depth={dataset.sid_depth}, codebook_size={dataset.codebook_size}, "
        f"vocab_size={dataset.vocab_size}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["infer_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = TIGER(config).to(device)
    logging.info(model.n_parameters)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {num_params:,d}")
    print(model.n_parameters)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    key_k = 5 if 5 in config["topk_list"] else config["topk_list"][0]
    best_recall = -1.0
    early_stop_counter = 0

    for epoch in range(config["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logging.info(
            f"[Epoch {epoch+1}/{config['num_epochs']}] Train loss: {train_loss:.4f}"
        )
        print(f"[Epoch {epoch+1}] Train loss: {train_loss:.4f}")

        val_recalls, val_ndcgs = evaluate(
            model, val_loader, config["topk_list"], config["beam_size"], device
        )
        logging.info(f"Validation Recall: {val_recalls}")
        logging.info(f"Validation NDCG:   {val_ndcgs}")
        print(
            f"[Epoch {epoch+1}] Val Recall: {val_recalls}, "
            f"Val NDCG: {val_ndcgs}"
        )

        cur_recall = val_recalls[key_k]

        if cur_recall > best_recall:
            best_recall = cur_recall
            early_stop_counter = 0

            torch.save(model.state_dict(), config["save_path"])
            logging.info(
                f"[Best@Epoch{epoch+1}] New best Recall@{key_k}={best_recall:.6f}, "
                f"model saved to {config['save_path']}"
            )
            print(
                f"[Epoch {epoch+1}] *** New best Recall@{key_k}={best_recall:.6f}, "
                f"model saved."
            )
        else:
            early_stop_counter += 1
            logging.info(
                f"No improvement in Recall@{key_k}. "
                f"Early stop counter: {early_stop_counter}"
            )
            print(
                f"[Epoch {epoch+1}] Recall@{key_k} not improved "
                f"({cur_recall:.6f} <= {best_recall:.6f}), "
                f"early_stop_counter={early_stop_counter}/{config['early_stop']}"
            )
            if early_stop_counter >= config["early_stop"]:
                logging.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
