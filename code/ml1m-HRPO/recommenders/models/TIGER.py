import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _split_tags(value: Any) -> List[str]:
    raw = _clean_text(value)
    if not raw:
        return []
    return [p.strip() for p in re.split(r"[|,/;]+", raw) if p.strip()]


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_review_count(summary: Any) -> int:
    text = _clean_text(summary).lower()
    if not text:
        return 0
    m = re.search(r"review count:\s*([0-9]+)", text)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _canonical_tag(tag: str) -> str:
    x = _clean_text(tag).lower()
    mapping = {
        "all beauty": "all beauty",
        "hair care": "hair care",
        "skin care": "skin care",
        "makeup": "makeup",
        "fragrance": "fragrance",
        "cleanser": "cleanser",
        "mask": "mask",
        "bath": "bath",
        "tools": "tools",
        "accessories": "accessories",
        "personal care": "personal care",
    }
    return mapping.get(x, x)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / max(union, 1))


def _sinkhorn_transport(
    row_mass: np.ndarray,
    col_mass: np.ndarray,
    cost: np.ndarray,
    *,
    epsilon: float = 0.35,
    n_iter: int = 16,
) -> np.ndarray:
    if cost.size == 0:
        return np.zeros_like(cost, dtype=np.float32)
    r = np.asarray(row_mass, dtype=np.float64).reshape(-1)
    c = np.asarray(col_mass, dtype=np.float64).reshape(-1)
    r = np.clip(r, 1e-8, None)
    c = np.clip(c, 1e-8, None)
    r = r / r.sum()
    c = c / c.sum()
    K = np.exp(-np.asarray(cost, dtype=np.float64) / max(float(epsilon), 1e-6))
    K = np.clip(K, 1e-8, None)
    u = np.ones_like(r)
    v = np.ones_like(c)
    for _ in range(max(int(n_iter), 1)):
        Kv = np.maximum(K @ v, 1e-8)
        u = r / Kv
        KTu = np.maximum(K.T @ u, 1e-8)
        v = c / KTu
    plan = (u[:, None] * K) * v[None, :]
    plan_sum = float(plan.sum())
    if plan_sum > 0:
        plan = plan / plan_sum
    return plan.astype(np.float32)


class TIGERBackbone(nn.Module):
    """Native TIGER backbone (T5 over SID token sequences)."""

    def __init__(self, config: Dict):
        super().__init__()
        t5_cfg = T5Config(
            num_layers=int(config["num_layers"]),
            num_decoder_layers=int(config["num_decoder_layers"]),
            d_model=int(config["d_model"]),
            d_ff=int(config["d_ff"]),
            num_heads=int(config["num_heads"]),
            d_kv=int(config["d_kv"]),
            dropout_rate=float(config["dropout_rate"]),
            vocab_size=int(config["vocab_size"]),
            pad_token_id=int(config["pad_token_id"]),
            eos_token_id=int(config["eos_token_id"]),
            decoder_start_token_id=int(config["pad_token_id"]),
            feed_forward_proj=str(config.get("feed_forward_proj", "relu")),
        )
        self.model = T5ForConditionalGeneration(t5_cfg)

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss, out.logits

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def decode_with_hidden(self, input_ids, attention_mask=None, decoder_input_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return out.logits, out.decoder_hidden_states[-1]


class TokenCreditTransportHead(nn.Module):
    """Predict signed block credit for a specific SID token under a decode context."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        token_dim: int = 32,
        mlp_dim: int = 128,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.token_dim = int(token_dim)
        self.mlp_dim = int(mlp_dim)
        self.token_emb = nn.Embedding(self.vocab_size, self.token_dim)
        self.norm = nn.LayerNorm(self.hidden_size + self.token_dim)
        self.fc1 = nn.Linear(self.hidden_size + self.token_dim, self.mlp_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(self.mlp_dim, 1)

    def _merge(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_emb(token_ids)
        x = torch.cat([hidden_states, tok], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            out = self._merge(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                token_ids.reshape(-1),
            )
            return out.view(hidden_states.shape[0], hidden_states.shape[1])
        return self._merge(hidden_states, token_ids)

    def score_all_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [B, D]")
        bsz = int(hidden_states.shape[0])
        token_ids = torch.arange(self.vocab_size, device=hidden_states.device, dtype=torch.long)
        hidden = hidden_states.unsqueeze(1).expand(bsz, self.vocab_size, hidden_states.shape[-1])
        tok = self.token_emb(token_ids).unsqueeze(0).expand(bsz, self.vocab_size, self.token_dim)
        x = torch.cat([hidden, tok], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x).squeeze(-1)
        return x


class TIGER(nn.Module):
    """
    Native TIGER model for Agent4Rec simulation.
    Requires:
    - checkpoint: recommenders/weights/<dataset>/TIGER/Saved/epoch=best.tiger.native.pth
    - SID mapping: recommenders/weights/<dataset>/TIGER/Saved/sid_mapping_internal.csv
    """

    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data
        self.device = torch.device(args.cuda)
        self.n_items = int(data.n_items)

        self.popularity = np.zeros(self.n_items, dtype=np.float32)
        for item, users in data.train_item_list.items():
            self.popularity[int(item)] = float(len(users))
        pmax = float(self.popularity.max())
        if pmax > 0:
            self.popularity = self.popularity / pmax
        self.novelty = 1.0 - self.popularity

        self.max_hist_items = int(os.getenv("TIGER_MAX_HIST_ITEMS", "50"))
        self.beam_width = int(os.getenv("TIGER_BEAM_WIDTH", "64"))
        self.decode_rounds = int(os.getenv("TIGER_DECODE_ROUNDS", "3"))
        self.pop_coef = float(os.getenv("TIGER_POP_COEF", "0.05"))
        self.novel_coef = float(os.getenv("TIGER_NOVEL_COEF", "0.08"))
        self.attr_profile = str(getattr(args, "tiger_attr_profile", os.getenv("TIGER_ATTR_PROFILE", "default")) or "default")
        self.attr_history_items = max(
            int(getattr(args, "tiger_attr_history_items", os.getenv("TIGER_ATTR_HISTORY_ITEMS", "12"))),
            1,
        )
        self.decode_attr_scale = float(os.getenv("TIGER_ATTR_DECODE_SCALE", "0.18"))
        self.decode_cf_scale = float(os.getenv("TIGER_ATTR_COUNTERFACTUAL_SCALE", "0.08"))
        self.decode_block_scale = float(os.getenv("TIGER_ATTR_BLOCK_SCALE", "0.08"))
        self.decode_transport_scale = float(
            getattr(args, "tiger_transport_scale", os.getenv("TIGER_TRANSPORT_SCALE", "0.20"))
        )
        self.decode_transport_topk = max(
            int(getattr(args, "tiger_transport_topk", os.getenv("TIGER_TRANSPORT_TOPK", "12"))),
            2,
        )
        self.transport_enabled = str(
            getattr(args, "tiger_transport_enabled", os.getenv("TIGER_TRANSPORT_ENABLED", "auto")) or "auto"
        ).lower()

        self.backbone = None
        self.sid_depth = None
        self.iid2sid_tok = None
        self.sid2iid: Dict[Tuple[int, ...], int] = {}
        self.sid_prefix_to_next: Dict[Tuple[int, ...], List[int]] = {}
        self.score_cache: Dict[int, np.ndarray] = {}
        self.attr_cache: Dict[Tuple[Tuple[int, ...], int], Dict[str, Any]] = {}
        self.transport_cache: Dict[Tuple[Tuple[int, ...], int], Dict[str, Any]] = {}
        self.item_catalog = self._load_item_catalog(dataset=args.dataset)
        self.sid_token_freq: List[Dict[int, int]] = []
        self.transport_head: Optional[TokenCreditTransportHead] = None
        self.transport_meta: Dict[str, Any] = {}

        model_path = getattr(args, "model_path", "Saved")
        ckpt_path = self._resolve_checkpoint_path(dataset=args.dataset, model_path=model_path)
        sid_path = self._resolve_sid_mapping_path(dataset=args.dataset, model_path=model_path)
        transport_head_path, transport_meta_path = self._resolve_transport_paths(dataset=args.dataset, model_path=model_path)
        if ckpt_path is None or sid_path is None:
            print("[TIGER] missing checkpoint or SID mapping, fallback to popularity scores.")
            return

        try:
            self._load_native_tiger(ckpt_path, sid_path)
            print(f"[TIGER] loaded native checkpoint: {ckpt_path}")
            print(f"[TIGER] loaded SID mapping: {sid_path}")
            if transport_head_path is not None and transport_meta_path is not None:
                self._load_transport_head(transport_head_path, transport_meta_path)
                print(f"[TIGER] loaded credit transport head: {transport_head_path}")
        except Exception as e:
            print(f"[TIGER] load failed ({e}), fallback to popularity scores.")
            self.backbone = None
            self.iid2sid_tok = None
            self.sid2iid = {}
            self.sid_prefix_to_next = {}
            self.transport_head = None

    def _resolve_checkpoint_path(self, dataset: str, model_path: str):
        env_path = os.getenv("TIGER_CKPT", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        model_path = str(model_path or "Saved")
        candidates = [
            Path(f"recommenders/weights/{dataset}/TIGER/{model_path}/epoch=best.tiger.native.pth"),
            Path(f"recommenders/weights/{dataset}/TIGER/{model_path}/tiger_native_last.pth"),
            Path(f"recommenders/weights/{dataset}/TIGER/Saved/epoch=best.tiger.native.pth"),
            Path(f"recommenders/weights/{dataset}/TIGER/Saved/tiger_native_last.pth"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _resolve_sid_mapping_path(self, dataset: str, model_path: str):
        env_path = os.getenv("TIGER_SID_MAP", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        model_path = str(model_path or "Saved")
        candidates = [
            Path(f"recommenders/weights/{dataset}/TIGER/{model_path}/sid_mapping_internal.csv"),
            Path(f"recommenders/weights/{dataset}/TIGER/Saved/sid_mapping_internal.csv"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _resolve_transport_paths(self, dataset: str, model_path: str):
        env_head = os.getenv("TIGER_TRANSPORT_HEAD", "").strip()
        env_meta = os.getenv("TIGER_TRANSPORT_META", "").strip()
        if env_head and env_meta:
            head = Path(env_head)
            meta = Path(env_meta)
            if head.exists() and meta.exists():
                return head, meta

        model_path = str(model_path or "Saved")
        candidates = [
            (
                Path(f"recommenders/weights/{dataset}/TIGER/{model_path}/credit_transport_head.pt"),
                Path(f"recommenders/weights/{dataset}/TIGER/{model_path}/credit_transport_meta.json"),
            ),
            (
                Path(f"recommenders/weights/{dataset}/TIGER/Saved/credit_transport_head.pt"),
                Path(f"recommenders/weights/{dataset}/TIGER/Saved/credit_transport_meta.json"),
            ),
        ]
        for head, meta in candidates:
            if head.exists() and meta.exists():
                return head, meta
        return None, None

    def _default_config(self, vocab_size: int):
        return {
            "num_layers": 3,
            "num_decoder_layers": 3,
            "d_model": 128,
            "d_ff": 512,
            "num_heads": 4,
            "d_kv": 16,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu",
            "vocab_size": int(vocab_size),
            "pad_token_id": 0,
            "eos_token_id": 0,
        }

    def _load_native_tiger(self, ckpt_path: Path, sid_path: Path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            sid_depth = int(ckpt["sid_depth"])
            config = ckpt.get("config", None)
            if config is None:
                codebook_size = int(ckpt.get("codebook_size", 128))
                config = self._default_config(vocab_size=codebook_size + 1)
        else:
            raise ValueError("invalid TIGER checkpoint format")

        self.sid_depth = int(sid_depth)
        self.iid2sid_tok, self.sid2iid = self._load_sid_mapping(sid_path, self.sid_depth)
        self._prepare_sid_token_stats()

        inferred_vocab = int(self.iid2sid_tok.max()) + 1
        config["vocab_size"] = int(max(int(config.get("vocab_size", inferred_vocab)), inferred_vocab))
        config["pad_token_id"] = 0
        config["eos_token_id"] = 0

        self.backbone = TIGERBackbone(config)
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.eval()
        self.backbone = self.backbone.to(self.device)

    def _load_transport_head(self, head_path: Path, meta_path: Path):
        if self.backbone is None:
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        head = TokenCreditTransportHead(
            hidden_size=int(meta.get("hidden_size", int(self.backbone.model.config.d_model))),
            vocab_size=int(meta.get("vocab_size", int(self.backbone.model.config.vocab_size))),
            token_dim=int(meta.get("token_dim", 32)),
            mlp_dim=int(meta.get("mlp_dim", 128)),
        )
        state = torch.load(head_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        head.load_state_dict(state, strict=True)
        head.eval()
        self.transport_head = head.to(self.device)
        self.transport_meta = dict(meta)

    def _load_sid_mapping(self, sid_path: Path, sid_depth: int):
        df = pd.read_csv(sid_path)
        sid_cols = [c for c in df.columns if c.startswith("sid_")]
        sid_cols = sorted(sid_cols, key=lambda x: int(x.split("_")[1]))
        if len(sid_cols) != sid_depth:
            raise ValueError(f"sid depth mismatch: ckpt={sid_depth}, mapping={len(sid_cols)}")

        if "item_id" in df.columns:
            item_col = "item_id"
        elif "video_id" in df.columns:
            item_col = "video_id"
        else:
            raise ValueError("SID mapping must contain item_id or video_id")

        iid2sid_tok = np.zeros((self.n_items + 1, sid_depth), dtype=np.int64)
        sid2iid = {}

        for _, row in df.iterrows():
            iid = int(row[item_col])
            if iid < 0 or iid >= self.n_items:
                continue
            raw = row[sid_cols].astype(int).values
            tok = (raw + 1).astype(np.int64)
            iid2sid_tok[iid + 1] = tok
            sid2iid[tuple(tok.tolist())] = iid

        # Fill any missing item with deterministic fallback code.
        missing = np.where(iid2sid_tok[1:].sum(axis=1) == 0)[0].tolist()
        if missing:
            max_tok = int(iid2sid_tok.max())
            codebook = max(max_tok, 2)
            for iid in missing:
                x = int(iid)
                digits = [0] * sid_depth
                for p in range(sid_depth - 1, -1, -1):
                    digits[p] = (x % codebook) + 1
                    x //= codebook
                tok = np.asarray(digits, dtype=np.int64)
                iid2sid_tok[iid + 1] = tok
                sid2iid[tuple(tok.tolist())] = iid

        self._prepare_sid_prefix_trie(sid2iid)
        return iid2sid_tok, sid2iid

    def _prepare_sid_prefix_trie(self, sid2iid: Dict[Tuple[int, ...], int]) -> None:
        trie: Dict[Tuple[int, ...], set] = {}
        for tok in sid2iid.keys():
            prefix: Tuple[int, ...] = tuple()
            for value in tok:
                trie.setdefault(prefix, set()).add(int(value))
                prefix = tuple(list(prefix) + [int(value)])
        self.sid_prefix_to_next = {prefix: sorted(int(v) for v in values) for prefix, values in trie.items()}

    def _load_item_catalog(self, dataset: str) -> Dict[int, Dict[str, Any]]:
        path = Path(f"datasets/{dataset}/simulation/movie_detail.csv")
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        if "movie_id" in df.columns:
            item_ids = df["movie_id"].astype(int).tolist()
        else:
            item_ids = list(range(len(df)))

        catalog: Dict[int, Dict[str, Any]] = {}
        for idx, item_id in enumerate(item_ids):
            row = df.iloc[idx]
            tags = [_canonical_tag(x) for x in _split_tags(getattr(row, "genres", "")) if _clean_text(x)]
            rating = _parse_float(getattr(row, "rating", 0.0), 0.0)
            review_count = _parse_review_count(getattr(row, "summary", ""))
            quality = 0.65 * max(min(rating / 5.0, 1.0), 0.0) + 0.35 * max(
                min(np.log1p(review_count) / np.log1p(500.0), 1.0),
                0.0,
            )
            catalog[int(item_id)] = {
                "item_id": int(item_id),
                "title": _clean_text(getattr(row, "title", "")),
                "tags": tags,
                "rating": float(rating),
                "review_count": int(review_count),
                "quality": float(quality),
            }
        return catalog

    def _prepare_sid_token_stats(self) -> None:
        self.sid_token_freq = []
        if self.iid2sid_tok is None or self.sid_depth is None:
            return
        for pos in range(int(self.sid_depth)):
            col = self.iid2sid_tok[1:, pos].astype(int).tolist()
            freq: Dict[int, int] = {}
            for tok in col:
                freq[int(tok)] = int(freq.get(int(tok), 0) + 1)
            self.sid_token_freq.append(freq)

    def _token_specificity(self, pos: int, token: int) -> float:
        if pos < 0 or pos >= len(self.sid_token_freq):
            return 0.0
        freq = int(self.sid_token_freq[pos].get(int(token), 0))
        if freq <= 0 or self.n_items <= 0:
            return 0.0
        return float(1.0 - min(np.log1p(freq) / np.log1p(max(self.n_items, 2)), 1.0))

    def _history_prior_weights(self, history_items: Sequence[int], candidate_iid: int) -> np.ndarray:
        cand_info = self.item_catalog.get(int(candidate_iid), {})
        cand_tags = cand_info.get("tags", [])
        cand_tok = self.iid2sid_tok[int(candidate_iid) + 1] if self.iid2sid_tok is not None else None
        priors: List[float] = []
        hist_len = max(len(history_items), 1)
        for idx, iid in enumerate(history_items):
            info = self.item_catalog.get(int(iid), {})
            hist_tags = info.get("tags", [])
            tag_match = _jaccard(hist_tags, cand_tags)
            prefix = 0.0
            if cand_tok is not None and 0 <= int(iid) < self.n_items:
                hist_tok = self.iid2sid_tok[int(iid) + 1]
                matched = 0
                for a, b in zip(hist_tok.tolist(), cand_tok.tolist()):
                    if int(a) == int(b):
                        matched += 1
                    else:
                        break
                prefix = float(matched / max(len(cand_tok), 1))
            recency = float(np.exp(-0.18 * max(hist_len - idx - 1, 0)))
            priors.append(recency * (0.55 + 0.30 * tag_match + 0.15 * prefix))
        arr = np.asarray(priors, dtype=np.float32)
        if arr.size == 0:
            return arr
        arr = np.clip(arr, 1e-6, None)
        return arr / float(arr.sum())

    def _empty_transport_bundle(self, history_len: int) -> Dict[str, Any]:
        depth = max(int(self.sid_depth or 1), 1)
        return {
            "support": 0.0,
            "counterfactual_gap": 0.0,
            "residual_support": 0.0,
            "decode_bonus": 0.0,
            "block_weights": [1.0 / float(depth)] * depth,
            "block_support": [0.0] * depth,
            "history_weights": [1.0 / max(int(history_len), 1)] * int(history_len) if history_len > 0 else [],
            "top_history": [],
            "transport": [],
        }

    def _build_affinity_inputs(self, history_items: Sequence[int], candidate_iid: int):
        if self.iid2sid_tok is None or self.sid_depth is None:
            return None
        hist = [int(i) for i in history_items if 0 <= int(i) < self.n_items]
        if not hist:
            return None
        cand_tok = self.iid2sid_tok[int(candidate_iid) + 1]
        cand_info = self.item_catalog.get(int(candidate_iid), {})
        cand_tags = cand_info.get("tags", [])
        row_mass = self._history_prior_weights(hist, int(candidate_iid))
        col_mass = np.asarray(
            [float(int(self.sid_depth) - pos) for pos in range(int(self.sid_depth))],
            dtype=np.float32,
        )
        col_mass = np.clip(col_mass, 1e-6, None)
        col_mass = col_mass / float(col_mass.sum())
        affinity = np.zeros((len(hist), int(self.sid_depth)), dtype=np.float32)
        for h_idx, hist_iid in enumerate(hist):
            hist_info = self.item_catalog.get(int(hist_iid), {})
            hist_tags = hist_info.get("tags", [])
            tag_match = _jaccard(hist_tags, cand_tags)
            hist_tok = self.iid2sid_tok[int(hist_iid) + 1]
            prefix = 0
            for b_idx in range(int(self.sid_depth)):
                if int(hist_tok[b_idx]) == int(cand_tok[b_idx]):
                    prefix += 1
                else:
                    break
            for b_idx in range(int(self.sid_depth)):
                prefix_match = 0.0
                if prefix > 0:
                    prefix_match = float(min(prefix, b_idx + 1) / float(b_idx + 1))
                token_match = 1.0 if int(hist_tok[b_idx]) == int(cand_tok[b_idx]) else 0.0
                specificity = self._token_specificity(b_idx, int(cand_tok[b_idx]))
                affinity[h_idx, b_idx] = float(
                    min(0.42 * tag_match + 0.28 * prefix_match + 0.18 * token_match + 0.12 * specificity, 1.0)
                )
        return {
            "hist": hist,
            "cand_tok": cand_tok,
            "row_mass": row_mass,
            "col_mass": col_mass,
            "affinity": affinity,
        }

    def build_credit_transport_targets(
        self,
        history_items: Sequence[int],
        candidate_iid: int,
        page_credit: float,
    ) -> Dict[str, Any]:
        payload = self._build_affinity_inputs(history_items, int(candidate_iid))
        if payload is None:
            return {
                "page_credit": float(page_credit),
                "positive_mass": float(max(float(page_credit), 0.0)),
                "negative_mass": float(max(-float(page_credit), 0.0)),
                "positive_transport": [],
                "negative_transport": [],
                "block_credit": [0.0] * max(int(self.sid_depth or 1), 1),
                "history_credit": [],
                "conservation_gap": float(page_credit),
            }
        hist = payload["hist"]
        row_mass = np.asarray(payload["row_mass"], dtype=np.float32)
        col_mass = np.asarray(payload["col_mass"], dtype=np.float32)
        affinity = np.clip(np.asarray(payload["affinity"], dtype=np.float32), 0.0, 1.0)
        pos_mass = float(max(float(page_credit), 0.0))
        neg_mass = float(max(-float(page_credit), 0.0))
        pos_transport = np.zeros_like(affinity, dtype=np.float32)
        neg_transport = np.zeros_like(affinity, dtype=np.float32)
        if pos_mass > 0.0:
            pos_transport = _sinkhorn_transport(
                row_mass=row_mass,
                col_mass=col_mass,
                cost=1.0 - affinity,
                epsilon=0.32,
                n_iter=18,
            )
        if neg_mass > 0.0:
            neg_transport = _sinkhorn_transport(
                row_mass=row_mass,
                col_mass=col_mass,
                cost=affinity,
                epsilon=0.32,
                n_iter=18,
            )
        block_credit = pos_mass * pos_transport.sum(axis=0) - neg_mass * neg_transport.sum(axis=0)
        history_credit = pos_mass * pos_transport.sum(axis=1) - neg_mass * neg_transport.sum(axis=1)
        return {
            "page_credit": float(page_credit),
            "positive_mass": float(pos_mass),
            "negative_mass": float(neg_mass),
            "positive_transport": pos_transport.astype(np.float32).tolist(),
            "negative_transport": neg_transport.astype(np.float32).tolist(),
            "block_credit": [float(v) for v in block_credit.astype(np.float32).tolist()],
            "history_credit": [float(v) for v in history_credit.astype(np.float32).tolist()],
            "conservation_gap": float(float(page_credit) - float(np.sum(block_credit))),
            "hist_items": [int(v) for v in hist],
        }

    def _candidate_attribution_core(self, history_items: Sequence[int], candidate_iid: int) -> Dict[str, Any]:
        payload = self._build_affinity_inputs(history_items, int(candidate_iid))
        if payload is None:
            return self._empty_transport_bundle(len(history_items))
        hist = payload["hist"]
        cand_tok = payload["cand_tok"]
        affinity = np.clip(np.asarray(payload["affinity"], dtype=np.float32), 0.0, 1.0)
        cost = 1.0 - affinity
        row_mass = np.asarray(payload["row_mass"], dtype=np.float32)
        col_mass = np.asarray(payload["col_mass"], dtype=np.float32)

        plan = _sinkhorn_transport(row_mass=row_mass, col_mass=col_mass, cost=cost, epsilon=0.32, n_iter=18)
        support = float(np.sum(plan * affinity))
        history_weights = np.asarray(plan.sum(axis=1), dtype=np.float32)
        raw_block_weights = np.asarray(plan.sum(axis=0), dtype=np.float32)
        raw_block_support = np.asarray(
            [
                float(np.sum(plan[:, pos] * affinity[:, pos])) * (0.65 + 0.35 * self._token_specificity(pos, int(cand_tok[pos])))
                for pos in range(int(self.sid_depth))
            ],
            dtype=np.float32,
        )
        block_weights = raw_block_weights * np.maximum(raw_block_support, 1e-6)
        if float(block_weights.sum()) > 0:
            block_weights = block_weights / float(block_weights.sum())
        else:
            block_weights = np.full((int(self.sid_depth),), 1.0 / float(int(self.sid_depth)), dtype=np.float32)

        if history_weights.size > 1:
            top_idx = int(np.argmax(history_weights))
            mask = np.ones((len(hist),), dtype=bool)
            mask[top_idx] = False
            residual = self._candidate_attribution_core([hist[i] for i in range(len(hist)) if mask[i]], int(candidate_iid))
            residual_support = float(residual.get("support", 0.0))
            counterfactual_gap = max(float(support - residual_support), 0.0)
        else:
            residual_support = 0.0
            counterfactual_gap = float(support)

        block_alignment = float(np.dot(block_weights, np.clip(raw_block_support, 0.0, 1.0)))
        decode_bonus = float(
            0.55 * support
            + 0.25 * residual_support
            + 0.20 * block_alignment
            - 0.25 * counterfactual_gap
        )
        top_history = []
        order = np.argsort(-history_weights)
        for idx in order[: min(len(order), 3)]:
            hist_iid = int(hist[int(idx)])
            top_history.append(
                {
                    "item_id": hist_iid,
                    "title": self.item_catalog.get(hist_iid, {}).get("title", ""),
                    "weight": round(float(history_weights[int(idx)]), 4),
                }
            )
        return {
            "support": float(support),
            "counterfactual_gap": float(counterfactual_gap),
            "residual_support": float(residual_support),
            "decode_bonus": float(decode_bonus),
            "block_weights": [float(v) for v in block_weights.tolist()],
            "block_support": [float(v) for v in raw_block_support.tolist()],
            "history_weights": [float(v) for v in history_weights.tolist()],
            "top_history": top_history,
            "transport": plan.astype(np.float32).tolist(),
        }

    def get_candidate_attribution(
        self,
        candidate_iid: int,
        *,
        history_items: Optional[Sequence[int]] = None,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        hist_items = list(history_items) if history_items is not None else list(self.data.train_user_list.get(int(user_id or 0), []))
        hist_items = [int(i) for i in hist_items if 0 <= int(i) < self.n_items]
        if not hist_items:
            result = self._empty_transport_bundle(0)
            result.update(
                {
                    "transport_available": False,
                    "pred_block_credit": [0.0] * max(int(self.sid_depth or 1), 1),
                    "pred_page_credit": 0.0,
                    "pred_prefix_credit": [0.0] * max(int(self.sid_depth or 1), 1),
                }
            )
            return result
        hist_tail = tuple(hist_items[-self.attr_history_items :])
        cache_key = (hist_tail, int(candidate_iid))
        cached = self.attr_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self._candidate_attribution_core(history_items=list(hist_tail), candidate_iid=int(candidate_iid))
        result.update(self._predict_token_credit_bundle(list(hist_tail), int(candidate_iid)))
        self.attr_cache[cache_key] = result
        return result

    @torch.no_grad()
    def _predict_token_credit_bundle(self, history_items: Sequence[int], candidate_iid: int) -> Dict[str, Any]:
        if self.transport_head is None or self.backbone is None or self.iid2sid_tok is None:
            depth = max(int(self.sid_depth or 1), 1)
            return {
                "transport_available": False,
                "pred_block_credit": [0.0] * depth,
                "pred_page_credit": 0.0,
                "pred_prefix_credit": [0.0] * depth,
            }
        hist_tail = tuple(int(i) for i in history_items[-self.attr_history_items :] if 0 <= int(i) < self.n_items)
        cache_key = (hist_tail, int(candidate_iid))
        cached = self.transport_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        input_ids, attn = self._build_history_tokens(list(hist_tail))
        if input_ids is None:
            depth = max(int(self.sid_depth or 1), 1)
            return {
                "transport_available": False,
                "pred_block_credit": [0.0] * depth,
                "pred_page_credit": 0.0,
                "pred_prefix_credit": [0.0] * depth,
            }
        target_tok = self.iid2sid_tok[int(candidate_iid) + 1].astype(np.int64).tolist()
        decoder_inputs = [0] + target_tok[:-1]
        enc_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        enc_attn = torch.tensor([attn], dtype=torch.long, device=self.device)
        dec_input = torch.tensor([decoder_inputs], dtype=torch.long, device=self.device)
        logits, hidden = self.backbone.decode_with_hidden(
            input_ids=enc_input,
            attention_mask=enc_attn,
            decoder_input_ids=dec_input,
        )
        _ = logits  # kept for symmetry/debugging
        token_ids = torch.tensor([target_tok], dtype=torch.long, device=self.device)
        pred = self.transport_head(hidden, token_ids).detach().cpu().numpy().reshape(-1).astype(np.float32)
        prefix = np.cumsum(pred).astype(np.float32)
        out = {
            "transport_available": True,
            "pred_block_credit": [float(v) for v in pred.tolist()],
            "pred_page_credit": float(np.sum(pred)),
            "pred_prefix_credit": [float(v) for v in prefix.tolist()],
        }
        self.transport_cache[cache_key] = dict(out)
        return out

    def _allowed_next_tokens(self, prefix: Sequence[int]) -> List[int]:
        allowed = self.sid_prefix_to_next.get(tuple(int(v) for v in prefix), [])
        if allowed:
            return [int(v) for v in allowed if int(v) > 0]
        vocab = int(self.backbone.model.config.vocab_size) if self.backbone is not None else 0
        return [int(v) for v in range(1, max(vocab, 1))]

    @torch.no_grad()
    def _transport_guided_sequences(self, input_ids: torch.Tensor, attn: torch.Tensor) -> List[Tuple[List[int], float]]:
        if self.backbone is None or self.transport_head is None or self.sid_depth is None:
            return []
        beams: List[Tuple[List[int], float]] = [([], 0.0)]
        for _step in range(int(self.sid_depth)):
            prefixes = [[0] + seq for seq, _score in beams]
            max_len = max(len(p) for p in prefixes)
            dec = np.zeros((len(prefixes), max_len), dtype=np.int64)
            for row_idx, prefix in enumerate(prefixes):
                dec[row_idx, -len(prefix) :] = np.asarray(prefix, dtype=np.int64)
            dec_input = torch.tensor(dec, dtype=torch.long, device=self.device)
            enc_input = input_ids.repeat(len(prefixes), 1)
            enc_attn = attn.repeat(len(prefixes), 1)
            logits, hidden = self.backbone.decode_with_hidden(
                input_ids=enc_input,
                attention_mask=enc_attn,
                decoder_input_ids=dec_input,
            )
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            credit_logits = self.transport_head.score_all_tokens(hidden[:, -1, :])
            step_scores = log_probs + float(self.decode_transport_scale) * credit_logits
            if step_scores.shape[1] > 0:
                step_scores[:, 0] = -1e9

            candidates: List[Tuple[List[int], float]] = []
            for beam_idx, (seq, base_score) in enumerate(beams):
                allowed = self._allowed_next_tokens(seq)
                if not allowed:
                    continue
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=self.device)
                allowed_scores = step_scores[beam_idx, allowed_tensor]
                topk = min(int(self.decode_transport_topk), int(allowed_scores.shape[0]))
                if topk <= 0:
                    continue
                vals, idxs = torch.topk(allowed_scores, k=topk, dim=0)
                for local_pos in range(topk):
                    token = int(allowed_tensor[int(idxs[local_pos])].item())
                    score = float(base_score + float(vals[local_pos].item()))
                    candidates.append((seq + [token], score))
            if not candidates:
                break
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[: int(self.beam_width)]
            beams = candidates
        return beams

    def _build_history_tokens(self, history: Sequence[int]):
        if self.iid2sid_tok is None:
            return None, None

        seq = [int(i) for i in history[-self.max_hist_items :]]
        toks: List[int] = []
        for iid in seq:
            if 0 <= iid < self.n_items:
                toks.extend(self.iid2sid_tok[iid + 1].tolist())

        max_len = self.max_hist_items * self.sid_depth
        if len(toks) > max_len:
            toks = toks[-max_len:]
        pad_len = max_len - len(toks)
        input_ids = [0] * pad_len + toks
        attn = [0] * pad_len + [1] * len(toks)
        return input_ids, attn

    @torch.no_grad()
    def _score_with_tiger(self, user_id: int, history_items: Optional[Sequence[int]] = None):
        score_map: Dict[int, float] = {}
        if history_items is None:
            history = self.data.train_user_list.get(user_id, [])
        else:
            history = [int(i) for i in history_items]
        hist_set = set(int(i) for i in history)
        hist_tail = [int(i) for i in history[-self.attr_history_items :] if 0 <= int(i) < self.n_items]

        input_ids, attn = self._build_history_tokens(history)
        if input_ids is None:
            return score_map

        cur_input = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        cur_attn = torch.tensor([attn], dtype=torch.long, device=self.device)

        for _ in range(self.decode_rounds):
            use_transport_decode = (
                self.transport_head is not None
                and self.transport_enabled != "off"
            )
            ranked_sequences: List[Tuple[List[int], float]] = []
            if use_transport_decode:
                ranked_sequences = self._transport_guided_sequences(cur_input, cur_attn)
            if ranked_sequences:
                best_raw = max(float(score) for _, score in ranked_sequences)
                gen = [seq for seq, _ in ranked_sequences]
                raw_scores = [float(np.exp(float(score) - best_raw)) for _, score in ranked_sequences]
            else:
                gen_out = self.backbone.generate(
                    input_ids=cur_input,
                    attention_mask=cur_attn,
                    num_beams=self.beam_width,
                    num_return_sequences=self.beam_width,
                    max_length=self.sid_depth + 1,
                    early_stopping=True,
                    do_sample=False,
                )
                if gen_out.ndim != 2 or gen_out.shape[0] == 0:
                    break
                gen = gen_out[:, 1 : 1 + self.sid_depth].detach().cpu().numpy().tolist()
                raw_scores = [float(self.beam_width - rank) for rank in range(len(gen))]

            best_new_iid = None
            for rank, seq_tok in enumerate(gen):
                tok = tuple(int(x) for x in seq_tok)
                if not tok or any(x <= 0 for x in tok):
                    continue
                iid = self.sid2iid.get(tok, None)
                if iid is None or iid in hist_set:
                    continue
                beam_score = float(raw_scores[rank]) if rank < len(raw_scores) else float(self.beam_width - rank)
                if self.attr_profile == "scope_attrv3" and hist_tail:
                    attr = self.get_candidate_attribution(int(iid), history_items=hist_tail, user_id=int(user_id))
                    transport_page_credit = float(attr.get("pred_page_credit", 0.0))
                    beam_mult = (
                        1.0
                        + self.decode_attr_scale * float(attr.get("decode_bonus", 0.0))
                        + self.decode_cf_scale * float(attr.get("residual_support", 0.0))
                        + self.decode_block_scale * float(np.dot(
                            np.asarray(attr.get("block_weights", []), dtype=np.float32),
                            np.clip(np.asarray(attr.get("block_support", []), dtype=np.float32), 0.0, 1.0),
                        ))
                        + 0.5 * self.decode_transport_scale * transport_page_credit
                    )
                    beam_score = beam_score * max(float(beam_mult), 0.05)
                score_map[iid] = score_map.get(iid, 0.0) + beam_score
                if best_new_iid is None:
                    best_new_iid = iid

            if best_new_iid is None:
                break

            # Autoregressive-style rolling history update.
            next_tok = self.iid2sid_tok[best_new_iid + 1].tolist()
            flat = cur_input[0].detach().cpu().numpy().tolist() + next_tok
            max_len = self.max_hist_items * self.sid_depth
            flat = flat[-max_len:]
            pad = max_len - len(flat)
            flat = [0] * pad + flat
            mask = [0 if t == 0 else 1 for t in flat]
            cur_input = torch.tensor([flat], dtype=torch.long, device=self.device)
            cur_attn = torch.tensor([mask], dtype=torch.long, device=self.device)
            hist_set.add(best_new_iid)

        return score_map

    def _get_user_score(self, user_id: int, history_items: Optional[Sequence[int]] = None):
        use_cache = history_items is None
        if use_cache and user_id in self.score_cache:
            return self.score_cache[user_id]

        base = self.pop_coef * self.popularity + self.novel_coef * self.novelty
        score = base.copy().astype(np.float32)

        if self.backbone is not None and self.iid2sid_tok is not None:
            rec_scores = self._score_with_tiger(int(user_id), history_items=history_items)
            if rec_scores:
                max_s = max(float(v) for v in rec_scores.values())
                if max_s > 0:
                    for iid, v in rec_scores.items():
                        score[int(iid)] += float(v) / max_s

        if use_cache:
            self.score_cache[user_id] = score
        return score

    def cuda(self, device):
        self.device = torch.device(device)
        if self.backbone is not None:
            self.backbone = self.backbone.to(self.device)
        if self.transport_head is not None:
            self.transport_head = self.transport_head.to(self.device)
        return self

    def predict(self, users, items=None, histories: Optional[Sequence[Optional[Sequence[int]]]] = None):
        if items is None:
            items = np.arange(self.n_items, dtype=np.int64)
        else:
            items = np.asarray(items, dtype=np.int64)

        out = np.zeros((len(users), len(items)), dtype=np.float32)
        for i, uid in enumerate(users):
            history = None
            if histories is not None and i < len(histories):
                history = histories[i]
            full = self._get_user_score(int(uid), history_items=history)
            out[i] = full[items]
        return out
