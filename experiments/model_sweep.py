"""Fine-tune other LLM backbones on the same radiomics prompts and compare them.

Reproduces Part A (train-only standardization, mRMR, training-format prompts) and Part B
(4-bit QLoRA, LoRA rank 16 on the attention Q/V projections, class-weighted cross-entropy) of
the notebook for any backbone with a sequence-classification head, then evaluates:

  - classification on validation / calibration / test (Acc, F1, AUC, sensitivity)
  - how well softmax confidence separates correct from incorrect predictions
  - LTT on the softmax score, with the notebook's thresholds and risks (R1 <= 5%, R2 <= 20%)
  - the exploratory miss-rate guarantee (share of all cancers sent to auto-benign <= 5% / 10%)

One model per call, so long runs can be split across Colab sessions. Results go to
<out>/<model>/ and one row per run is appended to <out>/summary.csv.

Run from the repo root:

    python experiments/model_sweep.py --model Qwen/Qwen2.5-3B
    python experiments/model_sweep.py --model Qwen/Qwen2.5-7B --batch 4 --grad-accum 4
    python experiments/model_sweep.py --model Qwen/Qwen2.5-1.5B --n-features 16 --max-length 512
    python experiments/model_sweep.py --dry-run          # data + prompts only, no GPU
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from conformal_triage import calibrate_triage, predict_malignant_proba, render_prompt  # noqa: E402

SPLITS = ["train", "validation", "calibration", "test"]
META_COLUMNS = ["id", "sample_id", "image_name", "official_split", "analysis_split",
                "label", "label_name", "mask_source", "source_split"]

# Same candidate thresholds as notebook D6, fixed in advance.
P_T_GRID = [round(x, 2) for x in np.arange(0.05, 0.50, 0.05)]
P_U_GRID = [round(x, 2) for x in np.arange(0.95, 0.50, -0.05)]
MISS_GRID = [round(x, 2) for x in np.arange(0.01, 0.51, 0.01)]


# ----------------------------------------------------------------------------- data

def prepare_data(n_features: int, seed: int):
    """Part A10-A11: train-only imputation, z-scoring and mRMR; training-format prompts."""
    from mrmr import mrmr_classif

    tables = {s: pd.read_csv(REPO / "features" / f"radiomics_{s}.csv") for s in SPLITS}
    train = tables["train"]
    feature_columns = [c for c in train.columns
                       if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(train[c])]
    raw = {s: t.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan)
           for s, t in tables.items()}
    valid = [c for c in feature_columns
             if raw["train"][c].notna().sum() > 0 and raw["train"][c].nunique(dropna=True) > 1]

    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    scaled = {"train": pd.DataFrame(scaler.fit_transform(imputer.fit_transform(raw["train"][valid])),
                                    columns=valid)}
    for s in SPLITS[1:]:
        scaled[s] = pd.DataFrame(scaler.transform(imputer.transform(raw[s][valid])), columns=valid)

    np.random.seed(seed)
    selected = list(mrmr_classif(X=scaled["train"], y=pd.Series(train["label"].astype(int)),
                                 K=min(n_features, len(valid)), show_progress=False))

    data = {}
    for s in SPLITS:
        X = scaled[s][selected].to_numpy(dtype=np.float32)
        meta = tables[s][["id", "official_split", "label"]].reset_index(drop=True)
        data[s] = pd.DataFrame({
            "sample_id": meta["official_split"].astype(str) + "/" + meta["id"].astype(str),
            "label": meta["label"].astype(int),
            "prompt": [render_prompt(selected, x) for x in X],
        })
    return data, selected


# ----------------------------------------------------------------------------- model

def train_model(args, data):
    import torch
    import torch.nn as nn
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    # Q/V projections as in the notebook; architectures with a fused projection (e.g. Phi-3)
    # only expose qkv_proj.
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = args.target_modules.split(",") if args.target_modules else (
        ["q_proj", "v_proj"] if {"q_proj", "v_proj"} <= names else ["qkv_proj"])
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=targets))
    model.print_trainable_parameters()

    class PromptDataset(Dataset):
        def __init__(self, df):
            self.texts, self.labels = df["prompt"].tolist(), df["label"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=args.max_length,
                            padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(self.labels[i], dtype=torch.long)}

    longest = max(len(tokenizer(p)["input_ids"]) for s in SPLITS for p in data[s]["prompt"])
    print(f"Longest prompt: {longest} tokens (max_length {args.max_length})")
    if longest > args.max_length:
        raise ValueError(f"Prompts would be truncated; raise --max-length above {longest}.")

    counts = data["train"]["label"].value_counts().sort_index()
    class_weights = torch.tensor((counts.sum() / (2.0 * counts)).values, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # The loss below is a per-batch mean, so the Trainer must divide by the
            # accumulation steps itself (see notebook B5).
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=class_weights.to(out.logits.device))(out.logits, labels)
            return (loss, out) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        return {"auc": roc_auc_score(labels, probs)}

    steps_per_epoch = math.ceil(len(data["train"]) / (args.batch * args.grad_accum))
    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.run_dir / "checkpoints"), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch * 2,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
            lr_scheduler_type="cosine", warmup_steps=math.ceil(0.05 * steps_per_epoch * args.epochs),
            logging_steps=20, save_strategy="epoch", eval_strategy="epoch", save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model="auc",
            fp16=True, gradient_checkpointing=True, report_to="none", seed=args.seed),
        train_dataset=PromptDataset(data["train"]), eval_dataset=PromptDataset(data["validation"]),
        compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(args.run_dir / "adapter"))
    pd.DataFrame(trainer.state.log_history).to_csv(args.run_dir / "train_log.csv", index=False)
    return trainer.model, tokenizer


# ----------------------------------------------------------------------------- evaluation

def first_failure_threshold(scores, errors, n_total, grid, alpha, delta):
    """Fixed-sequence binomial test of P(errors among the scores <= t) / n_total <= alpha."""
    best = None
    for t in grid:
        k = int((errors & (scores <= t)).sum())
        if binom.cdf(k, n_total, alpha) > delta:
            break
        best = t
    return best


def evaluate(probs: dict, labels: dict, delta: float = 0.05) -> dict:
    row = {}
    for s in ["validation", "calibration", "test"]:
        p, y = probs[s], labels[s]
        pred = (p >= 0.5).astype(int)
        row.update({f"{s}_acc": accuracy_score(y, pred), f"{s}_f1": f1_score(y, pred),
                    f"{s}_auc": roc_auc_score(y, p), f"{s}_sens": recall_score(y, pred),
                    f"{s}_auc_correct_softmax": roc_auc_score(pred == y, np.abs(p - 0.5))})

    pc, yc, pt, yt = probs["calibration"], labels["calibration"], probs["test"], labels["test"]

    rule = calibrate_triage(pc, yc, P_T_GRID, P_U_GRID, 0.05, 0.20, delta)
    dec = rule.decide(pt)
    row.update({
        "ltt_t": rule.t, "ltt_u": rule.u,
        "ltt_test_auto_benign": int((dec == 0).sum()), "ltt_test_auto_malignant": int((dec == 1).sum()),
    })

    for alpha in (0.05, 0.10):
        tag = f"miss{int(alpha * 100)}"
        t = first_failure_threshold(pc, yc == 1, int((yc == 1).sum()), MISS_GRID, alpha, delta)
        row[f"{tag}_t"] = t
        if t is not None:
            row[f"{tag}_cal_automation"] = float((pc <= t).mean())
            row[f"{tag}_test_automation"] = float((pt <= t).mean())
            row[f"{tag}_test_miss_rate"] = float(((pt <= t) & (yt == 1)).sum() / (yt == 1).sum())
    return row


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="Hugging Face model id")
    ap.add_argument("--n-features", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8, help="per-device train batch size")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--target-modules", default="", help="comma-separated; default: q_proj,v_proj or qkv_proj")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO / "outputs" / "model_sweep"),
                    help="output folder; point it at Google Drive to survive the Colab session")
    ap.add_argument("--dry-run", action="store_true", help="prepare data and prompts only")
    args = ap.parse_args()

    data, selected = prepare_data(args.n_features, args.seed)
    print(f"{len(selected)} features (mRMR on train): {selected}")
    print("Example prompt:", data["validation"]["prompt"].iloc[0])
    if args.dry_run:
        return

    tag = f"{args.model.split('/')[-1]}_f{args.n_features}_e{args.epochs}_s{args.seed}"
    args.run_dir = Path(args.out) / tag
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()} | {"selected_features": selected}, indent=2))

    start = time.time()
    model, tokenizer = train_model(args, data)
    probs, labels = {}, {}
    for s in ["validation", "calibration", "test"]:
        probs[s] = predict_malignant_proba(model, tokenizer, data[s]["prompt"].tolist(),
                                           batch_size=args.batch * 2, max_length=args.max_length)
        labels[s] = data[s]["label"].to_numpy()
        data[s].assign(prob_malignant=probs[s]).drop(columns="prompt").to_csv(
            args.run_dir / f"probs_{s}.csv", index=False)

    row = {"run": tag, "model": args.model, "n_features": len(selected), "epochs": args.epochs,
           "minutes": round((time.time() - start) / 60, 1), **evaluate(probs, labels)}
    pd.Series(row).to_json(args.run_dir / "metrics.json", indent=2)
    summary = Path(args.out) / "summary.csv"
    pd.DataFrame([row]).to_csv(summary, mode="a", header=not summary.exists(), index=False)
    print(pd.Series(row).to_string())
    print("Saved:", args.run_dir, "and", summary)


if __name__ == "__main__":
    main()
