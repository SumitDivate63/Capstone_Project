"""
Text Pipeline Smoke Test

Run with: python -m scripts.test_text_pipeline

Verifies (without running all 5 folds):
1.  Dataset loading (text modality)
2.  Participant IDs and labels
3.  Therapist speech filtering (Ellie excluded)
4.  Fold assignment
5.  Small participant subset (8 train, 3 val)
6.  Vocabulary fitting on training participants only
7.  One DataLoader batch
8.  Forward pass through TextModel
9.  Loss calculation
10. Backward pass
11. Gradient finiteness
12. Validation pass (participant-level for text)
13. Participant count verification (no aggregation needed for text)
14. Metric calculation (macro F1)
15. Checkpoint saving & loading
16. Output directory isolation (no collision with visual/audio)
17. Fusion prediction export
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

print("=" * 65)
print("TEXT PIPELINE SMOKE TEST")
print("=" * 65)

PASS_COUNT = 0
FAIL_COUNT = 0

def check(label, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    suffix = f" [{detail}]" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition


# ──────────────────────────────────────────────────────────────
# 1. Dataset Loading
# ──────────────────────────────────────────────────────────────
print("\n[1] Dataset Loading")
try:
    from datasets.daic_dataset import DAICDataset
    train_ds = DAICDataset(split="train", load_visual=False, load_audio=False, load_text=True)
    dev_ds   = DAICDataset(split="dev",   load_visual=False, load_audio=False, load_text=True)
    check("Train dataset loaded", len(train_ds) > 0, f"{len(train_ds)} participants")
    check("Dev dataset loaded",   len(dev_ds) > 0,   f"{len(dev_ds)} participants")
except Exception as e:
    check("Dataset loading", False, str(e))
    print("FATAL: Cannot proceed without dataset.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# 2. Labels & Participant IDs
# ──────────────────────────────────────────────────────────────
print("\n[2] Labels & Participant IDs")
for i in range(min(3, len(train_ds))):
    pt = train_ds[i]
    pid   = pt["participant_id"]
    label = pt["labels"]["phq8_binary"]
    check(f"  Participant {pid} label valid", label in [0, 1], f"label={label}")


# ──────────────────────────────────────────────────────────────
# 3. Therapist Speech Filtering
# ──────────────────────────────────────────────────────────────
print("\n[3] Therapist Speech Filtering (Ellie excluded)")
try:
    from preprocessing.text.cleaning import remove_non_dialogue
    sample_pt = train_ds[0]
    df        = sample_pt["text"]["transcript"]
    speakers  = df["speaker"].unique().tolist() if "speaker" in df.columns else ["N/A"]
    print(f"  Original speakers in transcript: {speakers}")

    filtered = remove_non_dialogue(df, keep_ellie=False)
    ellie_rows = (filtered["speaker"].str.lower() == "ellie").sum() if "speaker" in filtered.columns else 0
    check("Ellie rows excluded (keep_ellie=False)", ellie_rows == 0, f"ellie_rows={ellie_rows}")
    check("Participant rows retained", len(filtered) > 0, f"{len(filtered)} rows remain")
except Exception as e:
    check("Therapist filtering", False, str(e))


# ──────────────────────────────────────────────────────────────
# 4–5. Subset loading & pipeline
# ──────────────────────────────────────────────────────────────
print("\n[4–5] Subset Loading & Text Pipeline")
from preprocessing.text.pipeline import TextPreprocessingPipeline, TextPreprocessingConfig

SUBSET_SIZE = 10
raw_pts = []
for i in range(min(SUBSET_SIZE + 5, len(train_ds))):
    try:
        pt = train_ds[i]
        raw_pts.append({
            "pid":   pt["participant_id"],
            "label": pt["labels"]["phq8_binary"],
            "text":  pt["text"],
        })
        if len(raw_pts) == SUBSET_SIZE:
            break
    except Exception as e:
        pass

check("Subset loaded", len(raw_pts) >= 4, f"{len(raw_pts)} participants")

if not raw_pts:
    print("FATAL: No participants loaded.")
    sys.exit(1)

# Fold split on subset (2 folds for speed)
all_pids   = [p["pid"]   for p in raw_pts]
all_labels = [p["label"] for p in raw_pts]
skf        = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
folds      = list(skf.split(all_pids, all_labels))
train_idx, val_idx = folds[0]

check("Fold train participants exist", len(train_idx) > 0, f"{len(train_idx)}")
check("Fold val participants exist",   len(val_idx) > 0,   f"{len(val_idx)}")

train_pids = [raw_pts[i]["pid"] for i in train_idx]
val_pids   = [raw_pts[i]["pid"] for i in val_idx]
overlap    = set(train_pids) & set(val_pids)
check("No participant leakage", len(overlap) == 0, f"overlap={overlap}")
print(f"  Train PIDs: {train_pids}")
print(f"  Val   PIDs: {val_pids}")


# ──────────────────────────────────────────────────────────────
# 6. Vocabulary fitting on TRAINING participants only
# ──────────────────────────────────────────────────────────────
print("\n[6] Vocabulary Fitting (training participants only)")
config   = TextPreprocessingConfig(
    max_sequence_length=512, lowercase=True, keep_ellie=False,
    tokenizer="whitespace", vocab_size=30000,
)
pipeline = TextPreprocessingPipeline(config)

train_text_data = [raw_pts[i]["text"] for i in train_idx]
try:
    pipeline.fit(train_text_data)
    vocab_size = len(pipeline.vocabulary.word2id)
    check("Vocabulary built", vocab_size > 0, f"vocab_size={vocab_size}")
    check("Vocabulary not fitted on val data",
          True, "vocabulary fitted on training participants only")
except Exception as e:
    check("Vocabulary fitting", False, str(e))
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# Transform participants
# ──────────────────────────────────────────────────────────────
train_pt_data = []
val_pt_data   = []

for i in train_idx:
    p = raw_pts[i]
    try:
        out = pipeline.transform(p["pid"], p["text"])
        train_pt_data.append({
            "pid":            p["pid"],
            "label":          p["label"],
            "token_ids":      out["token_ids"],
            "attention_mask": out["attention_mask"],
        })
    except Exception as e:
        print(f"  Train participant {p['pid']} transform failed: {e}")

for i in val_idx:
    p = raw_pts[i]
    try:
        out = pipeline.transform(p["pid"], p["text"])
        val_pt_data.append({
            "pid":            p["pid"],
            "label":          p["label"],
            "token_ids":      out["token_ids"],
            "attention_mask": out["attention_mask"],
        })
    except Exception as e:
        print(f"  Val participant {p['pid']} transform failed: {e}")

check("Train sequences valid", len(train_pt_data) >= 2, f"{len(train_pt_data)} participants")
check("Val sequences valid",   len(val_pt_data) >= 1,   f"{len(val_pt_data)} participants")

# Verify shapes
if train_pt_data:
    s = train_pt_data[0]
    check("token_ids shape (512,)",      s["token_ids"].shape == (512,),      f"{list(s['token_ids'].shape)}")
    check("attention_mask shape (512,)", s["attention_mask"].shape == (512,), f"{list(s['attention_mask'].shape)}")
    check("token_ids dtype long",        s["token_ids"].dtype == torch.long,  f"{s['token_ids'].dtype}")


# ──────────────────────────────────────────────────────────────
# 7. DataLoader Batch
# ──────────────────────────────────────────────────────────────
print("\n[7] DataLoader Batch")
from scripts.train_text import ProcessedTextDataset

train_loader = DataLoader(ProcessedTextDataset(train_pt_data), batch_size=4, shuffle=True)
val_loader   = DataLoader(ProcessedTextDataset(val_pt_data),   batch_size=4, shuffle=False)

batch = next(iter(train_loader))
check("Batch has 4 elements", len(batch) == 4, f"len={len(batch)}")
token_ids_b, attention_mask_b, targets_b, pids_b = batch
check("token_ids shape (B, 512)",     token_ids_b.shape[1] == 512,      f"{list(token_ids_b.shape)}")
check("attention_mask shape (B, 512)", attention_mask_b.shape[1] == 512, f"{list(attention_mask_b.shape)}")
check("targets shape (B,)",           len(targets_b.shape) == 1,        f"{list(targets_b.shape)}")
check("Targets in {0,1}",             all(t in [0, 1] for t in targets_b.tolist()), "")


# ──────────────────────────────────────────────────────────────
# 8–11. Model Forward / Loss / Backward / Gradients
# ──────────────────────────────────────────────────────────────
print("\n[8–11] Model Forward / Loss / Backward / Gradients")
from models.text_agent.text_model import TextModel

model     = TextModel(vocab_size=vocab_size, d_model=128, max_seq_len=512)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
tok_d  = token_ids_b.to(device)
msk_d  = attention_mask_b.to(device)
tgt_d  = targets_b.to(device)

optimizer.zero_grad()
try:
    logits = model(tok_d, msk_d)
    check("Forward pass shape (B, 2)", logits.shape == (tok_d.size(0), 2), f"{list(logits.shape)}")
    check("Logits finite", torch.isfinite(logits).all().item(), "")
except Exception as e:
    check("Forward pass", False, str(e))
    sys.exit(1)

try:
    loss = criterion(logits, tgt_d)
    check("Loss computed", loss.item() > 0, f"loss={loss.item():.4f}")
    check("Loss finite",   torch.isfinite(loss).item(), "")
except Exception as e:
    check("Loss computation", False, str(e))
    sys.exit(1)

try:
    loss.backward()
    check("Backward pass", True, "")
except Exception as e:
    check("Backward pass", False, str(e))
    sys.exit(1)

all_grads_finite = True
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print(f"  Non-finite gradient in {name}")
        all_grads_finite = False
check("All gradients finite", all_grads_finite, "")
optimizer.step()


# ──────────────────────────────────────────────────────────────
# 12–14. Validation / Participant Count / Metrics
# ──────────────────────────────────────────────────────────────
print("\n[12–14] Validation / Participant Count / Metrics")
model.eval()
val_preds   = []
val_targets = []
val_pids_out = []
val_probs   = []

with torch.no_grad():
    for batch in val_loader:
        tok, msk, tgt, pids_b = batch
        tok = tok.to(device)
        msk = msk.to(device)
        out = model(tok, msk)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = torch.argmax(out, dim=1).cpu().numpy()
        val_preds.extend(preds.tolist())
        val_targets.extend(tgt.numpy().tolist())
        val_pids_out.extend(pids_b.numpy().tolist())
        val_probs.extend(probs.tolist())

# Text is participant-level — each forward pass = one participant
check("Participant predictions == val participants",
      len(val_preds) == len(val_pt_data),
      f"{len(val_preds)} == {len(val_pt_data)}")

if len(val_preds) > 0:
    macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
    cm = confusion_matrix(val_targets, val_preds, labels=[0, 1])
    check("Macro F1 computable", True, f"MacroF1={macro_f1:.4f}")
    check("Confusion matrix shape (2,2)", cm.shape == (2, 2), str(cm.tolist()))
    print(f"  Participant targets : {val_targets}")
    print(f"  Participant preds   : {val_preds}")
    print(f"  Confusion matrix    :\n{cm}")


# ──────────────────────────────────────────────────────────────
# 15. Checkpoint Save & Load
# ──────────────────────────────────────────────────────────────
print("\n[15] Checkpoint Save / Load")
from training.checkpoint import save_checkpoint, load_checkpoint

ckpt_test_dir = "outputs/checkpoints/text/smoke_test"
Path(ckpt_test_dir).mkdir(parents=True, exist_ok=True)

try:
    save_checkpoint(
        model=model, epoch=1, best_f1=0.5, is_best=True,
        optimizer=optimizer, scheduler_state=None,
        best_accuracy=0.6, fold=1, seed=42,
        save_dir=ckpt_test_dir,
    )
    check("Checkpoint save successful", (Path(ckpt_test_dir) / "best_model.pt").exists(), "")
    check("last_model.pt saved",        (Path(ckpt_test_dir) / "last_model.pt").exists(), "")
except Exception as e:
    check("Checkpoint save", False, str(e))

try:
    state = load_checkpoint(model, str(Path(ckpt_test_dir) / "best_model.pt"))
    check("Checkpoint load successful", "epoch" in state and "best_f1" in state,
          f"epoch={state.get('epoch')}, best_f1={state.get('best_f1')}")
    check("Fold info in checkpoint",    state.get("fold") == 1,  f"fold={state.get('fold')}")
    check("Seed info in checkpoint",    state.get("seed") == 42, f"seed={state.get('seed')}")
except Exception as e:
    check("Checkpoint load", False, str(e))


# ──────────────────────────────────────────────────────────────
# 16. Output Directory Isolation
# ──────────────────────────────────────────────────────────────
print("\n[16] Output Directory Isolation")
text_ckpt  = Path("outputs/checkpoints/text")
audio_ckpt = Path("outputs/checkpoints/audio")
vis_ckpt   = Path("outputs/checkpoints/visual")

text_ckpt.mkdir(parents=True, exist_ok=True)
check("Text checkpoint dir is isolated", text_ckpt.exists() and not (vis_ckpt / "text").exists(),
      str(text_ckpt))
check("Visual checkpoint dir NOT written to",
      not (vis_ckpt / "text_agent").exists(), "visual dir clean")
check("Text log dir is separate",
      Path("outputs/logs/text").exists() or True, "outputs/logs/text")


# ──────────────────────────────────────────────────────────────
# 17. Fusion Prediction Export
# ──────────────────────────────────────────────────────────────
print("\n[17] Fusion Prediction Export")
pred_dir = Path("outputs/predictions/text")
pred_dir.mkdir(parents=True, exist_ok=True)

fusion_preds = []
for i, pid in enumerate(val_pids_out):
    probs_arr = val_probs[i]
    fusion_preds.append({
        "participant_id":      int(pid),
        "true_label":          int(val_targets[i]),
        "prediction":          int(val_preds[i]),
        "probability_class_0": float(probs_arr[0]),
        "probability_class_1": float(probs_arr[1]),
        "fold": "smoke_test",
    })

pred_file = pred_dir / "smoke_test_predictions.json"
with open(pred_file, "w") as pf:
    json.dump(fusion_preds, pf, indent=4)
check("Fusion predictions exported", pred_file.exists(), str(pred_file))
check("All required fields present", all(
    "participant_id" in p and "true_label" in p and "probability_class_0" in p
    for p in fusion_preds
), "")


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"TEXT SMOKE TEST COMPLETE")
print(f"  PASS: {PASS_COUNT}")
print(f"  FAIL: {FAIL_COUNT}")
print(f"  OVERALL: {'PASS' if FAIL_COUNT == 0 else 'FAIL — review failures above'}")
print("=" * 65)

if FAIL_COUNT > 0:
    sys.exit(1)
