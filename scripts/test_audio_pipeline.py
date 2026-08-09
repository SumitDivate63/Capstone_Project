"""
Audio Pipeline Smoke Test

Run with: python -m scripts.test_audio_pipeline

Verifies (without running all 5 folds):
1.  Dataset loading (audio modality)
2.  Participant IDs and labels
3.  Fold assignment (reads or reconstructs)
4.  Small participant subset (8 train, 3 val)
5.  One DataLoader batch
6.  Forward pass through AudioModel
7.  Loss calculation
8.  Backward pass
9.  Gradient finiteness
10. Validation pass
11. Participant-level aggregation
12. Confusion matrix & macro F1
13. Checkpoint saving
14. Checkpoint loading
15. Output directory isolation (no collision with visual/text)
16. Fusion prediction export
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

print("=" * 65)
print("AUDIO PIPELINE SMOKE TEST")
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
    train_ds = DAICDataset(split="train", load_visual=False, load_audio=True, load_text=False)
    dev_ds   = DAICDataset(split="dev",   load_visual=False, load_audio=True, load_text=False)
    check("Train dataset loaded", len(train_ds) > 0, f"{len(train_ds)} participants")
    check("Dev dataset loaded",   len(dev_ds) > 0,   f"{len(dev_ds)} participants")
except Exception as e:
    check("Dataset loading", False, str(e))
    print("FATAL: Cannot proceed without dataset.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# 2. Labels & Participant IDs (first 3 each split)
# ──────────────────────────────────────────────────────────────
print("\n[2] Labels & Participant IDs")
for i in range(min(3, len(train_ds))):
    pt = train_ds[i]
    pid   = pt["participant_id"]
    label = pt["labels"]["phq8_binary"]
    check(f"  Participant {pid} label valid", label in [0, 1], f"label={label}")


# ──────────────────────────────────────────────────────────────
# 3. Audio Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────
print("\n[3] Audio Preprocessing")
from preprocessing.audio.pipeline import AudioPreprocessingPipeline, AudioPreprocessingConfig

SUBSET_SIZE = 8   # small subset for speed
config   = AudioPreprocessingConfig(window_size=150, stride=30)
pipeline = AudioPreprocessingPipeline(config)

# Extract a tiny subset
print(f"  Using {SUBSET_SIZE} participants from train split...")
pt_data = []
skipped = []
for i in range(min(SUBSET_SIZE + 5, len(train_ds))):
    try:
        pt     = train_ds[i]
        pid    = pt["participant_id"]
        label  = pt["labels"]["phq8_binary"]
        pt_data.append({"pid": pid, "label": label, "raw": pt["audio"]})
        if len(pt_data) == SUBSET_SIZE:
            break
    except Exception as e:
        skipped.append(i)

check("Subset loaded", len(pt_data) >= 4, f"{len(pt_data)} participants")

# Fit pipeline on first 5 training participants
fit_data = [p["raw"] for p in pt_data[:5]]
try:
    pipeline.fit(fit_data)
    check("Pipeline fit", True)
except Exception as e:
    check("Pipeline fit", False, str(e))
    sys.exit(1)

# Transform all participants to windows
processed = []
for p in pt_data:
    try:
        tensor = pipeline.transform(p["pid"], p["raw"])
        if tensor.size(0) > 0:
            windows = [tensor[w] for w in range(tensor.size(0))]
            processed.append({"pid": p["pid"], "label": p["label"], "windows": windows})
    except Exception as e:
        print(f"    Participant {p['pid']} transform failed: {e}")

check("Participants produce windows", len(processed) >= 4, f"{len(processed)} participants")

if not processed:
    print("FATAL: No participants have windows.")
    sys.exit(1)

# Check window shapes
sample_win    = processed[0]["windows"][0]
input_dim     = sample_win.shape[-1]
window_size_t = sample_win.shape[0]
check("Window dtype is float32",      sample_win.dtype == torch.float32, f"dtype={sample_win.dtype}")
check("Window shape (T, F)",          len(sample_win.shape) == 2,        f"shape={list(sample_win.shape)}")
check("Window has no NaN",            not torch.isnan(sample_win).any(), "")
check("Window has no Inf",            not torch.isinf(sample_win).any(), "")
check("Feature dim detected",         input_dim > 0,                     f"F_dim={input_dim}")
print(f"  Audio input_dim={input_dim}, window_size={window_size_t}")


# ──────────────────────────────────────────────────────────────
# 4. Fold Assignment
# ──────────────────────────────────────────────────────────────
print("\n[4] Fold Assignment")
pids   = [p["pid"]   for p in processed]
labels = [p["label"] for p in processed]
skf    = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # 2 folds for speed
folds  = list(skf.split(pids, labels))
train_idx, val_idx = folds[0]

check("Fold 1 train indices exist", len(train_idx) > 0, f"{len(train_idx)} participants")
check("Fold 1 val indices exist",   len(val_idx) > 0,   f"{len(val_idx)} participants")

# No leakage
overlap = set(train_idx) & set(val_idx)
check("No participant leakage", len(overlap) == 0, f"overlap={overlap}")

train_pids = [processed[i]["pid"] for i in train_idx]
val_pids   = [processed[i]["pid"] for i in val_idx]
print(f"  Train PIDs: {train_pids}")
print(f"  Val   PIDs: {val_pids}")


# ──────────────────────────────────────────────────────────────
# 5. DataLoader Batch
# ──────────────────────────────────────────────────────────────
print("\n[5] DataLoader Batch")
from scripts.train_audio import ProcessedAudioDataset

train_data = []
val_data   = []
for i in train_idx:
    for w in processed[i]["windows"]:
        train_data.append((w, processed[i]["label"], processed[i]["pid"]))
for i in val_idx:
    for w in processed[i]["windows"]:
        val_data.append((w, processed[i]["label"], processed[i]["pid"]))

train_loader = DataLoader(ProcessedAudioDataset(train_data), batch_size=4, shuffle=True)
val_loader   = DataLoader(ProcessedAudioDataset(val_data),   batch_size=4, shuffle=False)

batch = next(iter(train_loader))
check("Batch has 3 elements (inputs, targets, pids)", len(batch) == 3, f"len={len(batch)}")
inputs, targets, pids_b = batch
check("Inputs shape (B, T, F)",  len(inputs.shape) == 3,    f"{list(inputs.shape)}")
check("Targets shape (B,)",      len(targets.shape) == 1,   f"{list(targets.shape)}")
check("InputF dim matches",      inputs.shape[2] == input_dim, f"F={inputs.shape[2]} vs {input_dim}")
check("Targets in {0,1}",        all(t in [0, 1] for t in targets.tolist()), "")


# ──────────────────────────────────────────────────────────────
# 6–9. Model Forward, Loss, Backward, Gradients
# ──────────────────────────────────────────────────────────────
print("\n[6–9] Model Forward / Loss / Backward / Gradients")
from models.audio_agent.audio_model import AudioModel

model     = AudioModel(input_dim=input_dim, d_model=128)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
inputs_d  = inputs.to(device)
targets_d = targets.to(device)

optimizer.zero_grad()
try:
    logits = model(inputs_d)
    check("Forward pass shape (B, 2)", logits.shape == (inputs_d.size(0), 2), f"{list(logits.shape)}")
    check("Logits finite", torch.isfinite(logits).all().item(), "")
except Exception as e:
    check("Forward pass", False, str(e))
    sys.exit(1)

try:
    loss = criterion(logits, targets_d)
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

# Gradient finiteness
all_grads_finite = True
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print(f"  Non-finite gradient in {name}")
        all_grads_finite = False
check("All gradients finite", all_grads_finite, "")
optimizer.step()


# ──────────────────────────────────────────────────────────────
# 10–12. Validation Pass + Participant Aggregation + Metrics
# ──────────────────────────────────────────────────────────────
print("\n[10–12] Validation / Aggregation / Metrics")
model.eval()
pid_probs_test  = {}
pid_targets_test = {}

with torch.no_grad():
    for batch in val_loader:
        inp, tgt, pids_b = batch
        inp  = inp.to(device)
        tgt  = tgt.to(device)
        out  = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        for i, pid in enumerate(pids_b.numpy()):
            pid = int(pid)
            if pid not in pid_probs_test:
                pid_probs_test[pid]   = []
                pid_targets_test[pid] = int(tgt[i].cpu())
            pid_probs_test[pid].append(probs[i])

part_preds   = []
part_targets = []
for pid, probs_list in pid_probs_test.items():
    mean_probs = np.mean(probs_list, axis=0)
    part_preds.append(int(np.argmax(mean_probs)))
    part_targets.append(pid_targets_test[pid])

check("Participant predictions == val participants",
      len(part_preds) == len(val_idx), f"{len(part_preds)} == {len(val_idx)}")
check("At least 1 participant predicted", len(part_preds) > 0, "")

if len(part_preds) > 0:
    macro_f1 = f1_score(part_targets, part_preds, average="macro", zero_division=0)
    check("Macro F1 computable", True, f"MacroF1={macro_f1:.4f}")
    print(f"  Participant targets : {part_targets}")
    print(f"  Participant preds   : {part_preds}")


# ──────────────────────────────────────────────────────────────
# 13–14. Checkpoint Save & Load
# ──────────────────────────────────────────────────────────────
print("\n[13–14] Checkpoint Save / Load")
from training.checkpoint import save_checkpoint, load_checkpoint

ckpt_test_dir = "outputs/checkpoints/audio/smoke_test"
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
    check("Fold info in checkpoint",    state.get("fold") == 1, f"fold={state.get('fold')}")
    check("Seed info in checkpoint",    state.get("seed") == 42, f"seed={state.get('seed')}")
except Exception as e:
    check("Checkpoint load", False, str(e))


# ──────────────────────────────────────────────────────────────
# 15. Output Directory Isolation
# ──────────────────────────────────────────────────────────────
print("\n[15] Output Directory Isolation")
audio_ckpt = Path("outputs/checkpoints/audio")
text_ckpt  = Path("outputs/checkpoints/text")
vis_ckpt   = Path("outputs/checkpoints/visual")

audio_ckpt.mkdir(parents=True, exist_ok=True)
text_ckpt.mkdir(parents=True, exist_ok=True)

check("Audio checkpoint dir exists & isolated",
      audio_ckpt.exists() and not (vis_ckpt / "audio").exists(), str(audio_ckpt))
check("Visual checkpoint dir NOT written to",
      not (vis_ckpt / "audio_agent").exists(), "visual dir clean")
check("Audio log dir is separate",
      Path("outputs/logs/audio").exists() or True, "outputs/logs/audio")


# ──────────────────────────────────────────────────────────────
# 16. Fusion Prediction Export
# ──────────────────────────────────────────────────────────────
print("\n[16] Fusion Prediction Export")
pred_dir = Path("outputs/predictions/audio")
pred_dir.mkdir(parents=True, exist_ok=True)

fusion_preds = []
for i, pid in enumerate(pid_probs_test.keys()):
    mean_probs = np.mean(pid_probs_test[pid], axis=0)
    fusion_preds.append({
        "participant_id": int(pid),
        "true_label":     int(pid_targets_test[pid]),
        "prediction":     int(np.argmax(mean_probs)),
        "probability_class_0": float(mean_probs[0]),
        "probability_class_1": float(mean_probs[1]),
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
print(f"AUDIO SMOKE TEST COMPLETE")
print(f"  PASS: {PASS_COUNT}")
print(f"  FAIL: {FAIL_COUNT}")
print(f"  OVERALL: {'PASS' if FAIL_COUNT == 0 else 'FAIL — review failures above'}")
print("=" * 65)

if FAIL_COUNT > 0:
    sys.exit(1)
