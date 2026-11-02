import os, glob, time, json, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report, roc_curve, auc)
from sklearn.manifold import TSNE


SEED        = 42
T_ORIG      = 1500      
DOWNSAMPLE  = 4         
T           = T_ORIG // DOWNSAMPLE
BATCH_SIZE  = 128
EPOCHS      = 25        
LR          = 1e-3
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 2
NUM_CLASSES = 18
AE_EPOCHS   = 12
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

plt.style.use("dark_background")
NEON = ["#00f5ff", "#ff00e6", "#39ff14", "#ffea00", "#ff5e00", "#7d5fff",
        "#00ff9f", "#ff2079", "#5ee1ff", "#ffd166", "#c77dff", "#06d6a0",
        "#ef476f", "#118ab2", "#ffb703", "#8338ec", "#fb5607", "#3a86ff"]
sns.set_palette(NEON)

CLASS_NAMES = ["Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", "Lay",
               "Lay-stand", "Pick", "Jump", "Push-up", "Sit-up", "Walk",
               "Walk-backward", "Walk-circle", "Run", "Stair-up",
               "Stair-down", "Table-tennis"]

OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

print("="*88)
print("AGENTIC HAR FRAMEWORK :: DEVICE =", DEVICE)
print("="*88)


search_roots = [
    "/kaggle/input/har-1",
    "/kaggle/input/datasets/arashnic/har-1",
    "/kaggle/input",
]
candidates = []
for root in search_roots:
    candidates += glob.glob(os.path.join(root, "**", "*subsample*"), recursive=True)
    candidates += glob.glob(os.path.join(root, "**", "*raw_samples*"), recursive=True)
candidates = [c for c in candidates if c.lower().endswith(".csv")]
if not candidates:
    raise FileNotFoundError("Could not locate the KU-HAR subsamples CSV under /kaggle/input. "
                             "Check the dataset is attached to this notebook.")
DATA_PATH = candidates[0]
print("Using data file:", DATA_PATH)

raw = pd.read_csv(DATA_PATH, header=None)

try:
    float(raw.iloc[0, 0])
except (ValueError, TypeError):
    raw = pd.read_csv(DATA_PATH, header=0)

data = raw.values.astype(np.float32)
assert data.shape[1] >= 9003, f"Unexpected column count: {data.shape[1]}"

signals = data[:, :9000].reshape(-1, 6, T_ORIG)          
labels  = data[:, 9000].astype(int)                      
lengths = data[:, 9001].astype(int)
serials = data[:, 9002].astype(int)

print(f"Loaded {signals.shape[0]} samples | {NUM_CLASSES} classes | raw T={T_ORIG}")


N = signals.shape[0]
signals_ds = signals.reshape(N, 6, T, DOWNSAMPLE).mean(axis=3)


X_train, X_temp, y_train, y_temp = train_test_split(
    signals_ds, labels, test_size=0.30, stratify=labels, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

ch_mean = X_train.mean(axis=(0, 2), keepdims=True)
ch_std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-6

def normalize(x):
    return (x - ch_mean) / ch_std

X_train_n, X_val_n, X_test_n = normalize(X_train), normalize(X_val), normalize(X_test)

print(f"Split -> train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")

class HARDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(HARDataset(X_train_n, y_train), batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(HARDataset(X_val_n,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(HARDataset(X_test_n,  y_test),  batch_size=BATCH_SIZE, shuffle=False)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HARFormerNet(nn.Module):
    def __init__(self, in_ch=6, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                dim_feedforward=256, dropout=0.1,
                                                batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.attn_pool_w = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x, return_embedding=False):
      
        z = self.cnn(x)                      
        z = z.transpose(1, 2)                 
        z = self.pos_enc(z)
        z = self.transformer(z)               
        attn_w = torch.softmax(self.attn_pool_w(z), dim=1)   
        embedding = (z * attn_w).sum(dim=1)   
        logits = self.head(embedding)
        if return_embedding:
            return logits, embedding
        return logits

model = HARFormerNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: HAR-FormerNet (CNN + Transformer-Encoder) | Params: {n_params:,}")


criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc, best_state = 0.0, None

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train: optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return tot_loss / total, correct / total

print("\n--- Training HAR-FormerNet ---")
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    scheduler.step()
    history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} "
          f"| val_loss={va_loss:.4f} acc={va_acc:.4f}")
train_time = time.time() - t0
model.load_state_dict(best_state)
print(f"\nBest val_acc={best_val_acc:.4f} | Total train time={train_time:.1f}s")


model.eval()
all_logits, all_true, all_embed = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits, emb = model(xb, return_embedding=True)
        all_logits.append(logits.cpu().numpy())
        all_embed.append(emb.cpu().numpy())
        all_true.append(yb.numpy())

y_logits = np.concatenate(all_logits)
y_prob   = F.softmax(torch.tensor(y_logits), dim=1).numpy()
y_pred   = y_logits.argmax(1)
y_true   = np.concatenate(all_true)
embeddings = np.concatenate(all_embed)

test_acc = accuracy_score(y_true, y_pred)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(OUT_DIR, "classification_report.csv"))

print("\n" + "="*88)
print("TEST-SET RESULTS")
print("="*88)
print(f"Accuracy           : {test_acc:.4f}")
print(f"Macro  Precision/Recall/F1 : {prec_m:.4f} / {rec_m:.4f} / {f1_m:.4f}")
print(f"Weighted Precision/Recall/F1 : {prec_w:.4f} / {rec_w:.4f} / {f1_w:.4f}")
print(report_df.round(4))

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

)
y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
macro_auc = np.mean(list(roc_auc.values()))
print(f"Macro-average ROC-AUC (OvR): {macro_auc:.4f}")


class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(32, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, in_ch, 7, stride=2, padding=3, output_padding=1),
        )
    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        if out.shape[-1] != x.shape[-1]: 
            out = F.interpolate(out, size=x.shape[-1], mode="linear", align_corners=False)
        return out

ae = ConvAutoencoder().to(DEVICE)
ae_opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
ae_loader = DataLoader(HARDataset(X_train_n, y_train), batch_size=BATCH_SIZE, shuffle=True)

print("\n--- Training unsupervised anomaly-detector (Conv1D Autoencoder) ---")
for epoch in range(1, AE_EPOCHS + 1):
    ae.train(); tot = 0.0
    for xb, _ in ae_loader:
        xb = xb.to(DEVICE)
        ae_opt.zero_grad()
        recon = ae(xb)
        loss = F.mse_loss(recon, xb)
        loss.backward(); ae_opt.step()
        tot += loss.item() * xb.size(0)
    if epoch % 4 == 0 or epoch == AE_EPOCHS:
        print(f"AE Epoch {epoch:02d}/{AE_EPOCHS} | recon_MSE={tot/len(X_train_n):.5f}")


ae.eval()
with torch.no_grad():
    tr_recon = ae(torch.tensor(X_train_n, dtype=torch.float32).to(DEVICE))
    tr_err = F.mse_loss(tr_recon, torch.tensor(X_train_n, dtype=torch.float32).to(DEVICE),
                         reduction="none").mean(dim=(1, 2)).cpu().numpy()
ANOMALY_THRESH = float(np.percentile(tr_err, 97))
print(f"Anomaly threshold (97th pct of train reconstruction error): {ANOMALY_THRESH:.5f}")


def inject_synthetic_anomalies(X, frac=0.15, seed=SEED):
    rng = np.random.RandomState(seed)
    Xa = X.copy()
    n = X.shape[0]
    idx = rng.choice(n, size=int(n * frac), replace=False)
    labels_anom = np.zeros(n, dtype=int)
    for i in idx:
        kind = rng.choice(["spike", "noise", "dropout"])
        if kind == "spike":
            pos = rng.randint(0, Xa.shape[2] - 5)
            Xa[i, :, pos:pos+5] += rng.uniform(5, 10)
        elif kind == "noise":
            Xa[i] += rng.normal(0, 3.0, size=Xa[i].shape)
        else:
            pos = rng.randint(0, Xa.shape[2] - 30)
            Xa[i, :, pos:pos+30] = 0.0
        labels_anom[i] = 1
    return Xa, labels_anom

X_test_anom_raw, y_anom_true = inject_synthetic_anomalies(X_test)
X_test_anom_n = normalize(X_test_anom_raw)

with torch.no_grad():
    recon = ae(torch.tensor(X_test_anom_n, dtype=torch.float32).to(DEVICE))
    err = F.mse_loss(recon, torch.tensor(X_test_anom_n, dtype=torch.float32).to(DEVICE),
                      reduction="none").mean(dim=(1, 2)).cpu().numpy()
y_anom_pred = (err > ANOMALY_THRESH).astype(int)
anom_prec, anom_rec, anom_f1, _ = precision_recall_fscore_support(
    y_anom_true, y_anom_pred, average="binary", zero_division=0)
anom_acc = accuracy_score(y_anom_true, y_anom_pred)
print(f"[Synthetic anomaly benchmark] acc={anom_acc:.4f} prec={anom_prec:.4f} "
      f"rec={anom_rec:.4f} f1={anom_f1:.4f}")


PROACTIVE_RULES = {
    "Sit": "Prolonged sitting detected -> suggest a short movement break.",
    "Lay": "Extended lying detected -> check inactivity duration / possible rest.",
    "Fall-risk": "Irregular motion signature -> possible fall / device drop, alerting caregiver.",
    "Stair-up": "Elevated exertion (stairs) -> monitor heart-rate proxy, hydration reminder.",
    "Stair-down": "Descending stairs -> balance caution notice.",
    "Run": "High-intensity activity -> log for fitness summary, hydration reminder.",
    "Push-up": "Strength activity detected -> log to workout tracker.",
    "Sit-up": "Strength activity detected -> log to workout tracker.",
    "default": "Activity logged to context-aware history."
}

class RealTimeHARAgent:
    """Agentic wrapper: perceives a sensor window, decides activity + anomaly
    state via the trained models, and acts with a context-aware suggestion."""
    def __init__(self, classifier, anomaly_model, mean, std, class_names,
                 anomaly_thresh, device, conf_thresh=0.55):
        self.clf, self.ae = classifier.eval(), anomaly_model.eval()

        self.mean = mean.squeeze(0)  
        self.std  = std.squeeze(0)   
        self.class_names = class_names
        self.thresh = anomaly_thresh
        self.device = device
        self.conf_thresh = conf_thresh

    def perceive(self, raw_window):
        
        x = (raw_window - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

    def decide(self, raw_window):
        x = self.perceive(raw_window)
        with torch.no_grad():
            logits = self.clf(x)
            prob = F.softmax(logits, dim=1).cpu().numpy()[0]
            recon = self.ae(x)
            err = F.mse_loss(recon, x, reduction="none").mean().item()
        cls_id = int(prob.argmax())
        return {
            "activity": self.class_names[cls_id],
            "confidence": float(prob[cls_id]),
            "anomaly_score": err,
            "is_anomalous": err > self.thresh,
            "low_confidence": prob[cls_id] < self.conf_thresh,
        }

    def act(self, decision):
        if decision["is_anomalous"]:
            msg = PROACTIVE_RULES["Fall-risk"]
        elif decision["low_confidence"]:
            msg = "Ambiguous motion pattern -> requesting more context / gentle check-in."
        else:
            msg = PROACTIVE_RULES.get(decision["activity"], PROACTIVE_RULES["default"])
        return msg

agent = RealTimeHARAgent(model, ae, ch_mean, ch_std, CLASS_NAMES,
                          ANOMALY_THRESH, DEVICE)


N_STREAM = min(300, len(X_test))
stream_idx = np.random.RandomState(SEED).choice(len(X_test), N_STREAM, replace=False)
latencies_ms, stream_log = [], []

for k, i in enumerate(stream_idx):
    window = X_test[i]
    t0 = time.perf_counter()
    decision = agent.decide(window)
    action = agent.act(decision)
    t1 = time.perf_counter()
    latencies_ms.append((t1 - t0) * 1000.0)
    stream_log.append({"step": k, "true_label": CLASS_NAMES[y_test[i]], **decision, "action": action})

latencies_ms = np.array(latencies_ms)
stream_df = pd.DataFrame(stream_log)
stream_df.to_csv(os.path.join(OUT_DIR, "agent_stream_log.csv"), index=False)

print("\n" + "="*88)
print("REAL-TIME PERFORMANCE (single-sample streaming inference)")
print("="*88)
print(f"Mean latency   : {latencies_ms.mean():.3f} ms")
print(f"P95 latency    : {np.percentile(latencies_ms, 95):.3f} ms")
print(f"Throughput     : {1000.0/latencies_ms.mean():.1f} samples/sec")
print(f"Model size     : {n_params:,} parameters")


summary = pd.DataFrame({
    "Metric": ["Test Accuracy", "Macro F1", "Weighted F1", "Macro ROC-AUC",
               "Anomaly-Detection F1 (synthetic)", "Mean Inference Latency (ms)",
               "Throughput (samples/sec)", "Trainable Parameters"],
    "Value": [f"{test_acc:.4f}", f"{f1_m:.4f}", f"{f1_w:.4f}", f"{macro_auc:.4f}",
              f"{anom_f1:.4f}", f"{latencies_ms.mean():.3f}",
              f"{1000.0/latencies_ms.mean():.1f}", f"{n_params:,}"]
})
summary.to_csv(os.path.join(OUT_DIR, "ieee_summary_table.csv"), index=False)
print("\nIEEE-STYLE SUMMARY TABLE")
print(summary.to_string(index=False))


fig = plt.figure(figsize=(22, 26), facecolor="#05060a")
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.28)

def style_ax(ax, title):
    ax.set_facecolor("#0a0c14")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="#c8d3f5")
    for spine in ax.spines.values():
        spine.set_color("#2a2f45")

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history["train_loss"], color=NEON[0], lw=2, label="train")
ax1.plot(history["val_loss"], color=NEON[1], lw=2, label="val")
style_ax(ax1, "Training / Validation Loss")
ax1.legend(facecolor="#0a0c14", labelcolor="white")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history["train_acc"], color=NEON[2], lw=2, label="train")
ax2.plot(history["val_acc"], color=NEON[3], lw=2, label="val")
style_ax(ax2, "Training / Validation Accuracy")
ax2.legend(facecolor="#0a0c14", labelcolor="white")

ax3 = fig.add_subplot(gs[1, :])
sns.heatmap(cm_norm, annot=False, cmap="mako", xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=ax3, cbar_kws={"label": "Proportion"})
ax3.set_xlabel("Predicted", color="white"); ax3.set_ylabel("True", color="white")
ax3.tick_params(colors="#c8d3f5", labelrotation=60)
style_ax(ax3, "Normalized Confusion Matrix (Test Set)")

ax4 = fig.add_subplot(gs[2, 0])
f1_per_class = [report[c]["f1-score"] for c in CLASS_NAMES]
ax4.bar(CLASS_NAMES, f1_per_class, color=NEON)
ax4.set_ylim(0, 1.05); ax4.tick_params(axis="x", labelrotation=75)
style_ax(ax4, "Per-Class F1-score")

ax5 = fig.add_subplot(gs[2, 1])
for i in range(0, NUM_CLASSES, 3):
    ax5.plot(fpr[i], tpr[i], lw=1.5, color=NEON[i % len(NEON)],
              label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:.2f})")
ax5.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax5.set_xlabel("False Positive Rate", color="white"); ax5.set_ylabel("True Positive Rate", color="white")
ax5.legend(fontsize=7, facecolor="#0a0c14", labelcolor="white", loc="lower right")
style_ax(ax5, f"ROC Curves (Macro AUC={macro_auc:.3f})")

ax6 = fig.add_subplot(gs[3, :])
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca")
emb_2d = tsne.fit_transform(embeddings[:min(2000, len(embeddings))])
lbl_2d = y_true[:min(2000, len(embeddings))]
for c in range(NUM_CLASSES):
    m = lbl_2d == c
    ax6.scatter(emb_2d[m, 0], emb_2d[m, 1], s=10, color=NEON[c % len(NEON)],
                label=CLASS_NAMES[c], alpha=0.75)
ax6.legend(fontsize=7, ncol=6, facecolor="#0a0c14", labelcolor="white", loc="upper center",
           bbox_to_anchor=(0.5, -0.08))
style_ax(ax6, "t-SNE of Learned Transformer Embeddings")

ax7 = fig.add_subplot(gs[4, 0])
ax7.hist(latencies_ms, bins=30, color=NEON[4], edgecolor="black")
ax7.axvline(latencies_ms.mean(), color=NEON[1], lw=2, linestyle="--",
            label=f"mean={latencies_ms.mean():.2f} ms")
ax7.legend(facecolor="#0a0c14", labelcolor="white")
style_ax(ax7, "Real-Time Inference Latency Distribution")

ax8 = fig.add_subplot(gs[4, 1])
anom_mask = stream_df["is_anomalous"].values
ax8.plot(stream_df["step"], stream_df["confidence"], color=NEON[5], lw=1.2, label="confidence")
ax8.scatter(stream_df["step"][anom_mask], stream_df["confidence"][anom_mask],
            color=NEON[8], s=35, label="anomaly flagged", zorder=5)
ax8.set_ylim(0, 1.05)
ax8.legend(facecolor="#0a0c14", labelcolor="white")
style_ax(ax8, "Agent Decision Timeline (Simulated Real-Time Stream)")

fig.suptitle("Agentic HAR Framework — Evaluation Dashboard (KU-HAR)",
             color="white", fontsize=18, fontweight="bold", y=0.995)
fig.savefig(os.path.join(OUT_DIR, "har_agentic_dashboard.png"), dpi=180,
            facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.show()

import matplotlib as mpl

plt.style.use("default")         
mpl.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "text.color":        "#111111",
    "xtick.color":       "#111111",
    "ytick.color":       "#111111",
    "axes.grid":         True,
    "grid.color":        "#d9d9d9",
    "grid.linewidth":    0.6,
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "legend.fontsize":   7,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "font.family":       "serif",
})


PUB = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
       "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22",
       "#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5", "#c49c94",
       "#f7b6d2", "#c7c7c7", "#dbdb8d"]

COL1 = 3.45   
COL2 = 7.16   

FIG_DIR = os.path.join(OUT_DIR, "ieee_figures")
os.makedirs(FIG_DIR, exist_ok=True)

def save_pub(fig, name, w, h):
    fig.set_size_inches(w, h)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))
axes[0].plot(history["train_loss"], color=PUB[1], lw=1.3, label="Train")
axes[0].plot(history["val_loss"],   color=PUB[0], lw=1.3, label="Val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
axes[0].set_title("(a) Training / Validation Loss")
axes[1].plot(history["train_acc"], color=PUB[1], lw=1.3, label="Train")
axes[1].plot(history["val_acc"],   color=PUB[0], lw=1.3, label="Val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
axes[1].set_title("(b) Training / Validation Accuracy")
save_pub(fig, "fig1_training_curves.png", COL2, 2.6)

fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.62))
sns.heatmap(cm_norm, cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Proportion"},
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.3, linecolor="#eeeeee", ax=ax, square=True)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.tick_params(axis="x", labelrotation=75)
ax.set_title("Normalized Confusion Matrix (Test Set)")
save_pub(fig, "fig2_confusion_matrix.png", COL2, COL2 * 0.62)

fig, ax = plt.subplots(figsize=(COL2, 2.6))
f1_per_class = [report[c]["f1-score"] for c in CLASS_NAMES]
ax.bar(CLASS_NAMES, f1_per_class, color=[PUB[i % len(PUB)] for i in range(NUM_CLASSES)],
       edgecolor="black", linewidth=0.4)
ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
ax.tick_params(axis="x", labelrotation=75)
ax.set_title("Per-Class F1-score (Test Set)")
save_pub(fig, "fig3_per_class_f1.png", COL2, 2.8)


fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.85))
show_idx = list(range(0, NUM_CLASSES, 3))
for j, i in enumerate(show_idx):
    ax.plot(fpr[i], tpr[i], lw=1.1, color=PUB[j % len(PUB)],
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:.2f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=6, loc="lower right")
ax.set_title(f"ROC Curves (Macro AUC={macro_auc:.3f})")
save_pub(fig, "fig4_roc_curves.png", COL1, COL1 * 0.9)


fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.6))
n_show = min(2000, len(embeddings))
emb_2d = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca").fit_transform(embeddings[:n_show])
lbl_2d = y_true[:n_show]
for c in range(NUM_CLASSES):
    m = lbl_2d == c
    ax.scatter(emb_2d[m, 0], emb_2d[m, 1], s=8, color=PUB[c % len(PUB)],
               label=CLASS_NAMES[c], alpha=0.8, linewidths=0)
ax.legend(fontsize=6, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
ax.set_xlabel("t-SNE dim. 1"); ax.set_ylabel("t-SNE dim. 2")
ax.set_title("t-SNE of Learned Transformer Embeddings")
save_pub(fig, "fig5_tsne_embedding.png", COL2, COL2 * 0.65)


fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.85))
ax.hist(latencies_ms, bins=30, color=PUB[1], edgecolor="black", linewidth=0.4)
ax.axvline(latencies_ms.mean(), color=PUB[0], lw=1.3, linestyle="--",
           label=f"mean = {latencies_ms.mean():.2f} ms")
ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count"); ax.legend()
ax.set_title("Real-Time Inference Latency Distribution")
save_pub(fig, "fig6_latency_distribution.png", COL1, COL1 * 0.9)

def bootstrap_metrics(n_boot=200, seed=SEED):
    rng = np.random.RandomState(seed)
    n, n_a = len(y_true), len(y_anom_true)
    accs, f1s, aucs, anomf1s = [], [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        _, _, f1b, _ = precision_recall_fscore_support(
            y_true[idx], y_pred[idx], average="macro", zero_division=0)
        f1s.append(f1b)
        ybin = label_binarize(y_true[idx], classes=list(range(NUM_CLASSES)))
        aucs_i = []
        for c in range(NUM_CLASSES):
            col = ybin[:, c]
            if 0 < col.sum() < len(col):
                fpr_i, tpr_i, _ = roc_curve(col, y_prob[idx, c])
                aucs_i.append(auc(fpr_i, tpr_i))
        aucs.append(np.mean(aucs_i) if aucs_i else np.nan)
        idx_a = rng.randint(0, n_a, n_a)
        _, _, f1a, _ = precision_recall_fscore_support(
            y_anom_true[idx_a], y_anom_pred[idx_a], average="binary", zero_division=0)
        anomf1s.append(f1a)
    return accs, f1s, [a for a in aucs if not np.isnan(a)], anomf1s

acc_b, f1_b, auc_b, anomf1_b = bootstrap_metrics()

fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.6))
box_data   = [acc_b, f1_b, auc_b, anomf1_b]
box_titles = ["(a) Accuracy", "(b) Macro F1", "(c) Macro ROC-AUC", "(d) Anomaly F1"]
for ax, data, title, color in zip(axes, box_data, box_titles, PUB[:4]):
    bp = ax.boxplot(data, widths=0.5, patch_artist=True,
                     medianprops=dict(color="black"))
    bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.75)
    ax.set_xticks([]); ax.set_title(title)
    ax.set_ylim(max(0, min(data) - 0.05), min(1.0, max(data) + 0.05))
fig.suptitle("Distribution of Evaluation Metrics (Bootstrap, n=200)", y=1.04, fontsize=10)
save_pub(fig, "fig7_metric_distributions.png", COL2, 2.8)

# ---- Fig. H: agent trajectory grid per activity class (double column, 2x4) -
# Analogous to the "trajectory comparison" figures: confidence and anomaly
# score over the streamed test samples, grouped by ground-truth activity.
REP_CLASSES = [c for c in ["Stand", "Sit", "Walk", "Run"] if c in stream_df["true_label"].unique()]
if len(REP_CLASSES) < 4:
    REP_CLASSES = list(stream_df["true_label"].value_counts().index[:4])

fig, axes = plt.subplots(2, len(REP_CLASSES), figsize=(COL2, 3.6), sharex=False)
for j, cls in enumerate(REP_CLASSES):
    sub = stream_df[stream_df["true_label"] == cls].reset_index(drop=True)
    axes[0, j].plot(sub.index, sub["confidence"], color=PUB[j % len(PUB)], lw=1.1)
    axes[0, j].axhline(agent.conf_thresh, color="gray", lw=0.7, linestyle="--")
    axes[0, j].set_title(cls, fontsize=8)
    axes[0, j].set_ylim(0, 1.05)
    if j == 0: axes[0, j].set_ylabel("Confidence")

    axes[1, j].plot(sub.index, sub["anomaly_score"], color=PUB[(j + 4) % len(PUB)], lw=1.1)
    axes[1, j].axhline(ANOMALY_THRESH, color="gray", lw=0.7, linestyle="--")
    axes[1, j].set_xlabel("Streamed sample idx")
    if j == 0: axes[1, j].set_ylabel("Anomaly score")
fig.suptitle("Agent Decision Trajectories by Activity Class", y=1.02, fontsize=10)
save_pub(fig, "fig8_agent_trajectories.png", COL2, 3.8)

# ---- Fig. I: case-comparison panel (double column, 1x3) --------------------
# Analogous to the "comparative trajectory" figure: raw 6-channel IMU signal
# for three individual windows, dual axis (accel vs gyro), annotated with the
# agent's predicted activity, confidence, and anomaly verdict.
rng_case = np.random.RandomState(SEED)
case_idx = rng_case.choice(len(X_test), 3, replace=False)
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.7))
t_axis = np.arange(T)
for k, i in enumerate(case_idx):
    window = X_test[i]
    decision = agent.decide(window)
    ax = axes[k]
    ax2 = ax.twinx()
    for ch, (name, color) in enumerate(zip(["Ax", "Ay", "Az"], PUB[:3])):
        ax.plot(t_axis, window[ch], color=color, lw=0.8, label=name)
    for ch, (name, color) in enumerate(zip(["Gx", "Gy", "Gz"], PUB[3:6])):
        ax2.plot(t_axis, window[3 + ch], color=color, lw=0.8, alpha=0.6, label=name)
    verdict = "ANOMALY" if decision["is_anomalous"] else "normal"
    ax.set_title(f"True: {CLASS_NAMES[y_test[i]]} | Pred: {decision['activity']}\n"
                 f"conf={decision['confidence']:.2f}, {verdict}", fontsize=7.5)
    ax.set_xlabel("Time step")
    if k == 0: ax.set_ylabel("Accel. (a.u.)")
    if k == 2: ax2.set_ylabel("Gyro (a.u.)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if k == 0:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5.5, ncol=2, loc="upper right")
fig.suptitle("Comparative Analysis: Sample IMU Trajectories vs. Agent Decisions", y=1.06, fontsize=10)
save_pub(fig, "fig9_case_comparison.png", COL2, 3.0)

print("\nAll artifacts saved to:", OUT_DIR)
print(" - classification_report.csv")
print(" - agent_stream_log.csv")
print(" - ieee_summary_table.csv")
print(" - har_agentic_dashboard.png (quick-look, dark theme)")
print(" - ieee_figures/fig1_training_curves.png")
print(" - ieee_figures/fig2_confusion_matrix.png")
print(" - ieee_figures/fig3_per_class_f1.png")
print(" - ieee_figures/fig4_roc_curves.png")
print(" - ieee_figures/fig5_tsne_embedding.png")
print(" - ieee_figures/fig6_latency_distribution.png")
print(" - ieee_figures/fig7_metric_distributions.png  (box-plot grid)")
print(" - ieee_figures/fig8_agent_trajectories.png    (trajectory grid)")
print(" - ieee_figures/fig9_case_comparison.png       (comparative case panel)")
print("\nDONE.")
