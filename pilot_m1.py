"""Item 1 feasibility check for M1 and M2.
M1 = GRU + temporal attention. M2 = M1 + post-training dynamic INT8 quantization.
Same data, split, windows, seed and training settings as pilot_m0.py.
Usage: python pilot_m1.py data/per_minute.csv
"""
import io, sys, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")

SEED, WIN, HOR, EPOCHS = 42, 30, 5, 10
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(1)

d = pd.read_csv(sys.argv[1], index_col=0, parse_dates=True)
req = d["req"].values.astype(np.float32)
tok = (d["ctx"] + d["gen"]).values.astype(np.float32)
mins = d.index.hour * 60 + d.index.minute
feat = np.stack([np.log1p(req), np.log1p(tok),
                 np.sin(2*np.pi*mins/1440), np.cos(2*np.pi*mins/1440)], 1).astype(np.float32)
n = len(req); n_tr, n_va = int(.70*n), int(.15*n)
mu, sd = feat[:n_tr].mean(0), feat[:n_tr].std(0) + 1e-6
X_all = (feat - mu) / sd
y_mu, y_sd = np.log1p(req[:n_tr]).mean(), np.log1p(req[:n_tr]).std()

def windows(lo, hi):
    xs, ys = [], []
    for t in range(lo + WIN, hi - HOR + 1):
        xs.append(X_all[t-WIN:t]); ys.append(req[t:t+HOR])
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys))
Xtr, ytr = windows(0, n_tr)
Xva, yva = windows(n_tr - WIN, n_tr + n_va)
norm = lambda y: (torch.log1p(y) - y_mu) / y_sd
denorm = lambda z: torch.expm1(z * y_sd + y_mu)

class GRUAttention(nn.Module):
    """M1: score every GRU hidden state, softmax the scores, use the weighted sum."""
    def __init__(s, h=64):
        super().__init__()
        s.gru = nn.GRU(4, h, batch_first=True)
        s.score = nn.Linear(h, 1)          # one score per minute (adds 65 parameters)
        s.head = nn.Linear(h, HOR)
    def forward(s, x):
        out, _ = s.gru(x)                          # (batch, 30, 64)
        w = torch.softmax(s.score(out), dim=1)     # (batch, 30, 1) attention weights
        ctx = (w * out).sum(1)                     # weighted sum over the 30 minutes
        return s.head(ctx)

def mae(m, X, y):
    m.eval()
    with torch.no_grad(): return (denorm(m(X)) - y).abs().mean().item()
def size_kb(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return round(b.getbuffer().nbytes / 1024, 1)
def latency(model, reps=1000):
    x = Xva[:1]; model.eval(); ts = []
    with torch.no_grad():
        for _ in range(100): model(x)
        for _ in range(reps):
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1000)
    q = np.percentile(ts, [25, 50, 75]); return round(q[1], 3), round(q[2] - q[0], 3)

res = {}
m = GRUAttention(); opt = torch.optim.Adam(m.parameters(), 1e-3)
t0 = time.time(); curve = []
for ep in range(EPOCHS):
    m.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i+64]; opt.zero_grad()
        l = nn.functional.mse_loss(m(Xtr[idx]), norm(ytr[idx])); l.backward(); opt.step()
    curve.append(round(mae(m, Xva, yva), 1))
res["M1_train_time_s"] = round(time.time() - t0, 1)
res["M1_val_mae_per_epoch"] = curve
res["M1_val_MAE"] = curve[-1]
res["M1_params"] = sum(p.numel() for p in m.parameters())
res["M1_size_kB"] = size_kb(m)
res["M1_cpu_latency_ms_median_IQR"] = latency(m)

# M2: post-training dynamic INT8 quantization of the GRU and all Linear layers
# (weights stored as INT8; softmax and the weighted sum stay in FP32)
q = torch.ao.quantization.quantize_dynamic(m, {nn.GRU, nn.Linear}, dtype=torch.qint8)
res["M2_val_MAE"] = round(mae(q, Xva, yva), 1)
res["M2_size_kB"] = size_kb(q)
res["M2_cpu_latency_ms_median_IQR"] = latency(q)
print(json.dumps(res, indent=1))