"""Pilot for Item 1: GRU baseline (M0) forecasting LLM request load.
Last 30 minutes -> requests/min for each of the next 5 minutes.
Time split: 70% train / 15% val / 15% test (test NOT used here).
"""
import io, sys, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn

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

class GRUForecaster(nn.Module):
    def __init__(s, h=64):
        super().__init__(); s.gru = nn.GRU(4, h, batch_first=True); s.head = nn.Linear(h, HOR)
    def forward(s, x):
        out, _ = s.gru(x); return s.head(out[:, -1])

def mae(m, X, y):
    m.eval()
    with torch.no_grad(): return (denorm(m(X)) - y).abs().mean().item()

res = {}
# 1) overfit-one-batch check
m = GRUForecaster(); opt = torch.optim.Adam(m.parameters(), 1e-2)
xb, yb = Xtr[:32], norm(ytr[:32])
for _ in range(300):
    opt.zero_grad(); l = nn.functional.mse_loss(m(xb), yb); l.backward(); opt.step()
res["overfit_one_batch_final_mse"] = round(l.item(), 5)

# 2) short training run
m = GRUForecaster(); opt = torch.optim.Adam(m.parameters(), 1e-3)
t0 = time.time(); curve = []
for ep in range(EPOCHS):
    m.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i+64]; opt.zero_grad()
        l = nn.functional.mse_loss(m(Xtr[idx]), norm(ytr[idx])); l.backward(); opt.step()
    curve.append(round(mae(m, Xva, yva), 1))
res["train_time_s"] = round(time.time() - t0, 1)
res["val_mae_per_epoch"] = curve
res["M0_val_MAE_req_per_min"] = curve[-1]

# simple reference: next 5 minutes = last observed minute
lastv = torch.expm1(Xva[:, -1, 0] * sd[0] + mu[0]).unsqueeze(1).repeat(1, HOR)
res["persistence_val_MAE_req_per_min"] = round((lastv - yva).abs().mean().item(), 1)
res["val_mean_req_per_min"] = round(yva.mean().item(), 1)

# 3) cost
res["params"] = sum(p.numel() for p in m.parameters())
def size_kb(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return round(b.getbuffer().nbytes / 1024, 1)
def latency(model, reps=1000):
    x = Xva[:1]; model.eval(); ts = []
    with torch.no_grad():
        for _ in range(100): model(x)
        for _ in range(reps):
            t = time.perf_counter(); model(x); ts.append((time.perf_counter() - t) * 1000)
    q = np.percentile(ts, [25, 50, 75]); return round(q[1], 3), round(q[2] - q[0], 3)
res["size_kB"] = size_kb(m)
res["cpu_latency_ms_median_IQR"] = latency(m)

# 4) feasibility check for M2: INT8 dynamic quantization on the GRU
q = torch.ao.quantization.quantize_dynamic(m, {nn.GRU, nn.Linear}, dtype=torch.qint8)
res["int8_check_size_kB"] = size_kb(q)
res["int8_check_val_MAE"] = round(mae(q, Xva, yva), 1)
res["int8_check_latency_ms_median_IQR"] = latency(q)
res["windows"] = {"train": len(Xtr), "val": len(Xva)}
print(json.dumps(res, indent=1))