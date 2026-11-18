from __future__ import annotations
import json, os, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

SYMBOLS = ["BTC-USD", "AAPL"]
SEQ = 30
ART = Path(os.getenv("FINRISK_LSTM_DIR", str(Path(__file__).resolve().parent / "artifacts" / "lstm")))
MODEL_FILE, META_FILE, HISTORY_FILE = ART/"lstm_volatility.pt", ART/"metadata.json", ART/"training_history.json"

class VolatilityLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, input_size)
    def forward(self, x):
        z, _ = self.lstm(x)
        z = self.dropout(z[:, -1, :])
        return self.fc2(self.relu(self.fc1(z)))

def download_returns(period="5y"):
    cols = []
    for symbol in SYMBOLS:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            raise ValueError(f"No market data for {symbol}")
        close = df["Close"]
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        ret = close.pct_change().rename(symbol)
        cols.append(ret)
    out = pd.concat(cols, axis=1, join="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(out) < 150:
        raise ValueError(f"Only {len(out)} aligned returns; need at least 150.")
    return out.astype("float32")

def make_sequences(returns):
    a = returns.to_numpy()
    X, y, dates = [], [], []
    for i in range(SEQ, len(a)):
        X.append(a[i-SEQ:i])       # information available through t-1
        y.append(np.abs(a[i]))     # target is |return[t]|, the next-day proxy
        dates.append(returns.index[i].strftime("%Y-%m-%d"))
    return np.asarray(X, np.float32), np.asarray(y, np.float32), dates

def score(y, p):
    e = p-y
    return {"mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "per_asset": {s: {"mae": float(np.mean(np.abs(e[:,j]))),
                              "rmse": float(np.sqrt(np.mean(e[:,j]**2)))}
                          for j,s in enumerate(SYMBOLS)}}

def train_model(period="5y", max_epochs=20, batch_size=32, seed=42):
    if not 1 <= max_epochs <= 50: raise ValueError("max_epochs must be 1..50")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    returns = download_returns(period)
    X, y, dates = make_sequences(returns)
    n=len(X); a=int(n*.70); b=int(n*.85)
    if a < 50 or b-a < 10 or n-b < 10: raise ValueError("Insufficient chronological split sizes")
    Xtr,Xv,Xte=X[:a],X[a:b],X[b:]
    ytr,yv,yte=y[:a],y[a:b],y[b:]
    # Fit scalers on training observations only.
    xm=Xtr.reshape(-1,2).mean(0); xs=Xtr.reshape(-1,2).std(0); xs=np.where(xs<1e-8,1,xs)
    ym=ytr.mean(0); ys=ytr.std(0); ys=np.where(ys<1e-8,1,ys)
    sx=lambda z:(z-xm)/xs
    sy=lambda z:(z-ym)/ys
    train_ds=TensorDataset(torch.tensor(sx(Xtr)),torch.tensor(sy(ytr)))
    loader=DataLoader(train_ds,batch_size=batch_size,shuffle=False)
    model=VolatilityLSTM().cpu()
    opt=torch.optim.Adam(model.parameters(),lr=.001); loss_fn=nn.MSELoss()
    xv=torch.tensor(sx(Xv)); yv_t=torch.tensor(sy(yv))
    hist={"train_loss":[],"validation_loss":[]}; best=float("inf"); best_state=None; stale=0; patience=4
    for epoch in range(max_epochs):
        model.train(); losses=[]
        for xb,yb in loader:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): vl=float(loss_fn(model(xv),yv_t).item())
        tl=float(np.mean(losses)); hist["train_loss"].append(tl); hist["validation_loss"].append(vl)
        if vl < best:
            best=vl; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    if best_state is None: raise RuntimeError("No trained model state")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred_s=model(torch.tensor(sx(Xte))).numpy()
    pred=np.maximum(pred_s*ys+ym,0)
    lstm_score=score(yte,pred)
    baseline=np.asarray([np.sqrt(2/np.pi)*np.std(seq[-20:],axis=0,ddof=1) for seq in Xte])
    baseline_score=score(yte,baseline)
    ART.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"input_size":2,"hidden_size":64},MODEL_FILE)
    meta={
      "model_name":"FinRisk LSTM Next-Day Volatility Proxy","model_version":"lstm-1.0",
      "trained_at_utc":datetime.now(timezone.utc).isoformat(),"symbols":SYMBOLS,"period_requested":period,
      "data_start":returns.index[0].strftime("%Y-%m-%d"),"data_end":returns.index[-1].strftime("%Y-%m-%d"),
      "aligned_return_observations":len(returns),"sequence_length":SEQ,
      "features":[s+"_daily_return" for s in SYMBOLS],
      "target_definition":"Next trading day's absolute close-to-close return |r[t+1]| per asset. Daily volatility proxy, not intraday realized volatility.",
      "split":{"train":len(Xtr),"validation":len(Xv),"test":len(Xte),"method":"chronological 70/15/15"},
      "architecture":{"input_shape":[SEQ,2],"lstm_units":64,"dropout":.2,"dense_units":32,"activation":"ReLU","output_units":2,"optimizer":"Adam","loss":"MSE","batch_size":batch_size,"max_epochs":max_epochs,"epochs_completed":len(hist["train_loss"]),"early_stopping_patience":patience,"seed":seed},
      "scaler":{"x_mean":xm.tolist(),"x_scale":xs.tolist(),"y_mean":ym.tolist(),"y_scale":ys.tolist(),"fit_on":"training split only"},
      "test_metrics":{"lstm":lstm_score,"rolling_volatility_baseline":baseline_score},
      "test_dates":{"start":dates[b],"end":dates[-1]}}
    META_FILE.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    HISTORY_FILE.write_text(json.dumps(hist,indent=2),encoding="utf-8")
    return {"status":"trained","model_version":meta["model_version"],"metadata":meta,"training_history":hist,"artifact_directory":str(ART)}

def status():
    if not META_FILE.exists() or not MODEL_FILE.exists(): return {"status":"not_trained","model_version":None}
    m=json.loads(META_FILE.read_text(encoding="utf-8"))
    return {"status":"trained","model_version":m["model_version"],"trained_at_utc":m["trained_at_utc"],"metadata":m}

def metrics():
    if not META_FILE.exists() or not HISTORY_FILE.exists(): raise FileNotFoundError("LSTM has not been trained")
    m=json.loads(META_FILE.read_text(encoding="utf-8")); h=json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return {"model_version":m["model_version"],"test_metrics":m["test_metrics"],"split":m["split"],"training_history":h,"dataset":{"start":m["data_start"],"end":m["data_end"],"observations":m["aligned_return_observations"],"symbols":m["symbols"]}}

def predict_latest():
    if not all(p.exists() for p in (MODEL_FILE,META_FILE)): raise FileNotFoundError("Train LSTM first")
    m=json.loads(META_FILE.read_text(encoding="utf-8")); r=download_returns("1y")
    if len(r)<SEQ: raise ValueError("Insufficient recent returns")
    sc=m["scaler"]; xm=np.array(sc["x_mean"],np.float32); xs=np.array(sc["x_scale"],np.float32)
    ym=np.array(sc["y_mean"],np.float32); ys=np.array(sc["y_scale"],np.float32)
    x=((r.iloc[-SEQ:].to_numpy(np.float32)-xm)/xs)[None,:,:]
    ck=torch.load(MODEL_FILE,map_location="cpu",weights_only=True)
    model=VolatilityLSTM(ck["input_size"],ck["hidden_size"]); model.load_state_dict(ck["state_dict"]); model.eval()
    with torch.no_grad(): z=model(torch.tensor(x)).numpy()[0]
    p=np.maximum(z*ys+ym,0)
    return {"model_version":m["model_version"],"symbols":dict(zip(SYMBOLS,[float(v) for v in p])),
            "unit":"absolute daily return decimal (0.02 = 2%)","target_definition":m["target_definition"],
            "sequence_length":SEQ,"last_observation_date":r.index[-1].strftime("%Y-%m-%d")}

