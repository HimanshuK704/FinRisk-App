# FinRisk LSTM volatility module

Optional PyTorch LSTM module. It does not replace the existing statistical risk engine.

## Target and data integrity
Input is 30 aligned trading days of close-to-close daily returns for BTC-USD and AAPL. The two-output model predicts each asset's next-day absolute close-to-close return `|r[t+1]|`. This is a daily volatility proxy, **not intraday realized volatility**. At prediction time, only completed historical returns are used.

Chronological split: 70% train, 15% validation, 15% test. X and y scalers are fitted on training data only. The rolling baseline is `sqrt(2/pi) * std(last 20 returns)`. Reported MAE/RMSE use the untouched chronological test split.

Architecture: input `(30,2)` -> LSTM(64) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(2); Adam, MSE, batch 32, up to 20 epochs, early stopping patience 4.

## Install
```powershell
pip install "torch>=2.2"
```
`yfinance` is already used by FinRisk.

## Add files
Copy `app/lstm_engine.py` and `app/lstm_api.py` into the existing `app` directory. In `app/main.py`, after the existing `app = FastAPI(...)` creation, add:
```python
from app.lstm_api import router as lstm_router
app.include_router(lstm_router)
```
Do not replace existing routes or middleware.

## Train and test
Start the existing FastAPI server, then:
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/lstm/train" -ContentType "application/json" -Body '{"period":"5y","max_epochs":20,"batch_size":32}'
Invoke-RestMethod "http://127.0.0.1:8000/api/lstm/status" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://127.0.0.1:8000/api/lstm/metrics" | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/lstm/predict" | ConvertTo-Json -Depth 10
```

## Artifacts
Training creates `app/artifacts/lstm/lstm_volatility.pt`, `metadata.json`, and `training_history.json`. No raw market data is stored. Render's normal filesystem is ephemeral. For deployment, either commit the small trained artifacts after checking their size, or set `FINRISK_LSTM_DIR` to a persistent disk path and train once there. Do not claim deployed predictions until this is verified.

Academic limitation: daily absolute returns are used as a volatility proxy. True realized volatility generally requires intraday returns.
