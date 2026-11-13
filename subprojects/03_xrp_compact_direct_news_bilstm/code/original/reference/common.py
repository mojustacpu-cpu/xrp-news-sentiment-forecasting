from pathlib import Path
import json,os,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1]
F=['EMA_20','MACD_Signal','SMA_20','SMA_10','EMA_10','Volatility_10','MACD','MFI','OBV','BTC_Close','open','high','low','close','volume','return','volatility','price_sma10','price_sma20']
EXOG=['BTC_Close']+F[:9]
H=24;L=30
ML='/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto_forecasting_research/.venv/bin/python'
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,default=str,allow_nan=False));q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def data():
 d=pd.read_csv(B/'data/predictors.csv');d.time=pd.to_datetime(d.time,utc=True)
 d['return']=d.close.pct_change();d['volatility']=d['return'].rolling(20).std()
 for p in [10,20]:d[f'price_sma{p}']=(d.close-d[f'SMA_{p}'])/d[f'SMA_{p}']
 return d.dropna(subset=F).reset_index(drop=True)
def arrays(fold):return {k:np.load(B/'configs'/f'{fold}_{k}.npy') for k in ['train','validation','evaluation']}
def metrics(y,p,o):
 e=p-y
 return dict(n=len(y),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),r2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),directional_accuracy=float(np.mean(np.sign(p-o)==np.sign(y-o))),bias=float(e.mean()))
def write_predictions(path,d,idx,p):
 pd.DataFrame(dict(origin_time=d.time.iloc[idx].to_numpy(),target_time=d.time.iloc[idx+H].to_numpy(),origin_close=d.close.iloc[idx].to_numpy(),actual_close=d.close.iloc[idx+H].to_numpy(),predicted_close=p)).to_csv(path,index=False)
