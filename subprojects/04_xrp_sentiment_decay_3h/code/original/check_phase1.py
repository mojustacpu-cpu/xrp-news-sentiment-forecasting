from pathlib import Path
import json,hashlib,sys,joblib
import numpy as np,pandas as pd,torch
P=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research/news-enhanced-crypto-research/phase1');sys.path.insert(0,str(P/'reference'));from neural import Model
v=np.load(P/'main_validation.npy');t=np.load(P/'main_train.npy');e=np.load(P/'main_evaluation.npy');baseline=pd.read_csv(P/'VALIDATION_ORIGINS.csv',float_precision='round_trip');torch.set_num_threads(4)
checks=[]
for name in ['E1B','E2_h3','E2_h6','E2_h12','E2_h24']:
 cfg=json.loads((P/name/'CONFIG.json').read_text());d=pd.read_csv(cfg['dataset'],float_precision='round_trip');sc=joblib.load(cfg['scalers']);f=cfg['features'];assert d.iloc[v][baseline.columns].reset_index(drop=True).equals(baseline)
 assert d.iloc[e].target_close_plus24h.isna().all();assert np.isfinite(d.iloc[np.r_[t,v]].target_close_plus24h).all()
 x=sc['input_scaler'].transform(d[f]);assert np.isfinite(x).all();assert t.min()>=29
 torch.manual_seed(42);m=Model(len(f),dict(hidden=128,bidirectional=True));xx=torch.tensor(np.stack([x[i-29:i+1] for i in t[:2]]),dtype=torch.float32);pred=m(xx);assert pred.shape==(2,1) and torch.isfinite(pred).all();pred.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters())
 checks.append(dict(candidate=name,parameters=cfg['parameters'],inputs=len(f),validation_origins=2136,final_targets_redacted=True,finite_forward_backward=True))
(P/'PREFLIGHT_VERIFICATION.json').write_text(json.dumps(checks,indent=2))
manifest=json.loads((P/'FROZEN_MANIFEST.json').read_text())
for file in ['train_candidate.py','pipeline.py','PREFLIGHT_VERIFICATION.json']:manifest['files'][str(P/file)]=hashlib.sha256((P/file).read_bytes()).hexdigest()
(P/'FROZEN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(checks,indent=2))
