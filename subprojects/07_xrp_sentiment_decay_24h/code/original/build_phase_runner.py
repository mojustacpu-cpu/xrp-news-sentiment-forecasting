from pathlib import Path
P=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research/news-enhanced-crypto-research');O=P/'phase1'
src=(P.parent/'subprojects/xrp-technical-btc-log-return-control/code/train.py').read_text()
# Preserve baseline training loop; supply frozen per-candidate datasets/scalers.
start=src.index('class Windows');end=src.index('def summarise():')
body=src[start:end]
a=body.index(' manifest=json.loads');b=body.index(' # Only train/validation targets')
body=body[:a]+''' manifest=json.loads((D/'CONFIG.json').read_text())
 plan=json.loads((B/'PROTOCOL.json').read_text())
 save(R/'PLAN.json',plan)
 save(R/'STARTED.json',dict(pid=os.getpid(),started_utc=str(pd.Timestamp.now(tz='UTC'))))
 d=pd.read_csv(manifest['dataset'],float_precision='round_trip')
 ids={k:np.load(B/f'main_{k}.npy') for k in ['train','validation']}
 av=pd.to_datetime(d.available_at_utc,utc=True)
 assert len(ids['validation'])==2136
 assert (pd.to_datetime(d.target_available_at_utc,utc=True)-av).eq(pd.Timedelta(hours=24)).all()
 assert av.iloc[ids['train'][-1]]+pd.Timedelta(hours=24)<av.iloc[ids['validation'][0]]
 sc=joblib.load(manifest['scalers'])
 assert np.array_equal(sc['fit_input_rows'],np.unique((ids['train'][:,None]-np.arange(30)).ravel()))
 assert np.array_equal(sc['fit_target_origins'],ids['train'])
 allx=sc['input_scaler'].transform(d[sc['feature_order']])
 joblib.dump(sc,R/'log_return_scalers.joblib')
'''+body[b:]
body=body.replace("manifest['technical_feature_order']+manifest['news_candidates'][name]","manifest['features']").replace("manifest['seeds']","plan['seeds']").replace('==564353',"==manifest['parameters']")
header='''from pathlib import Path
import sys,json,time,random,os,hashlib,traceback
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'reference'))
from neural import Model,predict
from common import metrics,save,sha
import numpy as np,pandas as pd,torch,joblib
from torch.utils.data import Dataset,DataLoader
name=sys.argv[1]
assert name in ['E1B','E2_h3','E2_h6','E2_h12','E2_h24']
D=B/name
R=D/'runs'
C=[name]
frozen=json.loads((B/'FROZEN_MANIFEST.json').read_text())
for path,digest in frozen['files'].items():
 assert sha(Path(path))==digest, f'Frozen input changed: {path}'
'''
footer='''if __name__=='__main__':
 try:
  main()
 except Exception:
  R.mkdir(parents=True,exist_ok=True)
  save(R/'FAILURE.json',dict(error=traceback.format_exc(),pid=os.getpid()))
  raise
'''
(O/'train_candidate.py').write_text(header+body+footer)
