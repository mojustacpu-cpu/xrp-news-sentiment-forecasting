from pathlib import Path
import sys,json,time,random,os,hashlib,traceback
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'reference'))
from neural import Model,predict
from common import metrics,save,sha
import numpy as np,pandas as pd,torch,joblib
from torch.utils.data import Dataset,DataLoader
name=sys.argv[1]
assert name == 'HBAR_E1B'
D=B/name
R=D/'runs'
C=[name]
frozen=json.loads((B/'FROZEN_MANIFEST.json').read_text())
for path,digest in frozen['files'].items():
 assert sha(Path(path))==digest, f'Frozen input changed: {path}'
class Windows(Dataset):
 def __init__(self,x,y,ids):self.x=x;self.y=y;self.ids=ids
 def __len__(self):return len(self.ids)
 def __getitem__(self,k):
  i=self.ids[k];return self.x[i-29:i+1],self.y[i]
def main():
 R.mkdir(parents=True,exist_ok=True)
 assert not (R/'STARTED.json').exists(),'Existing run preserved; no automatic restart'
 manifest=json.loads((D/'CONFIG.json').read_text())
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
 # Only train/validation targets are loaded into tensors.
 y=np.zeros((len(d),1),dtype=np.float32)
 used=np.concatenate(list(ids.values()))
 y[used]=sc['target_scaler'].transform(np.log(d.iloc[used].target_close_plus24h.to_numpy()/d.iloc[used].close.to_numpy())[:,None])
 y=torch.tensor(y)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 rows=[]
 for name in C:
  features=manifest['features']
  x=torch.tensor(allx[:,[sc['feature_order'].index(f) for f in features]],dtype=torch.float32)
  for seed in plan['seeds']:
   out=R/f'{name}_seed{seed}';out.mkdir()
   save(R/'STATUS.json',dict(status='training',candidate=name,seed=seed,completed=len(rows),total=5,pid=os.getpid()))
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
   model=Model(len(features),dict(hidden=128,bidirectional=True))
   assert sum(p.numel() for p in model.parameters())==manifest['parameters']
   dl=DataLoader(Windows(x,y,ids['train']),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
   vd=DataLoader(Windows(x,y,ids['validation']),batch_size=256)
   opt=torch.optim.Adam(model.parameters(),lr=.001)
   best=float('inf');stale=0;history=[];start=time.time()
   for epoch in range(1,201):
    model.train();total=0.;n=0
    for xx,yy in dl:
     opt.zero_grad();loss=((model(xx)-yy)**2).mean();assert torch.isfinite(loss)
     loss.backward();opt.step();total+=loss.item()*len(xx);n+=len(xx)
    vp=predict(model,vd);vl=float(np.mean((vp-y[ids['validation']].numpy().ravel())**2));assert np.isfinite(vl)
    if vl<best:
     best=vl;stale=0;selected=epoch
     torch.save(dict(model=model.state_dict(),features=features,epoch=epoch,seed=seed,plan_sha256=sha(R/'PLAN.json')),out/'best.pt')
    else:stale+=1
    history.append(dict(epoch=epoch,training_mse=total/n,validation_mse=vl,elapsed_seconds=time.time()-start))
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    save(out/'PROGRESS.json',dict(epoch=epoch,selected_epoch=selected,stale=stale))
    print(name,seed,epoch,total/n,vl,flush=True)
    if stale>=15:break
   model.load_state_dict(torch.load(out/'best.pt',weights_only=False)['model'])
   pred_return=sc['target_scaler'].inverse_transform(predict(model,vd)[:,None]).ravel()
   pred=d.iloc[ids['validation']].close.to_numpy()*np.exp(pred_return)
   assert np.isfinite(pred).all()
   z=d.iloc[ids['validation']]
   result=dict(candidate=name,seed=seed,selected_epoch=selected,epochs=epoch,parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.time()-start,**metrics(z.target_close_plus24h.to_numpy(),pred,z.close.to_numpy()))
   pd.DataFrame(dict(origin_time=z.available_at_utc,target_time=z.target_available_at_utc,actual=z.target_close_plus24h,predicted=pred)).to_csv(out/'validation_predictions.csv',index=False)
   save(out/'COMPLETE.json',result);rows.append(result)
   pd.DataFrame(rows).to_csv(R/'VALIDATION_ALL_SEEDS.csv',index=False)
 summary=pd.DataFrame(rows).groupby('candidate').agg(rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),mae_mean=('mae','mean'),r2_mean=('r2','mean'),direction_mean=('directional_accuracy','mean')).reset_index()
 summary.to_csv(R/'VALIDATION_COMPARISON.csv',index=False)
 ranked=sorted(summary.to_dict('records'),key=lambda r:(r['rmse_mean'],C.index(r['candidate'])))
 save(R/'SELECTION_LOCK.json',dict(selected=ranked[0],results=ranked,selection='Fixed XRP-selected compact news transfer to HBAR; epochs chosen by HBAR validation scaled-return MSE',evaluation_performed=False,limitations='Main validation only; earlier rolling replication not yet performed. Retrospective news coverage and mixed annotation provenance remain.'))
 save(R/'STATUS.json',dict(status='training_complete_summary_pending',completed=5,total=5,evaluation_performed=False))
if __name__=='__main__':
 try:
  main()
 except Exception:
  R.mkdir(parents=True,exist_ok=True)
  save(R/'FAILURE.json',dict(error=traceback.format_exc(),pid=os.getpid()))
  raise
