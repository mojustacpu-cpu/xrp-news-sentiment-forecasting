"""Log-return-target hybrid; identical architecture and inputs; validation only."""
from pathlib import Path
import sys,json,time,random,os,hashlib,traceback
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B/'reference'))
from neural import Model,predict
from common import metrics,save,sha
import numpy as np,pandas as pd,torch,joblib
from torch.utils.data import Dataset,DataLoader
D=B/'data/technical_only_v1'
R=B/'runs/log_return_v1'
C=['technical_btc']
class Windows(Dataset):
 def __init__(self,x,y,ids):self.x=x;self.y=y;self.ids=ids
 def __len__(self):return len(self.ids)
 def __getitem__(self,k):
  i=self.ids[k];return self.x[i-29:i+1],self.y[i]
def main():
 R.mkdir(parents=True,exist_ok=True)
 assert not (R/'STARTED.json').exists(),'Existing run preserved; no automatic restart'
 manifest=json.loads((D/'TRAINING_MANIFEST.json').read_text())
 assert sha(Path(manifest['dataset']))==manifest['sha256']
 cfg=json.loads((B/'configs/TRAINING_PROTOCOL_V4.json').read_text())
 plan=dict(candidates=C,seeds=manifest['seeds'],fold='main',selection='Technical-only control; no news features; checkpoint epochs selected on validation only. No evaluation inference. Target is log(P[t+24]/P[t]); MinMax target scaling fit on train only. Checkpoint selection retains validation scaled-target MSE.',data_sha256=manifest['sha256'],training=cfg['training'],architecture_source_sha256=sha(B/'reference/neural.py'))
 save(R/'PLAN.json',plan)
 save(R/'STARTED.json',dict(pid=os.getpid(),started_utc=str(pd.Timestamp.now(tz='UTC'))))
 d=pd.read_csv(manifest['dataset'],float_precision='round_trip')
 ids={k:np.load(D/f'main_{k}.npy') for k in ['train','validation']}
 av=pd.to_datetime(d.available_at_utc,utc=True)
 assert (pd.to_datetime(d.target_available_at_utc,utc=True)-av).eq(pd.Timedelta(hours=24)).all()
 assert av.iloc[ids['train'][-1]]+pd.Timedelta(hours=24)<av.iloc[ids['validation'][0]]
 sc=joblib.load(D/'main_training_only_scalers.joblib')
 assert np.array_equal(sc['fit_input_rows'],np.unique((ids['train'][:,None]-np.arange(30)).ravel()))
 assert np.array_equal(sc['fit_target_origins'],ids['train'])
 allx=sc['input_scaler'].transform(d[sc['feature_order']])
 from sklearn.preprocessing import MinMaxScaler
 returns=np.log(d.loc[ids['train'],'target_close_plus24h'].to_numpy()/d.loc[ids['train'],'close'].to_numpy())
 sc['target_scaler']=MinMaxScaler().fit(returns[:,None])
 joblib.dump(sc,R/'log_return_scalers.joblib')
 # Only train/validation targets are loaded into tensors.
 y=np.zeros((len(d),1),dtype=np.float32)
 used=np.concatenate(list(ids.values()))
 y[used]=sc['target_scaler'].transform(np.log(d.iloc[used].target_close_plus24h.to_numpy()/d.iloc[used].close.to_numpy())[:,None])
 y=torch.tensor(y)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 rows=[]
 for name in C:
  features=manifest['technical_feature_order']+manifest['news_candidates'][name]
  x=torch.tensor(allx[:,[sc['feature_order'].index(f) for f in features]],dtype=torch.float32)
  for seed in manifest['seeds']:
   out=R/f'{name}_seed{seed}';out.mkdir()
   save(R/'STATUS.json',dict(status='training',candidate=name,seed=seed,completed=len(rows),total=5,pid=os.getpid()))
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
   model=Model(len(features),dict(hidden=128,bidirectional=True))
   assert sum(p.numel() for p in model.parameters())==564353
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
 save(R/'SELECTION_LOCK.json',dict(selected=ranked[0],results=ranked,selection='Technical-only log-return control; epochs chosen by validation scaled-return MSE',evaluation_performed=False,limitations='Main validation only; earlier rolling replication not yet performed. Retrospective news coverage and mixed annotation provenance remain.'))
 save(R/'STATUS.json',dict(status='training_complete_summary_pending',completed=5,total=5,evaluation_performed=False))
def summarise():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 hybrid=pd.read_csv(B/'reference/HYBRID_RETURN_VALIDATION.csv')
 tech=pd.read_csv(R/'VALIDATION_ALL_SEEDS.csv')
 data=pd.read_csv(json.loads((D/'TRAINING_MANIFEST.json').read_text())['dataset'],float_precision='round_trip')
 ids=np.load(D/'main_validation.npy');z=data.iloc[ids]
 rows=[]
 for seed in tech.seed:
  t=pd.read_csv(R/f'technical_btc_seed{seed}/validation_predictions.csv')
  h=pd.read_csv(B/f'reference/hybrid_seed{seed}_validation_predictions.csv')
  assert t.origin_time.equals(h.origin_time) and t.target_time.equals(h.target_time)
  assert np.allclose(t.actual,h.actual,rtol=0,atol=1e-12)
  for name,pred in [('technical_log_return',t.predicted),('hybrid_log_return',h.predicted)]:
   rows.append(dict(model=name,seed=int(seed),**metrics(t.actual.to_numpy(),pred.to_numpy(),z.close.to_numpy())))
  hist=pd.read_csv(R/f'technical_btc_seed{seed}/history.csv')
  fig,ax=plt.subplots(figsize=(8,4));ax.plot(hist.epoch,hist.training_mse,label='Training');ax.plot(hist.epoch,hist.validation_mse,label='Validation');ax.legend();ax.set(xlabel='Epoch',ylabel='Scaled return MSE',title=f'Technical control seed {seed}');fig.tight_layout();fig.savefig(B/f'reports/loss_seed{seed}.png');plt.close(fig)
  fig,ax=plt.subplots(figsize=(10,4));ax.plot(pd.to_datetime(t.target_time),t.actual,label='Actual',color='black');ax.plot(pd.to_datetime(t.target_time),t.predicted,label='Technical',alpha=.7);ax.plot(pd.to_datetime(h.target_time),h.predicted,label='Hybrid',alpha=.7);ax.legend();ax.set(ylabel='XRP USDT',title=f'Matched validation predictions — seed {seed}');fig.autofmt_xdate();fig.tight_layout();fig.savefig(B/f'reports/comparison_seed{seed}.png');plt.close(fig)
 results=pd.DataFrame(rows);results.to_csv(B/'reports/MATCHED_VALIDATION_ALL_SEEDS.csv',index=False)
 summary=results.groupby('model')[['rmse','mae','r2','directional_accuracy']].agg(['mean','std']);summary.to_csv(B/'reports/MATCHED_VALIDATION_SUMMARY.csv')
 pers=metrics(z.target_close_plus24h.to_numpy(),z.close.to_numpy(),z.close.to_numpy());save(B/'reports/PERSISTENCE_VALIDATION.json',pers)
 delta=100*(hybrid.rmse.mean()/tech.rmse.mean()-1)
 text='# Technical versus hybrid: matched log-return targets\n\n'
 text+=f'Five seeds on identical2136 validation origins/targets. Technical mean RMSE {tech.rmse.mean():.6f}; hybrid {hybrid.rmse.mean():.6f}. Hybrid RMSE change {delta:+.2f}%. Persistence RMSE {pers["rmse"]:.6f}.\n\n'
 text+='Both use log-return target, the same data periods, technical/BTC values, scalers for shared inputs, training settings and checkpoint criterion. Only the8news inputs are removed. Same layer design; parameter count changes naturally from572545 to564353. This is a feature ablation, not exactly parameter-matched. Five seeds are not independent market replications. No final evaluation inference. Retrospective news coverage limitations remain.\n'
 (B/'reports/MATCHED_VALIDATION_REPORT.md').write_text(text)
 save(R/'STATUS.json',dict(status='completed_validation_only',completed=5,total=5,hybrid_rmse_change_percent=float(delta),evaluation_performed=False))
if __name__=='__main__':
 try:
  main()
  summarise()
 except Exception:
  R.mkdir(parents=True,exist_ok=True)
  save(R/'FAILURE.json',dict(error=traceback.format_exc(),pid=os.getpid()))
  raise
