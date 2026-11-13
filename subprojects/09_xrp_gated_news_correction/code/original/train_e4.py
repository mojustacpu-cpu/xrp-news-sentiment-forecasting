from pathlib import Path
import sys,json,hashlib,time,os,random,traceback
import numpy as np,pandas as pd,torch,joblib
from torch.utils.data import TensorDataset,DataLoader
from gated_model import NewsCorrection
P=Path(__file__).resolve().parents[1]
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n')
def metrics(y,p,o):
 e=p-y;return dict(n=len(y),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),r2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)),directional_accuracy=float(np.mean(np.sign(p-o)==np.sign(y-o))),bias=float(e.mean()))
def forward(m,dl,scale):
 m.eval();parts=[[],[],[],[]]
 with torch.no_grad():
  for xx,bb,mask,yy in dl:
   result=m(xx,bb,mask,scale)
   for dest,z in zip(parts,result):dest.append(z.numpy().ravel())
 return [np.concatenate(v) for v in parts]
def main():
 assert not (P/'STARTED.json').exists(),'Existing run preserved; no automatic restart'
 frozen=json.loads((P/'FROZEN_MANIFEST.json').read_text());assert all(sha(f)==v for f,v in frozen['files'].items())
 plan=json.loads((P/'PROTOCOL.json').read_text());save(P/'STARTED.json',dict(pid=os.getpid(),started_utc=pd.Timestamp.now(tz='UTC')))
 torch.set_num_threads(plan['training']['threads']);torch.use_deterministic_algorithms(True)
 d=pd.read_csv(P/'data/TRAINING_UNSCALED.csv',float_precision='round_trip');sc=joblib.load(P/'data/scalers.joblib');ids={k:np.load(P/f'data/main_{k}.npy') for k in ['train','validation']};scale=float(sc['target_scaler'].scale_[0]);n=sc['news_scaler'].transform(d[sc['news_features']]);mask=np.load(P/'data/eligible_news_mask.npy');data={}
 for split,idx in ids.items():
  xx=torch.tensor(np.stack([n[i-29:i+1] for i in idx]),dtype=torch.float32);yy=torch.tensor(sc['target_scaler'].transform(np.log(d.target_close_plus24h.iloc[idx].to_numpy()/d.close.iloc[idx].to_numpy())[:,None]),dtype=torch.float32);mm=torch.tensor(mask[idx,None]);data[split]=(xx,mm,yy)
 rows=[];runroot=P/'runs';runroot.mkdir(exist_ok=True)
 for config in plan['models']:
  name=config['name'];(runroot/name).mkdir(exist_ok=True)
  for seed in plan['seeds']:
   r=runroot/name/f'seed{seed}';r.mkdir();save(P/'STATUS.json',dict(status='training',candidate=name,seed=seed,completed=len(rows),total=10,pid=os.getpid(),progress_file=str(r/'PROGRESS.json')))
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=NewsCorrection(len(sc['news_features']),config['gated']);assert sum(p.numel() for p in model.parameters())==config['trainable_parameters']
   cached=np.load(P/f'cache/E0_seed{seed}.npz');loaders={}
   for split,(xx,mm,yy) in data.items():
    base=torch.tensor(cached[split+'_scaled'][:,None]);ds=TensorDataset(xx,base,mm,yy);loaders[split]=DataLoader(ds,batch_size=32 if split=='train' else 256,shuffle=split=='train',generator=torch.Generator().manual_seed(seed))
   baseline=forward(model,loaders['validation'],scale)[0];assert np.array_equal(baseline,cached['validation_scaled']);save(r/'INITIALISATION.json',dict(zero_correction_matches_E0_exactly=True,baseline_validation_mse=float(np.mean((baseline-data['validation'][2].numpy().ravel())**2))))
   opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');stale=0;hist=[];start=time.time()
   with (r/'execution.log').open('x') as log:
    for epoch in range(1,201):
     model.train();total=0.;count=0;msesum=0.;penaltysum=0.
     for xx,bb,mm,yy in loaders['train']:
      opt.zero_grad();pred,gate,delta,cor=model(xx,bb,mm,scale);mse=((pred-yy)**2).mean();penalty=.001*(cor**2).mean();loss=mse+penalty;assert torch.isfinite(loss);loss.backward();opt.step();total+=loss.item()*len(xx);msesum+=mse.item()*len(xx);penaltysum+=penalty.item()*len(xx);count+=len(xx)
     vp=forward(model,loaders['validation'],scale)[0];vl=float(np.mean((vp-data['validation'][2].numpy().ravel())**2));assert np.isfinite(vl)
     if vl<best:
      best=vl;stale=0;selected=epoch
      torch.save(dict(model=model.state_dict(),seed=seed,epoch=epoch,config=config,news_features=sc['news_features'],plan_sha256=sha(P/'PROTOCOL.json'),frozen_E0_sha256=next(x['checkpoint_sha256'] for x in json.loads((P/'E0_REUSE.json').read_text())['seeds'] if x['seed']==seed)),r/'best.pt')
     else:stale+=1
     hist.append(dict(epoch=epoch,training_mse=msesum/count,training_loss=total/count,correction_penalty=penaltysum/count,validation_mse=vl,elapsed_seconds=time.time()-start));pd.DataFrame(hist).to_csv(r/'history.csv',index=False);save(r/'PROGRESS.json',dict(epoch=epoch,selected_epoch=selected,stale=stale,validation_mse=vl))
     line=f'{name} seed={seed} epoch={epoch} train={msesum/count:.9g} validation={vl:.9g} best={selected}';print(line,flush=True);log.write(line+'\n');log.flush()
     if stale>=15:break
   model.load_state_dict(torch.load(r/'best.pt',weights_only=False)['model']);vp,gates,deltas,cor=forward(model,loaders['validation'],scale);idx=ids['validation'];origin=d.close.iloc[idx].to_numpy();actual=d.target_close_plus24h.iloc[idx].to_numpy();pred=origin*np.exp(sc['target_scaler'].inverse_transform(vp[:,None]).ravel());basepred=origin*np.exp(sc['target_scaler'].inverse_transform(cached['validation_scaled'][:,None]).ravel());assert np.isfinite(pred).all();assert np.array_equal(vp[mask[idx]==0],cached['validation_scaled'][mask[idx]==0]);assert sha(P/f'cache/E0_seed{seed}.npz')==frozen['files'][str(P/f'cache/E0_seed{seed}.npz')]
   df=pd.DataFrame(dict(origin_time=d.available_at_utc.iloc[idx].to_numpy(),target_time=d.target_available_at_utc.iloc[idx].to_numpy(),origin_close=origin,actual=actual,predicted=pred,technical_predicted=basepred,news_mask=mask[idx],raw_gate=gates,effective_gate=mask[idx]*gates,raw_correction_return=deltas,applied_correction_return=cor/scale));df.to_csv(r/'validation_predictions.csv',index=False)
   result=dict(candidate=name,seed=seed,selected_epoch=selected,epochs=epoch,trainable_parameters=config['trainable_parameters'],frozen_parameters=config['frozen_parameters'],total_parameters=config['trainable_parameters']+config['frozen_parameters'],training_seconds=time.time()-start,**metrics(actual,pred,origin));save(r/'COMPLETE.json',result);rows.append(result);pd.DataFrame(rows).to_csv(P/'VALIDATION_ALL_SEEDS.csv',index=False)
 assert len(rows)==10
 summary=pd.DataFrame(rows).groupby('candidate').agg(**{f'{k}_{op}':(k,op) for k in ['rmse','mae','r2','directional_accuracy','training_seconds'] for op in ['mean','std']}).reset_index().sort_values('rmse_mean');summary.to_csv(P/'VALIDATION_COMPARISON.csv',index=False);save(P/'STATUS.json',dict(status='10_fits_complete_verification_pending',completed=10,total=10,final_evaluation_performed=False))
if __name__=='__main__':
 try:main()
 except Exception:
  save(P/'FAILURE.json',dict(pid=os.getpid(),error=traceback.format_exc()));raise
