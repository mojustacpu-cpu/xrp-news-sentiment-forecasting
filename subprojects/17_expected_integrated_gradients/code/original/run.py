from pathlib import Path
import sys,json,time,hashlib,os
import numpy as np,pandas as pd,torch,joblib
B=Path(__file__).resolve().parents[1];ROOT=B.parents[2];P=ROOT/'news-enhanced-crypto-research/phase1';sys.path.insert(0,str(P/'reference'));from neural import Model
assert not (B/'RUN_STARTED.json').exists(),'Preserve existing run; no automatic restart'
cfg=json.loads((P/'E1B/CONFIG.json').read_text());d=pd.read_csv(cfg['dataset'],float_precision='round_trip');sc=joblib.load(cfg['scalers']);xx=sc['input_scaler'].transform(d[cfg['features']]);sy=float(sc['target_scaler'].scale_[0]);torch.set_num_threads(4)
plan={'method':'Expected integrated gradients with Gauss Legendre quadrature over all 32 frozen training references','seeds':[42,123,2026,7,99],'backgrounds':[0,1],'origins':1000,'quadrature':[64,128,256],'adaptive_rule':'increase quadrature if abs(additivity residual)>1e-5+0.01*abs(output difference); retain and flag unresolved cases','aggregation':'sum absolute attribution over lags then mean over origins','not_exact_shap':True,'output':'unscaled predicted 24h log return','checkpoint_hashes':{str(s):hashlib.sha256((P/f'E1B/runs/E1B_seed{s}/best.pt').read_bytes()).hexdigest() for s in [42,123,2026,7,99]},'no_training_or_final_model_selection':True}
(B/'FULL_PROTOCOL.json').write_text(json.dumps(plan,indent=2));(B/'RUN_STARTED.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time()}));outdir=B/'results';outdir.mkdir(exist_ok=True);checks=[];importance=[]
try:
 for seed in plan['seeds']:
  ck=torch.load(P/f'E1B/runs/E1B_seed{seed}/best.pt',map_location='cpu',weights_only=False);m=Model(25,dict(hidden=128,bidirectional=True));m.load_state_dict(ck['model']);m.eval();assert ck['features']==cfg['features']
  for bi in plan['backgrounds']:
   old=np.load(B.parent/f'results/E1B_seed{seed}_background{bi}_ATTRIBUTIONS.npz');rows=old['origin_rows'];bids=old['background_rows'];X=torch.tensor(np.stack([xx[i-29:i+1] for i in rows]),dtype=torch.float32);bg=torch.tensor(np.stack([xx[i-29:i+1] for i in bids]),dtype=torch.float32)
   with torch.no_grad():yp=np.concatenate([m(X[i:i+128]).numpy().ravel() for i in range(0,len(X),128)]);delta=(yp-m(bg).numpy().mean())/sy
   aa=[];residual=[];used=[];start=time.time()
   for k,x in enumerate(X):
    for steps in plan['quadrature']:
     nodes,w=np.polynomial.legendre.leggauss(steps);nodes=torch.tensor((nodes+1)/2,dtype=torch.float32);w=torch.tensor(w/2,dtype=torch.float32);a=torch.zeros_like(x)
     for lo in range(0,len(bg),4):
      refs=bg[lo:lo+4];diff=x[None]-refs;mix=(refs[:,None]+nodes[None,:,None,None]*diff[:,None]).reshape(-1,30,25).requires_grad_(True);g=torch.autograd.grad(m(mix).sum(),mix)[0].reshape(len(refs),steps,30,25);a+=((g*w[None,:,None,None]).sum(1)*diff).sum(0)/len(bg)/sy
     residual_k=float(delta[k]-a.sum().item())
     if abs(residual_k)<=1e-5+.01*abs(delta[k]):break
    aa.append(a.detach().numpy());residual.append(residual_k);used.append(steps)
    if k%10==0:(B/'STATUS.json').write_text(json.dumps({'status':'running','seed':seed,'background':bi,'completed_origins':k+1,'total_origins':1000,'completed_seed_backgrounds':len(checks),'seconds_current':time.time()-start}))
   a=np.stack(aa);res=np.array(residual);np.savez_compressed(outdir/f'E1B_seed{seed}_background{bi}.npz',values=a,residual=res,quadrature_points=used,origin_rows=rows,background_rows=bids,features=cfg['features'],output_difference=delta)
   imp=abs(a).sum(1).mean(0);news=[f in cfg['news_features'] for f in cfg['features']];z={'seed':seed,'background':bi,'mean_absolute_residual':float(abs(res).mean()),'max_absolute_residual':float(abs(res).max()),'failed_tolerance':int((abs(res)>1e-5+.01*abs(delta)).sum()),'news_share_percent':float(100*imp[news].sum()/imp.sum()),'seconds':time.time()-start};checks.append(z)
   for f,v in zip(cfg['features'],imp):importance.append({'seed':seed,'background':bi,'feature':f,'mean_absolute_attribution':float(v)})
   pd.DataFrame(checks).to_csv(outdir/'PRECISION_AND_SHARES.csv',index=False);pd.DataFrame(importance).to_csv(outdir/'GLOBAL_IMPORTANCE.csv',index=False);print(json.dumps(z),flush=True)
 (B/'STATUS.json').write_text(json.dumps({'status':'computation_complete_requires_review_and_report','checks':checks},indent=2))
except Exception as e:
 (B/'FAILURE.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'completed_seed_backgrounds':len(checks)}));raise
