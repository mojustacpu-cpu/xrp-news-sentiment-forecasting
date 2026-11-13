from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd,torch,joblib
B=Path(__file__).resolve().parents[1];ROOT=B.parents[2];P=ROOT/'news-enhanced-crypto-research/phase1';sys.path.insert(0,str(P/'reference'));from neural import Model
cfg=json.loads((P/'E1B/CONFIG.json').read_text());d=pd.read_csv(cfg['dataset'],float_precision='round_trip');sc=joblib.load(cfg['scalers']);xx=sc['input_scaler'].transform(d[cfg['features']]);sy=float(sc['target_scaler'].scale_[0]);old=np.load(B.parent/'results/E1B_seed42_background0_ATTRIBUTIONS.npz');orig=old['origin_rows'];bids=old['background_rows'];chosen=np.linspace(0,len(orig)-1,20,dtype=int);rows=orig[chosen];X=torch.tensor(np.stack([xx[i-29:i+1] for i in rows]),dtype=torch.float32);bg=torch.tensor(np.stack([xx[i-29:i+1] for i in bids]),dtype=torch.float32)
cp=P/'E1B/runs/E1B_seed42/best.pt';ck=torch.load(cp,map_location='cpu',weights_only=False);m=Model(25,dict(hidden=128,bidirectional=True));m.load_state_dict(ck['model']);m.eval();torch.set_num_threads(4)
plan={'method':'Deterministic expected integrated gradients averaged over all 32 original training references; Gauss Legendre quadrature','pilot_origins':rows.tolist(),'seed':42,'quadrature_points':[16,32,64],'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'preserves_original_outputs':True,'not_exact_shap':True,'purpose':'numerical convergence, not forecasting model selection'}
(B/'PROTOCOL.json').write_text(json.dumps(plan,indent=2));checks=[]
with torch.no_grad():delta=(m(X).numpy().ravel()-m(bg).numpy().mean())/sy
for steps in [16,32,64]:
 start=time.time();nodes,weights=np.polynomial.legendre.leggauss(steps);nodes=(nodes+1)/2;weights=weights/2;aa=[]
 for k,x in enumerate(X):
  out=torch.zeros_like(x)
  for lo in range(0,len(bg),4):
   refs=bg[lo:lo+4];diff=x[None]-refs;mix=(refs[:,None]+torch.tensor(nodes,dtype=torch.float32)[None,:,None,None]*diff[:,None]).reshape(-1,30,25).requires_grad_(True)
   grads=torch.autograd.grad(m(mix).sum(),mix)[0].reshape(len(refs),steps,30,25);integ=(grads*torch.tensor(weights,dtype=torch.float32)[None,:,None,None]).sum(1);out+=(integ*diff).sum(0)/len(bg)/sy
  aa.append(out.detach().numpy());(B/'STATUS.json').write_text(json.dumps({'status':'pilot_running','quadrature_points':steps,'completed':k+1,'total':20}))
 a=np.stack(aa);res=delta-a.sum((1,2));np.savez_compressed(B/f'pilot_q{steps}.npz',values=a,residual=res,origin_rows=rows,background_rows=bids,features=cfg['features']);v={'quadrature_points':steps,'mean_absolute_residual':float(abs(res).mean()),'max_absolute_residual':float(abs(res).max()),'mean_absolute_output_delta':float(abs(delta).mean()),'seconds':time.time()-start};checks.append(v);print(json.dumps(v),flush=True);(B/'PILOT_RESULTS.json').write_text(json.dumps(checks,indent=2))
(B/'STATUS.json').write_text(json.dumps({'status':'pilot_complete_requires_convergence_review','results':checks},indent=2))
