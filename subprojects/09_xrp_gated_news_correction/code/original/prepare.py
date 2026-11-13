from pathlib import Path
import json,hashlib,shutil,sys
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import RobustScaler
P=Path(__file__).resolve().parents[1];ROOT=P.parent;OLD=ROOT/'phase2_e3';PH1=ROOT/'phase1'
sys.path.insert(0,str(P/'code'));sys.path.insert(0,str(PH1/'reference'))
from neural import Model
from gated_model import NewsCorrection
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')
assert not (P/'PREPARATION_COMPLETE.json').exists(),'Existing preparation preserved'
for stage in [OLD,PH1]:
 assert all(sha(f)==v for f,v in json.loads((stage/'FROZEN_MANIFEST.json').read_text())['files'].items())
(P/'data').mkdir(exist_ok=True);(P/'cache').mkdir(exist_ok=True)
reuse=json.loads((PH1/'E0_E1_REUSE.json').read_text())['E0'];cfg=json.loads((OLD/'E3/CONFIG.json').read_text());d=pd.read_csv(cfg['dataset'],float_precision='round_trip');news=cfg['news_features'];tech=reuse['feature_order'];sc=joblib.load(cfg['scalers']);e0sc=joblib.load(reuse['scaler_path']);assert sha(reuse['scaler_path'])==reuse['scaler_sha256']
shutil.copy2(cfg['dataset'],P/'data/TRAINING_UNSCALED.csv');shutil.copy2(OLD/'VALIDATION_ORIGINS.csv',P/'data/VALIDATION_ORIGINS.csv')
ids={k:np.load(OLD/f'main_{k}.npy') for k in ['train','validation','evaluation']}
for k,x in ids.items():np.save(P/f'data/main_{k}.npy',x)
fit=np.unique((ids['train'][:,None]-np.arange(30)).ravel());sx=RobustScaler().fit(d.iloc[fit][news]);assert np.array_equal(sx.center_,sc['input_scaler'].center_[19:]) and np.array_equal(sx.scale_,sc['input_scaler'].scale_[19:]);assert np.array_equal(e0sc['target_scaler'].scale_,sc['target_scaler'].scale_)
joblib.dump(dict(news_scaler=sx,target_scaler=sc['target_scaler'],news_features=news,technical_features=tech,fit_input_rows=fit,fit_target_origins=ids['train']),P/'data/scalers.joblib')
assert len(ids['validation'])==2136 and d.iloc[ids['evaluation']].target_close_plus24h.isna().all()
av=pd.to_datetime(d.available_at_utc,utc=True);ta=pd.to_datetime(d.target_available_at_utc,utc=True);assert (ta-av).eq(pd.Timedelta(hours=24)).all();assert av.diff().dropna().eq(pd.Timedelta(hours=1)).all();assert ta.iloc[ids['train'][-1]]<av.iloc[ids['validation'][0]] and ta.iloc[ids['validation'][-1]]<av.iloc[ids['evaluation'][0]]
base=pd.read_csv(reuse['data'],float_precision='round_trip');assert np.array_equal(base[tech].to_numpy(),d[tech].to_numpy())
mask=1-d.no_relevant_news.to_numpy();assert np.isin(mask,[0,1]).all();np.save(P/'data/eligible_news_mask.npy',mask.astype(np.float32))
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);tx=e0sc['input_scaler'].transform(d[tech]);checks=[]
for item in reuse['seeds']:
 assert sha(item['checkpoint'])==item['checkpoint_sha256'] and sha(item['predictions'])==item['predictions_sha256']
 ck=torch.load(item['checkpoint'],weights_only=False,map_location='cpu');m=Model(len(tech),dict(hidden=128,bidirectional=True));m.load_state_dict(ck['model']);m.eval()
 for pp in m.parameters():pp.requires_grad_(False)
 cache={}
 for split in ['train','validation']:
  idx=ids[split];parts=[]
  with torch.no_grad():
   for j in range(0,len(idx),256):parts.append(m(torch.tensor(np.stack([tx[i-29:i+1] for i in idx[j:j+256]]),dtype=torch.float32)).numpy().ravel())
  cache[split+'_scaled']=np.concatenate(parts)
 pred=d.close.iloc[ids['validation']].to_numpy()*np.exp(e0sc['target_scaler'].inverse_transform(cache['validation_scaled'][:,None]).ravel());orig=pd.read_csv(item['predictions'],float_precision='round_trip');assert np.array_equal(orig.actual,d.target_close_plus24h.iloc[ids['validation']].to_numpy());assert np.max(abs(pred-orig.predicted))<1e-7
 assert pd.to_datetime(orig.origin_time,utc=True).equals(av.iloc[ids['validation']].reset_index(drop=True))
 np.savez_compressed(P/f"cache/E0_seed{item['seed']}.npz",**cache)
 checks.append(dict(seed=item['seed'],checkpoint=item['checkpoint'],checkpoint_sha256=item['checkpoint_sha256'],prediction_max_error=float(np.max(abs(pred-orig.predicted))),frozen_parameters=True,cache_sha256=sha(P/f"cache/E0_seed{item['seed']}.npz")))
 print('Cached frozen E0',item['seed'],flush=True)
scale=float(sc['target_scaler'].scale_[0]);xx=torch.tensor(sx.transform(d[news])[:60].reshape(2,30,8),dtype=torch.float32);bb=torch.randn(2,1);ms=torch.tensor([[1.],[0.]])
models=[]
for gated in [True,False]:
 torch.manual_seed(42);m=NewsCorrection(8,gated);m.eval();pred,g,delta,cor=m(xx,bb,ms,scale);assert torch.equal(pred,bb);m.correction.bias.data.fill_(.01);pred,g,delta,cor=m(xx,bb,ms,scale);assert torch.equal(pred[1],bb[1]);assert torch.equal(m(xx,bb,ms,scale,disabled=True)[0],bb);assert ((g>=0)&(g<=1)).all();pred.square().mean().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters());models.append(dict(name='E4_gated' if gated else 'E4_ungated',gated=gated,trainable_parameters=sum(p.numel() for p in m.parameters()),frozen_parameters=564353))
protocol=dict(stage='E4 only',authorization='User resumed after E3 completion and E4 next-stage announcement. Ten fits, no subsequent stages.',models=models,seeds=[42,123,2026,7,99],news_features=news,technical_features=tech,window_hours=30,horizon_hours=24,target='log(P[t+24]/P[t])',half_life_hours=6,news_mask='1-no_relevant_news at forecast origin; positive specificity mass in causal trailing48h window',architecture='Frozen seed-matched E0 plus one-layer unidirectional GRU16 news encoder, dropout0.2, linear scalar correction in unscaled logreturn units, optional sigmoid linear gate. Zero correction-head weights and bias.',fusion='scaled_prediction=E0_scaled+target_scale*news_mask*gate*delta_raw_return',training=dict(optimizer='Adam',lr=.001,weight_decay=.0001,correction_penalty=.001,loss='scaled-return MSE + 0.001 mean squared scaled correction',batch_size=32,max_epochs=200,patience=15,threads=4,device='cpu'),checkpoint_selection='Minimum validation scaled-return MSE excluding penalty, epochs1..200; epoch0 is diagnostic only',configuration_selection='Five-seed mean validation reconstructed-price RMSE; compare with previous models without replacing an inferior outcome',statistics='Paired circular moving-block bootstrap on seed-averaged squared errors:2000replicates seed2026,72h primary and24/168h sensitivity; compare each E4 variant vsE0,E1B,E3 and gated vsungated:7 contrasts;95% and Bonferroni99.285714% intervals. Disclose validation selection optimism accumulated across earlier stages.',baseline_training_limitation='E0 training forecasts are in-sample frozen predictions, not cross-fitted; E0 checkpoint was selected on this same validation block. This is a staged correction model, not an independent validation replication.',no_news_behavior='Exactly E0 when mask0 or correction disabled; a gate value is model behaviour, not causal impact.',final_evaluation_performed=False,stop_after_E4=True)
save(P/'PROTOCOL.json',protocol);save(P/'E0_REUSE.json',reuse);save(P/'PREFLIGHT_VERIFICATION.json',dict(passed=True,checks=checks,train_only_news_scaling=True,zero_initial_correction_exact=True,disabled_correction_exact=True,inactive_news_exact=True,finite_forward_backward=True,validation_origins=2136,technical_values_unchanged=True,final_targets_redacted=True,models=models));save(P/'PREPARATION_COMPLETE.json',dict(status='prepared',utc=pd.Timestamp.now(tz='UTC')))
print(json.dumps(models))
