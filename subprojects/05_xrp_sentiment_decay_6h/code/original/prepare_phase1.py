from pathlib import Path
import json,shutil,hashlib,sys
import pandas as pd,numpy as np,joblib
from sklearn.preprocessing import RobustScaler
P=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research/news-enhanced-crypto-research'); S=P.parent/'subprojects'; E=S/'xrp-technical-btc-log-return-control'; D=E/'data/technical_only_v1'; O=P/'phase1';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str))
assert not (O/'FROZEN_MANIFEST.json').exists()
(O/'reference').mkdir(exist_ok=True)
for f in ['neural.py','common.py']:shutil.copy2(E/'reference'/f,O/'reference'/f)
sc=joblib.load(E/'runs/log_return_v1/log_return_scalers.joblib'); tech=sc['feature_order']; m=json.loads((D/'TRAINING_MANIFEST.json').read_text());d=pd.read_csv(m['dataset'],float_precision='round_trip');ids={k:np.load(D/f'main_{k}.npy') for k in ['train','validation','evaluation']}
assert len(ids['validation'])==2136
for k,v in ids.items():np.save(O/f'main_{k}.npy',v)
fit=np.unique((ids['train'][:,None]-np.arange(30)).ravel()); assert np.array_equal(fit,sc['fit_input_rows'])
configs=[]
for name,h in [('E1B',3),('E2_h3',3),('E2_h6',6),('E2_h12',12),('E2_h24',24)]:
 f=P/f'SP03_scientific_equation/configs/prepared_features_v1/news_delay0_hl{h}.csv';n=pd.read_csv(f,float_precision='round_trip');assert pd.to_datetime(n.origin_time,utc=True).equals(pd.to_datetime(d.available_at_utc,utc=True))
 news=['direct_sentiment','direct_specificity','direct_stance','direct_uncertainty','direct_news_present','hours_since_latest_news_capped48'] if name=='E1B' else ['equation_sentiment','no_relevant_news']
 z=d.copy()
 for c in news:z[c]=n[c]
 # Retain full hourly inputs but redact final evaluation labels in training artifacts.
 used=np.concatenate([ids['train'],ids['validation']]); z.loc[~z.index.isin(used),'target_close_plus24h']=np.nan
 out=O/name;out.mkdir();z.to_csv(out/'training.csv',index=False)
 features=tech+news;sx=RobustScaler().fit(z.iloc[fit][features]);assert np.array_equal(sx.center_[:len(tech)],sc['input_scaler'].center_) and np.array_equal(sx.scale_[:len(tech)],sc['input_scaler'].scale_)
 bundle=dict(input_scaler=sx,target_scaler=sc['target_scaler'],feature_order=features,fit_input_rows=fit,fit_target_origins=ids['train']);joblib.dump(bundle,out/'scalers.joblib')
 sys.path.insert(0,str(O/'reference'));from neural import Model
 cfg=dict(name=name,half_life_hours=None if name=='E1B' else h,delay_hours=0,news_features=news,features=features,parameters=sum(p.numel() for p in Model(len(features),dict(hidden=128,bidirectional=True)).parameters()),dataset=str(out/'training.csv'),scalers=str(out/'scalers.joblib'),source_scenario_sha256=sha(f))
 save(out/'CONFIG.json',cfg);configs.append(cfg)
z=d.iloc[ids['validation']][['available_at_utc','target_available_at_utc','close','target_close_plus24h']];z.to_csv(O/'VALIDATION_ORIGINS.csv',index=False)
protocol=json.loads((P/'configs/PROPOSED_PROTOCOL.json').read_text());protocol.update(status='approved_phase1_only',approved_new_fits=25,candidates=configs,stop_after_phase1=True,E0_E1_retrained=False,final_evaluation_performed=False,phase1_statistics='Paired circular moving-block bootstrap on seed-averaged squared errors:72h primary,24/168h sensitivity,2000 replicates,seed2026. Five candidates versus E0 and E1:10 comparisons; report unadjusted95% and Bonferroni99.5% intervals, validation-selection optimism disclosed. Numerical improvement alone is not robust evidence.')
save(O/'PROTOCOL.json',protocol)
shutil.copy2(P/'SP01_technical_baselines/manifests/BASELINE_AND_DIRECT_REUSE_VERIFICATION.json',O/'E0_E1_REUSE.json')
files={str(p):sha(p) for p in O.rglob('*') if p.is_file()}
for p in (P/'SP03_scientific_equation/configs/prepared_features_v1').glob('*.csv'):files[str(p)]=sha(p)
files[m['dataset']]=sha(m['dataset']);files[str(E/'runs/log_return_v1/log_return_scalers.joblib')]=sha(E/'runs/log_return_v1/log_return_scalers.joblib')
save(O/'FROZEN_MANIFEST.json',dict(created_utc=str(pd.Timestamp.now(tz='UTC')),files=files,validation_count=2136,technical_inputs_exact=True,target_scaler_reused=True,all16scenarios_preserved=True,final_labels_redacted=True))
print('Frozen five configurations; validation origins 2136; training-only scalers verified.')
