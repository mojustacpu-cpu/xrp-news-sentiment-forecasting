from pathlib import Path
import json,math,time,sys,hashlib
import numpy as np,pandas as pd
from scipy.stats import binom
B=Path(__file__).resolve().parents[2];O=B/'audit/results';R=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research');P=R/'news-enhanced-crypto-research/phase1';Q=R/'Research_Final_Experiments';OLD=B.parent/'publication_manuscript_20261108_story_enriched';SEED=2026
cfg=json.loads((P/'E1B/CONFIG.json').read_text());news=pd.read_csv(cfg['dataset'],float_precision='round_trip');reuse=json.loads((P/'E0_E1_REUSE.json').read_text())['E0'];d=pd.read_csv(reuse['data'],float_precision='round_trip');times=pd.DatetimeIndex(pd.to_datetime(d.available_at_utc,utc=True));tr=np.load(P/'main_train.npy');va=np.load(P/'main_validation.npy');te=np.load(P/'main_evaluation.npy');articles=pd.read_csv(R/'subprojects/original-hybird-extended-oct-2026/data/final_news_v3/NEWS_ARTICLES_FINAL_CALCULATED.csv');pub=pd.to_datetime(articles.published_at_utc,utc=True,format='mixed');at=pub.dt.ceil('h');assert at.is_monotonic_increasing
count=articles.assign(available=at).groupby('available').size().reindex(times,fill_value=0).to_numpy();c24=pd.Series(count).rolling(24,min_periods=1).sum().to_numpy();qt=at[articles.xrp_specificity.gt(0)].astype('int64').to_numpy();tt=times.asi8;recent=np.searchsorted(qt,tt,side='right')-np.searchsorted(qt,tt-6*3600*10**9,side='left')>0
th=json.loads((OLD/'analysis/results/TRAINING_DERIVED_THRESHOLDS.json').read_text());active=news.direct_news_present.to_numpy().astype(bool);sp=news.direct_specificity.to_numpy();u=news.direct_uncertainty.to_numpy();stance=news.direct_stance.to_numpy();vol=d.volatility.to_numpy();event=pd.read_csv(R/'subprojects/xrp-sentiment-enhanced-bilstm-controlled/reports/turning_points_top20/TWENTY_NEWS_EVENT_MEASUREMENTS.csv');ev=np.zeros(len(d),bool)
for t in pd.to_datetime(event.published_utc,utc=True,format='mixed'):ev|=(times>=t)&(times<t+pd.Timedelta(hours=24))
masks={'High volatility':vol>th['volatility_q90'],'Other volatility':vol<=th['volatility_q90'],'Current-hour news':active,'No current-hour news':~active,'XRP-specific news within 6h':recent,'No XRP-specific news within 6h':~recent,'Positive XRP stance':active&(stance>0),'Negative XRP stance':active&(stance<0),'High specificity with news':active&(sp>=th['specificity_median_active']),'Lower specificity with news':active&(sp<th['specificity_median_active']),'High uncertainty with news':active&(u>th['uncertainty_median_active']),'Lower uncertainty with news':active&(u<=th['uncertainty_median_active']),'High trailing24h article count':c24>th['frequency24_median'],'Other trailing24h article count':c24<=th['frequency24_median'],'Curated event window':ev,'Outside curated event window':~ev}
old=pd.read_csv(OLD/'analysis/results/CONDITIONAL_PERFORMANCE.csv');v=np.load(P/'analysis/VALIDATION_PREDICTIONS_ALL.npz');f=np.load(Q/'01_XRP_Final_Evaluation/results/PREDICTIONS_ARRAYS.npz');h=np.load(R/'news-enhanced-crypto-research/phase4_hbar/analysis/VALIDATION_PREDICTIONS_ALL.npz');eq=np.load(OLD/'analysis/results/E2_H6_LATER_PREDICTIONS.npz');jobs=[];layouts={}
# Each layout contains paired losses and masks; the same start indices are shared across comparisons.
for period,a,ii in [('validation',v,va),('later',f,te)]:
 e0=((a['E0']-a['actual'])**2).mean(0);e1=((a['E1B']-a['actual'])**2).mean(0);groups=[]
 for name,mask in masks.items():
  mm=mask[ii];r=old[(old.period==period)&(old.group==name)].iloc[0];assert mm.sum()==r.n;assert abs(np.sqrt(e1[mm].mean())-np.sqrt(e0[mm].mean())-r.difference)<1e-12
  groups.append((name,e0,e1,mm,32,'exploratory conditional'))
 if period=='validation':
  for _,r in pd.read_csv(P/'analysis/PAIRED_BLOCK_INTERVALS.csv').query('block_hours==72').iterrows():groups.append((r.candidate+' vs '+r.reference,((a[r.reference]-a['actual'])**2).mean(0),((a[r.candidate]-a['actual'])**2).mean(0),np.ones(len(e0),bool),10,'original validation family'))
 else:
  groups.extend([('E1B vs E0',e0,e1,np.ones(len(e0),bool),1,'later primary'),('E2_h6 vs E0',e0,((eq['predicted']-eq['actual'])**2).mean(0),np.ones(len(e0),bool),1,'supplementary post-exposure')])
 layouts[period]=groups
print('HBAR KEYS',h.files,flush=True)
hkeys=[k for k in h.files if k not in ['actual','origin']];print(hkeys,flush=True)
# use actual archived keys without substituting outcomes
ha=h['actual'];hk0=next(k for k in hkeys if 'technical' in k.lower());hk1=next(k for k in hkeys if 'e1b' in k.lower());layouts['HBAR']=[('HBAR_E1B vs technical',((h[hk0]-ha)**2).mean(0),((h[hk1]-ha)**2).mean(0),np.ones(len(ha),bool),1,'secondary transfer validation')]

def roll(x,L):
 n=len(x);xx=np.concatenate([x,x[:L]],axis=0);cum=np.concatenate([np.zeros((1,x.shape[1])),np.cumsum(xx,axis=0)],axis=0);return cum[L:L+n]-cum[:n]
def bootstrap(groups,block,reps,seed=SEED):
 n=len(groups[0][1]);m=math.ceil(n/block);last=n-(m-1)*block
 x=np.column_stack([q for name,a,b,mask,fam,scope in groups for q in [mask.astype(float),a*mask,b*mask]])
 full=roll(x,block);part=roll(x,last);rng=np.random.default_rng(seed);draws=np.empty((reps,len(groups)));counts=np.empty((reps,len(groups)),dtype=np.int32)
 for start in range(0,reps,500):
  k=min(500,reps-start);idx=rng.integers(0,n,size=(k,m));s=full[idx[:,:-1]].sum(1)+part[idx[:,-1]];s=s.reshape(k,len(groups),3);c=s[:,:,0];counts[start:start+k]=np.rint(c).astype(int)
  with np.errstate(divide='ignore',invalid='ignore'):draws[start:start+k]=np.sqrt(s[:,:,2]/c)-np.sqrt(s[:,:,1]/c)
 return draws,counts
# Test block-sum implementation against explicit resampled observations.
groups=layouts['validation'];bb,cc=bootstrap(groups,72,7);rng=np.random.default_rng(SEED);n=len(groups[0][1]);idx=(rng.integers(0,n,size=(7,math.ceil(n/72)))[:,:,None]+np.arange(72))%n;idx=idx.reshape(7,-1)[:,:n]
for j,(_,a,b,mm,_,_) in enumerate(groups):
 for k,ix in enumerate(idx):
  mask=mm[ix];assert cc[k,j]==mask.sum()
  if mask.any():assert abs(bb[k,j]-(np.sqrt(b[ix][mask].mean())-np.sqrt(a[ix][mask].mean())))<1e-11
(O/'BOOTSTRAP_ALGORITHM_VERIFICATION.json').write_text(json.dumps(dict(passed=True,explicit_replicates=7,comparisons=len(groups),tolerance=1e-11,rule='m−1 complete blocks plus exactly truncated final block; same paired indices and chronological masks'),indent=2))
t0=time.perf_counter();bootstrap(groups,72,1000);elapsed=time.perf_counter()-t0
bench=dict(seconds_per_1000_validation72=elapsed,rough_seconds_all_50000_blocks=elapsed*50*7,replicates=50000,estimated_peak_MB=150,paid_API=False,training=False);(O/'BENCHMARK.json').write_text(json.dumps(bench,indent=2));print('BENCHMARK',bench,flush=True)
if '--run' not in sys.argv:raise SystemExit(0)
if bench['rough_seconds_all_50000_blocks']>600:raise SystemExit('Estimate exceeds10minutes; request approval before proceeding')
reps=50000;rows=[];acf=[];mc=[];t0=time.perf_counter()
for period,groups in layouts.items():
 for block in [24,72,168]:
  draws,counts=bootstrap(groups,block,reps);np.savez_compressed(O/f'BOOTSTRAP_DRAWS_{period}_block{block}.npz',differences=draws,subgroup_counts=counts,names=np.array([g[0] for g in groups]),seed=SEED)
  for j,(name,a,b,mask,family,scope) in enumerate(groups):
   usable=draws[:,j][np.isfinite(draws[:,j])];good=len(usable);enough=good>=.95*reps;alpha=.05/family;riska=np.sqrt(a[mask].mean());riskb=np.sqrt(b[mask].mean());r=dict(period=period,group=name,scope=scope,n=int(mask.sum()),block_hours=block,reps=reps,seed=SEED,usable_resamples=good,zero_subgroup_resamples=int((counts[:,j]==0).sum()),family=family,adjusted_confidence=1-alpha,E0_rmse=riska,E1B_rmse=riskb,difference=riskb-riska,change_percent=100*(riskb/riska-1),interval_estimable=enough)
   if enough:
    lo,hi,al,ah=np.quantile(usable,[.025,.975,alpha/2,1-alpha/2]);r.update(ci95_low=lo,ci95_high=hi,adjusted_low=al,adjusted_high=ah,adjusted_improvement=bool(ah<0),adjusted_deterioration=bool(al>0))
    so=np.sort(usable)
    for endpoint,p in [('adjusted_low',alpha/2),('adjusted_high',1-alpha/2)]:
     ranks=np.clip(binom.ppf([.025,.975],good,p).astype(int),0,good-1);r[endpoint+'_mc95_low']=so[ranks[0]];r[endpoint+'_mc95_high']=so[ranks[1]]
    mc.append(dict(period=period,group=name,block_hours=block,family=family,tail_expected_draws=good*alpha/2,endpoint_mc95_low=r['adjusted_low_mc95_low'],endpoint_mc95_high=r['adjusted_low_mc95_high'],interpretation='Binomial order-statistic bracket approximates MonteCarlo uncertainty of the conditional bootstrap quantile, not a second sampling CI'))
   rows.append(r)
  print('COMPLETE',period,block,'elapsed',time.perf_counter()-t0,flush=True)
 # ACF is descriptive and does not estimate an optimal block after seeing results.
 a,b=groups[0][1],groups[0][2];loss=b-a
 for lag in [1,6,12,24,48,72,168]:
  acf.append(dict(period=period,lag_hours=lag,paired_squared_loss_acf=float(np.corrcoef(loss[:-lag],loss[lag:])[0,1])))
pd.DataFrame(rows).to_csv(O/'BOOTSTRAP_ALL_CORRECTED.csv',index=False);pd.DataFrame(mc).to_csv(O/'BOOTSTRAP_MONTE_CARLO.csv',index=False);pd.DataFrame(acf).to_csv(O/'PAIRED_LOSS_AUTOCORRELATION.csv',index=False)
primary=pd.DataFrame(rows);primary[(primary.scope=='exploratory conditional')&(primary.block_hours==72)].to_csv(O/'CONDITIONAL_PERFORMANCE_50000.csv',index=False)
(O/'STATISTICAL_AUDIT_COMPLETE.json').write_text(json.dumps(dict(status='complete',resamples=reps,blocks=[24,72,168],seed=SEED,seconds=time.perf_counter()-t0,comparisons_per_block={k:len(x) for k,x in layouts.items()},conditional_family=32,original_validation_family=10,conditional_definition_unchanged=True,no_model_training=True),indent=2));print('ALL STATISTICS COMPLETE',flush=True)
