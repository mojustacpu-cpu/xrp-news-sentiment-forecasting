from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
B=Path(__file__).resolve().parents[1];p=B/'results/XRP_FINAL_PREDICTIONS.csv';d=pd.read_csv(p);rows=[];reference=None
for (m,s),q in d.groupby(['model','seed']):
 q=q.sort_values('origin_time');identity=q[['origin_time','target_time','origin_close','actual']].reset_index(drop=True)
 if reference is None:reference=identity
 else:pd.testing.assert_frame_equal(identity,reference)
 assert len(q)==5262 and not q.origin_time.duplicated().any()
 assert ((pd.to_datetime(q.target_time,utc=True)-pd.to_datetime(q.origin_time,utc=True)).dt.total_seconds()==86400).all()
 e=q.predicted-q.actual;rows.append({'model':m,'seed':int(s),'rmse':float(np.sqrt(np.mean(e**2))),'mae':float(abs(e).mean()),'r2':float(1-np.sum(e**2)/np.sum((q.actual-q.actual.mean())**2)),'directional_accuracy':float(np.mean(np.sign(q.predicted-q.origin_close)==np.sign(q.actual-q.origin_close)))})
a=pd.DataFrame(rows);saved=pd.read_csv(B/'results/XRP_FINAL_SEED_RESULTS.csv');merged=a.merge(saved,on=['model','seed'],suffixes=('_fresh','_saved'))
errors={k:float(abs(merged[k+'_fresh']-merged[k+'_saved']).max()) for k in ['rmse','mae','r2','directional_accuracy']};assert max(errors.values())<1e-10
out={'passed':True,'prediction_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'identical_origins':5262,'seed_model_combinations':10,'exact_horizon_hours':24,'metric_max_abs_errors':errors,'previous_exposure':'Retrospective period previously examined elsewhere. Not a genuinely unseen prospective test. No fresh test period invented.','transform':'r=log(P[t+24]/P[t]); scaled prediction z is inverted with (z-target_scaler.min_[0])/target_scaler.scale_[0]; reconstructed price=P[t]*exp(rhat).'}
(B/'SAVED_METRICS_REVERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
