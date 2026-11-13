from common import *
import shutil
S=B.parent/'xrp-2026-data-original-model'
assert not (B/'configs/protocol.json').exists(),'Protocol exists; preserve frozen design'
for src,dest in [(S/'data/review_candidate/XRP_predictors_through_25Oct_Sydney.csv',B/'data/predictors.csv'),(S/'reference/train_9model_bilstm.py',B/'reference/original_train.py'),(S/'runs/fresh_20261031/reports/test_predictions.csv',B/'reference/original_evaluation.csv')]:shutil.copy2(src,dest)
d=data();times=pd.DatetimeIndex(d.time);assert times.is_unique and np.all(np.diff(times.asi8)==3600*10**9)
ref=pd.read_csv(B/'reference/original_evaluation.csv');main=times.get_indexer(pd.to_datetime(ref.last_input_time,utc=True));assert len(main)==5262 and (main>=0).all()
folds={};first=int(main[0])
for name,evaluation in [('rolling_1',np.arange(first-4320,first-2160-H)),('rolling_2',np.arange(first-2160,first-H)),('main',main)]:
 start=int(evaluation[0]);valstart=start-2160
 train=np.arange(L-1,valstart-H);val=np.arange(valstart,start-H)
 assert train[-1]+H<val[0] and val[-1]+H<evaluation[0]
 assert evaluation[-1]+H<len(d) and train[0]>=L-1
 for part,ids in [('train',train),('validation',val),('evaluation',evaluation)]:
  np.save(B/'configs'/f'{name}_{part}.npy',ids)
  pd.DataFrame(dict(origin_time=times[ids],target_time=times[ids+H])).to_csv(B/'data'/f'{name}_{part}_timestamps.csv',index=False)
 folds[name]={part:dict(n=len(ids),first_origin=str(times[ids[0]]),last_origin=str(times[ids[-1]]),first_target=str(times[ids[0]+H]),last_target=str(times[ids[-1]+H])) for part,ids in [('train',train),('validation',val),('evaluation',evaluation)]}
# Same architecture as archive; size-matched unidirectional hidden width chosen without data scores.
def count(n,h,bi):
 directions=2 if bi else 1
 return directions*(4*h*n+4*h*h+8*h)+directions*(4*h*(h*directions)+4*h*h+8*h)+(h*directions)*64+64+65
matched=min(range(128,300),key=lambda h:abs(count(19,h,False)-count(19,128,True)))
configs=[dict(name='bilstm_btc',btc=True,bidirectional=True,hidden=128),dict(name='bilstm_no_btc',btc=False,bidirectional=True,hidden=128),dict(name='lstm_btc',btc=True,bidirectional=False,hidden=128),dict(name='lstm_matched_btc',btc=True,bidirectional=False,hidden=matched)]
for c in configs:c['parameters']=count(19 if c['btc'] else 18,c['hidden'],c['bidirectional'])
p=dict(status='frozen_before_training',horizon_hours=H,input_window_hours=L,main_seeds=[42,123,2026,7,99],rolling_seeds=[42],rolling_models=['bilstm_btc','bilstm_no_btc','lstm_btc'],neural=configs,max_epochs=200,patience=15,min_delta=0.0,batch_size=32,adam_lr=.001,dropout=.2,scaling='RobustScaler inputs fit on unique training-window rows; MinMaxScaler targets fit only training targets',selection='lowest chronological validation MSE, never evaluation; fixed configuration means all five seeds are retained',folds=folds,main_evaluation='Previously examined historical period; not a pristine prospective holdout',target_alignment='Window includes origin; target close at origin+24',source_sha256=sha(B/'data/predictors.csv'),source=str(S),parameter_match='Nearest integer hidden width to original BiLSTM count; not exactly equal',statistics='Seed standard deviations describe training variability, not independent market replications',report_scope='Price models only plus separate GARCH; no old 25h metrics directly mixed with new 24h metrics')
save(B/'configs/protocol.json',p);save(B/'reports/DATA_AUDIT.json',dict(raw_first=str(times[0]),raw_last=str(times[-1]),finite_features=bool(np.isfinite(d[F]).all().all()),no_duplicate_hours=True,hourly_continuity=True,all_targets_exactly_24h=True,boundary_labels_purged=True,source_byte_identical=(B/'data/predictors.csv').read_bytes()==(S/'data/review_candidate/XRP_predictors_through_25Oct_Sydney.csv').read_bytes(),last_available_hour_not_used_in_main='Main origins preserved from archive; final target one hour earlier than archived 25h target'))
d.head(48).to_csv(B/'data/prepared_preview.csv',index=False)
print(json.dumps(p,indent=2))
