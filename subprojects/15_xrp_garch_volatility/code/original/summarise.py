from common import *
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=B/'reports';rows=[];vol=[];verification=[];selections=[]
for fold in ['main','rolling_1','rolling_2']:
 ids=arrays(fold);d=data();reference=None;validation_candidates=[]
 lock=json.loads((B/'runs'/fold/'SELECTION_LOCK.json').read_text())
 for path,expected in lock['checkpoints_and_selections'].items():assert sha(B/path)==expected
 assert ids['train'][-1]+H<ids['validation'][0] and ids['validation'][-1]+H<ids['evaluation'][0]
 for R in sorted((B/'runs'/fold).iterdir()):
  if not R.is_dir():continue
  result=R/'EVALUATION.json'
  assert result.exists(),f'Evaluation missing: {R}'
  e=json.loads(result.read_text());f=pd.read_csv(R/'evaluation_predictions.csv')
  assert np.array_equal(pd.to_datetime(f.origin_time,utc=True).to_numpy(),d.time.iloc[ids['evaluation']].to_numpy())
  assert np.array_equal(pd.to_datetime(f.target_time,utc=True).to_numpy(),d.time.iloc[ids['evaluation']+H].to_numpy())
  if e['model']=='garch':vol.append(e);continue
  actual=d.close.iloc[ids['evaluation']+H].to_numpy();origin=d.close.iloc[ids['evaluation']].to_numpy();assert np.allclose(f.actual_close,actual)
  check=metrics(actual,f.predicted_close.to_numpy(),origin)
  for key,value in check.items():assert np.isclose(e[key],value)
  complete=json.loads((R/'COMPLETE.json').read_text())
  validation_candidates.append(dict(model=e['model'],seed=e.get('seed'),validation_rmse=complete['selected']['rmse'] if e['model']=='arimax' else complete['validation_metrics']['rmse']))
  e.update({k:complete.get(k) for k in ['selected_epoch','epochs_run','parameter_count','training_seconds']});rows.append(e)
  if (R/'scalers.pkl').exists():
   sc=pickle.load(open(R/'scalers.pkl','rb'));assert sc['x_last_fit_row']==int(ids['train'][-1]);assert np.array_equal(sc['y_target_rows'],ids['train']+H)
   raw=d[sc['features']].iloc[:ids['train'][-1]+1].to_numpy();assert np.allclose(sc['x'].center_,np.median(raw,axis=0))
   assert np.isclose(sc['y'].data_min_[0],d.close.iloc[ids['train']+H].min()) and np.isclose(sc['y'].data_max_[0],d.close.iloc[ids['train']+H].max())
   hist=pd.read_csv(R/'history.csv');assert int(hist.loc[hist.validation_mse.idxmin(),'epoch'])==complete['selected_epoch']
   fig,ax=plt.subplots(figsize=(9,3));ax.plot(hist.epoch,hist.training_mse,label='Training');ax.plot(hist.epoch,hist.validation_mse,label='Validation');ax.axvline(complete['selected_epoch'],ls=':',color='gray');ax.set(xlabel='Epoch',ylabel='Scaled MSE',title=f'{fold} {e["model"]} seed {e["seed"]}');ax.legend();fig.tight_layout();fig.savefig(R/'loss.png',dpi=140);plt.close(fig)
 grouped=pd.DataFrame(validation_candidates).groupby('model',sort=False).validation_rmse.mean()
 winner=grouped.idxmin();eligible=[q for q in validation_candidates if q['model']==winner];choice=min(eligible,key=lambda q:q['validation_rmse'])
 selections.append(dict(fold=fold,selected_configuration=winner,mean_validation_rmse=float(grouped[winner]),representative_seed=choice['seed'],all_validation_scores=validation_candidates,criterion='validation only; APPROVED_REQUIREMENTS.json recorded before evaluations'))
 ev=ids['evaluation'];truth=d.close.iloc[ev+H].to_numpy();origin=d.close.iloc[ev].to_numpy();bm=metrics(truth,origin,origin);rows.append(dict(model='persistence',fold=fold,seed=None,**bm))
 for row in rows:
  if row['fold']==fold:row.update(rmse_relative_to_persistence=row['rmse']/bm['rmse'],rmse_improvement_pct=100*(1-row['rmse']/bm['rmse']))
 write_predictions(O/f'{fold}_persistence_predictions.csv',d,ev,origin)
 verification.append(dict(fold=fold,identical_targets=True,metrics_recomputed=True,training_scalers_checked=True,selected_epochs_match_validation_minimum=True))
price=pd.DataFrame(rows);price.to_csv(O/'PRICE_RESULTS_ALL_RUNS.csv',index=False);pd.DataFrame(vol).to_csv(O/'GARCH_RESULTS.csv',index=False)
save(O/'VALIDATION_SELECTION.json',selections)
summary=price.groupby(['fold','model']).agg(runs=('rmse','size'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),r2_mean=('r2','mean'),r2_sd=('r2','std'),direction_mean=('directional_accuracy','mean'),rmse_relative_to_persistence=('rmse_relative_to_persistence','mean'),rmse_improvement_pct=('rmse_improvement_pct','mean')).reset_index();summary.to_csv(O/'PRICE_SUMMARY.csv',index=False)
main=summary[summary.fold=='main'];fig,ax=plt.subplots(figsize=(10,4));ax.bar(main.model,main.rmse_mean,yerr=main.rmse_sd.fillna(0),capsize=4);ax.tick_params(axis='x',rotation=20);ax.set(ylabel='RMSE USDT',title='24 hour price error: mean and seed standard deviation');fig.tight_layout();fig.savefig(O/'main_comparison.png',dpi=170);plt.close(fig)
# Curves show predefined seed42, not best evaluation seed; ARIMAX has no seed.
fig,axs=plt.subplots(5,1,figsize=(10,10),sharex=True,sharey=True)
for ax,name in zip(axs,['bilstm_btc','bilstm_no_btc','lstm_btc','lstm_matched_btc','arimax']):
 r=B/'runs/main'/(name if name=='arimax' else name+'_seed42');f=pd.read_csv(r/'evaluation_predictions.csv');t=pd.to_datetime(f.target_time)
 ax.plot(t,f.actual_close,color='black',lw=.6,label='Actual');ax.plot(t,f.predicted_close,lw=.65,label=name);ax.legend(fontsize=8);ax.set_ylabel('USDT')
fig.autofmt_xdate();fig.tight_layout();fig.savefig(O/'main_curves_seed42.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4))
for name in ['bilstm_btc','bilstm_no_btc','lstm_btc','arimax']:
 z=price[(price.model==name)&((price.seed==42)|price.seed.isna())].set_index('fold').reindex(['rolling_1','rolling_2','main']);ax.plot(z.index,z.rmse,marker='o',label=name)
ax.set(ylabel='RMSE USDT',title='Earlier rolling periods and main period: predefined seed42');ax.legend();fig.tight_layout();fig.savefig(O/'rolling_comparison.png',dpi=170);plt.close(fig)
g=pd.read_csv(B/'runs/main/garch/evaluation_predictions.csv');fig,ax=plt.subplots(figsize=(10,3.5));ax.plot(pd.to_datetime(g.target_time),np.sqrt(g.actual_variance),label='Realised 24h',lw=.7);ax.plot(pd.to_datetime(g.target_time),np.sqrt(g.predicted_variance),label='GARCH',lw=.7);ax.legend();ax.set(ylabel='Volatility percent',title='Separate GARCH volatility forecast');fig.autofmt_xdate();fig.tight_layout();fig.savefig(O/'garch_curve.png',dpi=170);plt.close(fig)
save(O/'FINAL_NUMERICAL_VERIFICATION.json',dict(status='passed',folds=verification,main_neural_runs=20,rolling_neural_runs=6,persistence_all_origins_targets_verified=True,validation_only_configuration_selection=True,main_is_retrospective=True,source_sha256=sha(B/'data/predictors.csv')))
print(summary.to_string(index=False))
