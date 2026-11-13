from common import *
import pickle,time,warnings
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

def inputs(fold):
 d=data();ids=arrays(fold);x=d[EXOG].shift(H).to_numpy();y=np.log(d.close.to_numpy());start=H;end=int(ids['train'][-1]+H)
 sc=StandardScaler().fit(x[start:end+1]);return d,ids,sc,sc.transform(x),y,start,end

def rolling(res,y,x,origins,end):
 p=[]
 for o in origins:
  if o>end:res=res.extend(y[end+1:o+1],exog=x[end+1:o+1]);end=int(o)
  p.append(float(np.exp(res.forecast(H,exog=x[o+1:o+H+1])[-1])))
 return np.asarray(p)
def train(fold):
 R=B/'runs'/fold/'arimax';R.mkdir(parents=True,exist_ok=True)
 if (R/'COMPLETE.json').exists():return
 assert not (R/'STARTED.json').exists(),'ARIMAX partial run preserved'
 save(R/'STARTED.json',dict(pid=os.getpid()));d,ids,sc,x,y,start,end=inputs(fold);rows=[];t0=time.time()
 pickle.dump(sc,open(R/'scaler.pkl','wb'))
 for order in [(0,1,0),(1,1,0),(0,1,1)]:
  with warnings.catch_warnings(record=True) as w:
   fit=SARIMAX(y[start:end+1],exog=x[start:end+1],order=order,trend='n',enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=300,cov_type='none')
  key='_'.join(map(str,order));fit.save(R/f'{key}.pkl');ok=bool(fit.mle_retvals.get('converged',False));row=dict(order=order,key=key,converged=ok,warnings=[str(z.message) for z in w])
  if ok:
   vp=rolling(fit,y,x,ids['validation'],end);row.update(metrics(d.close.iloc[ids['validation']+H].to_numpy(),vp,d.close.iloc[ids['validation']].to_numpy()));write_predictions(R/f'validation_{key}.csv',d,ids['validation'],vp)
  rows.append(row)
 eligible=[r for r in rows if r['converged']];assert eligible,'All ARIMAX candidates failed to converge'
 best=min(eligible,key=lambda r:r['rmse']);save(R/'CANDIDATES.json',rows);save(R/'COMPLETE.json',dict(selected=best,training_seconds=time.time()-t0,fit_end=str(d.time.iloc[end]),final_refit=False,policy='Selected training-only fit retained; filter states update with observed data'))
 # Separate GARCH, same information cutoff, 24h integrated variance.
 G=B/'runs'/fold/'garch';G.mkdir(exist_ok=True);returns=100*np.diff(np.log(d.close.to_numpy()));t0=time.time();fit=arch_model(returns[:end],mean='Zero',vol='GARCH',p=1,q=1,dist='normal',rescale=False).fit(disp='off')
 fit.save if False else None
 assert fit.convergence_flag==0,'GARCH did not converge'
 pickle.dump(fit,open(G/'model.pkl','wb'));save(G/'COMPLETE.json',dict(parameters=fit.params.to_dict(),fit_returns=end,training_seconds=time.time()-t0,alpha_plus_beta=float(fit.params['alpha[1]']+fit.params['beta[1]'])))
def evaluate(fold):
 assert (B/'runs'/fold/'SELECTION_LOCK.json').exists()
 R=B/'runs'/fold/'arimax';d,ids,sc,x,y,start,end=inputs(fold);ev=ids['evaluation'];m=json.loads((R/'COMPLETE.json').read_text())
 if not (R/'EVALUATION.json').exists():
  fit=pickle.load(open(R/(m['selected']['key']+'.pkl'),'rb'));p=rolling(fit,y,x,ev,end);write_predictions(R/'evaluation_predictions.csv',d,ev,p);save(R/'EVALUATION.json',dict(model='arimax',fold=fold,seed=None,**metrics(d.close.iloc[ev+H].to_numpy(),p,d.close.iloc[ev].to_numpy())))
 G=B/'runs'/fold/'garch'
 if (G/'EVALUATION.json').exists():return
 fit=pickle.load(open(G/'model.pkl','rb'));r=100*np.diff(np.log(d.close.to_numpy()));v=np.empty(len(r));v[:end]=np.asarray(fit.conditional_volatility)**2;w,a,b=fit.params.to_numpy()
 for k in range(end,len(r)):v[k]=w+a*r[k-1]**2+b*v[k-1]
 preds=[]
 for i in ev:
  h=v[i];total=h
  for j in range(23):h=w+(a+b)*h;total+=h
  preds.append(total)
 actual=np.array([np.dot(r[i:i+24],r[i:i+24]) for i in ev]);p=np.array(preds);assert (p>0).all() and (actual>0).all()
 pd.DataFrame(dict(origin_time=d.time.iloc[ev].to_numpy(),target_time=d.time.iloc[ev+H].to_numpy(),actual_variance=actual,predicted_variance=p)).to_csv(G/'evaluation_predictions.csv',index=False)
 save(G/'EVALUATION.json',dict(model='garch',fold=fold,n=len(p),variance_rmse=float(np.sqrt(np.mean((p-actual)**2))),variance_mae=float(np.mean(abs(p-actual))),qlike=float(np.mean(np.log(p)+actual/p)),volatility_rmse=float(np.sqrt(np.mean((np.sqrt(p)-np.sqrt(actual))**2)))))
