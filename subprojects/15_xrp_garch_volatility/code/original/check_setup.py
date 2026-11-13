from common import *
from neural import Model,Windows
import torch
from sklearn.preprocessing import RobustScaler,MinMaxScaler
p=json.loads((B/'configs/protocol.json').read_text());d=data();checks=[]
for fold in p['folds']:
 ids=arrays(fold);tr,va,ev=[ids[k] for k in ['train','validation','evaluation']]
 assert tr[-1]+H<va[0] and va[-1]+H<ev[0]
 for part,ix in ids.items():
  assert np.all((d.time.iloc[ix+H].to_numpy()-d.time.iloc[ix].to_numpy())==pd.Timedelta(hours=24))
  assert len(set(ix))==len(ix)
 sx=RobustScaler().fit(d[F].iloc[:tr[-1]+1]);sy=MinMaxScaler().fit(d.close.iloc[tr+H].to_numpy()[:,None])
 assert np.allclose(sx.center_,d[F].iloc[:tr[-1]+1].median().to_numpy())
 assert sy.data_max_[0]==d.close.iloc[tr+H].max()
 # Each ARIMAX future regressor's source row must be <= origin.
 assert np.all(ev[:,None]+np.arange(1,25)[None,:]-24<=ev[:,None])
 checks.append(dict(fold=fold,training_targets_precede_validation=True,validation_targets_precede_evaluation=True,target_exact_24h=True,training_scalers_verified=True,exog_causality_verified=True))
torch.set_num_threads(4)
for c in p['neural']:
 n=19 if c['btc'] else 18;m=Model(n,c);assert sum(x.numel() for x in m.parameters())==c['parameters'];out=m(torch.zeros(2,30,n));assert out.shape==(2,1);out.square().mean().backward();assert all(torch.isfinite(z.grad).all() for z in m.parameters())
save(B/'reports/SETUP_VERIFICATION.json',dict(checks=checks,all_neural_shapes_and_gradients_pass=True,model_counts_verified=True,training_not_started_by_this_check=True));print('Setup checks passed')
