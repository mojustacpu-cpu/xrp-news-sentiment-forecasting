from common import *
import time,random,pickle
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import RobustScaler,MinMaxScaler
class Model(nn.Module):
 def __init__(self,n,c):
  super().__init__();h=c['hidden'];bi=c['bidirectional']
  self.lstm=nn.LSTM(n,h,2,batch_first=True,bidirectional=bi,dropout=.2)
  self.fc1=nn.Linear(h*(2 if bi else 1),64);self.fc_price=nn.Linear(64,1);self.drop=nn.Dropout(.2);self.relu=nn.ReLU()
 def forward(self,x):
  out,_=self.lstm(x);return self.fc_price(self.drop(self.relu(self.fc1(out[:,-1,:]))))
class Windows(Dataset):
 def __init__(self,x,y,ids):self.x=x;self.y=y;self.ids=ids
 def __len__(self):return len(self.ids)
 def __getitem__(self,k):
  i=self.ids[k];return self.x[i-L+1:i+1],self.y[i+H]
def predict(m,dl):
 m.eval();out=[]
 with torch.no_grad():
  for x,y in dl:out.append(m(x).numpy().ravel())
 return np.concatenate(out)
def train(fold,c,seed,protocol):
 R=B/'runs'/fold/f"{c['name']}_seed{seed}";R.mkdir(parents=True,exist_ok=True)
 if (R/'COMPLETE.json').exists():return
 assert not (R/'STARTED.json').exists(),f'Incomplete run preserved; no automatic retraining: {R}'
 save(R/'STARTED.json',dict(pid=os.getpid(),started_utc=str(pd.Timestamp.now(tz='UTC')),config=c,seed=seed))
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 d=data();ids=arrays(fold);features=[k for k in F if c['btc'] or k!='BTC_Close'];raw=d[features].to_numpy();price=d.close.to_numpy()
 sx=RobustScaler().fit(raw[:ids['train'][-1]+1]);sy=MinMaxScaler().fit(price[ids['train']+H,None])
 x=torch.tensor(sx.transform(raw),dtype=torch.float32);y=torch.tensor(sy.transform(price[:,None]),dtype=torch.float32)
 assert sx.n_features_in_==len(features)
 pickle.dump(dict(x=sx,y=sy,features=features,x_last_fit_row=int(ids['train'][-1]),y_target_rows=ids['train']+H),open(R/'scalers.pkl','wb'))
 gen=torch.Generator().manual_seed(seed)
 dl=DataLoader(Windows(x,y,ids['train']),batch_size=32,shuffle=True,generator=gen)
 vd=DataLoader(Windows(x,y,ids['validation']),batch_size=256)
 m=Model(len(features),c);assert sum(p.numel() for p in m.parameters())==c['parameters']
 opt=torch.optim.Adam(m.parameters(),lr=.001);best=float('inf');stale=0;history=[];t0=time.time();selected=0
 for epoch in range(1,protocol['max_epochs']+1):
  m.train();total=0.;count=0
  for xx,yy in dl:
   opt.zero_grad();pred=m(xx);loss=((pred-yy)**2).mean();assert torch.isfinite(loss)
   loss.backward();opt.step();total+=float(loss.detach())*len(xx);count+=len(xx)
  vp=predict(m,vd);vl=float(np.mean((vp-y[ids['validation']+H].numpy().ravel())**2));assert np.isfinite(vl)
  improved=vl<best
  if improved:
   best=vl;stale=0;selected=epoch
   q=R/'best.tmp';torch.save(dict(model=m.state_dict(),config=c,features=features,seed=seed,epoch=epoch,validation_mse=vl,protocol_sha256=sha(B/'configs/protocol.json')),q);q.replace(R/'best.pt')
  else:stale+=1
  history.append(dict(epoch=epoch,training_mse=total/count,validation_mse=vl,elapsed_seconds=time.time()-t0))
  pd.DataFrame(history).to_csv(R/'history.csv',index=False)
  save(R/'PROGRESS.json',dict(epoch=epoch,selected_epoch=selected,validation_mse=vl,best_validation_mse=best,stale_epochs=stale,elapsed_seconds=time.time()-t0))
  print(f'{fold} {c["name"]} seed={seed} epoch={epoch} train={total/count:.8g} val={vl:.8g} best={selected}',flush=True)
  if stale>=protocol['patience']:break
 m.load_state_dict(torch.load(R/'best.pt',weights_only=False)['model']);vp=sy.inverse_transform(predict(m,vd)[:,None]).ravel()
 write_predictions(R/'validation_predictions.csv',d,ids['validation'],vp)
 save(R/'COMPLETE.json',dict(model=c['name'],fold=fold,seed=seed,selected_epoch=selected,epochs_run=epoch,parameter_count=c['parameters'],training_seconds=time.time()-t0,validation_metrics=metrics(price[ids['validation']+H],vp,price[ids['validation']]),checkpoint_sha256=sha(R/'best.pt')))
def evaluate(fold,c,seed):
 R=B/'runs'/fold/f"{c['name']}_seed{seed}"
 if (R/'EVALUATION.json').exists():return
 assert (R/'COMPLETE.json').exists() and (B/'runs'/fold/'SELECTION_LOCK.json').exists()
 torch.set_num_threads(4);d=data();ids=arrays(fold)['evaluation'];sc=pickle.load(open(R/'scalers.pkl','rb'));sx,sy=sc['x'],sc['y'];x=torch.tensor(sx.transform(d[sc['features']].to_numpy()),dtype=torch.float32);y=torch.tensor(sy.transform(d.close.to_numpy()[:,None]),dtype=torch.float32)
 m=Model(len(sc['features']),c);ck=torch.load(R/'best.pt',weights_only=False);m.load_state_dict(ck['model']);p=sy.inverse_transform(predict(m,DataLoader(Windows(x,y,ids),batch_size=256))[:,None]).ravel()
 write_predictions(R/'evaluation_predictions.csv',d,ids,p);save(R/'EVALUATION.json',dict(model=c['name'],fold=fold,seed=seed,**metrics(d.close.iloc[ids+H].to_numpy(),p,d.close.iloc[ids].to_numpy())))
