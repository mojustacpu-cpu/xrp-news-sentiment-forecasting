import torch
from torch import nn
class NewsCorrection(nn.Module):
 def __init__(self,n=8,gated=True):
  super().__init__();self.gated=gated
  self.encoder=nn.GRU(n,16,num_layers=1,batch_first=True)
  self.dropout=nn.Dropout(.2);self.correction=nn.Linear(16,1)
  nn.init.zeros_(self.correction.weight);nn.init.zeros_(self.correction.bias)
  self.gate=nn.Linear(16,1) if gated else None
 def forward(self,x,base_scaled,mask,target_scale,disabled=False):
  _,hidden=self.encoder(x);h=self.dropout(hidden[-1]);delta=self.correction(h)
  gate=torch.sigmoid(self.gate(h)) if self.gated else torch.ones_like(delta)
  effective=mask*gate
  if disabled:effective=torch.zeros_like(effective)
  correction_scaled=target_scale*effective*delta
  return base_scaled+correction_scaled,gate,delta,correction_scaled
