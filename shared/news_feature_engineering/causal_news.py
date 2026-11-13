"""Retrospective publication-proxy features; no price or legacy impact inputs."""
import numpy as np
import pandas as pd

def aggregate(articles,origins,half_life=12.,lookback=48.,delay=0.,epsilon=1e-9):
 if half_life<=0 or lookback<=0 or delay<0:raise ValueError('Positive half life/lookback and nonnegative delay required')
 publication=pd.to_datetime(articles.published_at_utc,utc=True)
 if publication.isna().any():raise ValueError('Missing publication timestamp')
 available=(publication+pd.Timedelta(hours=delay)).dt.ceil('h')
 order=np.argsort(available.to_numpy(),kind='stable')
 times=available.astype('int64').to_numpy()[order]/3.6e12
 values=articles[['sentiment','xrp_specificity','stance_xrp','uncertainty']].to_numpy(dtype=float)[order]
 if not np.isfinite(values).all():raise ValueError('Missing semantic inputs')
 if (np.abs(values[:,0])>1).any() or ((values[:,1]<0)|(values[:,1]>1)).any():raise ValueError('Out of range')
 origins=pd.DatetimeIndex(pd.to_datetime(origins,utc=True))
 if origins.has_duplicates or not origins.is_monotonic_increasing:raise ValueError('Ordered unique origins required')
 output=[]
 for stamp in origins:
  t=stamp.value/3.6e12;left=np.searchsorted(times,t-lookback,side='left');right=np.searchsorted(times,t,side='right')
  age=t-times[left:right];assert (age>=0).all()
  subset=values[left:right];decay_weight=np.exp(-np.log(2)*age/half_life);w=decay_weight*subset[:,1];mass=w.sum()
  recent_start=np.searchsorted(times,t-1,side='right');recent=values[recent_start:right]
  active=len(recent)>0;means=recent.mean(axis=0) if active else np.zeros(4)
  last_age=min(lookback,t-times[right-1]) if right else lookback
  output.append({'equation_sentiment':float(w@subset[:,0]/(epsilon+mass)),'no_relevant_news':int(mass==0),'eligible_news_present':int(right>left),'news_intensity':float(np.log1p(decay_weight.sum())),'equation_stance':float(w@subset[:,2]/(epsilon+mass)),'equation_uncertainty':float(w@subset[:,3]/(epsilon+mass)),'eligible_articles':right-left,'relevance_weight_sum':float(mass),'direct_sentiment':float(means[0]),'direct_specificity':float(means[1]),'direct_stance':float(means[2]),'direct_uncertainty':float(means[3]),'direct_news_present':int(active),'hours_since_latest_news_capped48':float(last_age),'maximum_available_timestamp':str(available.iloc[order[right-1]]) if right>left else None})
 return pd.DataFrame(output,index=origins).rename_axis('origin_time')
