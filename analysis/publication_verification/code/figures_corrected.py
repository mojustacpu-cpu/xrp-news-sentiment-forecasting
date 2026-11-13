from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
B=Path(__file__).resolve().parents[2];F=B/'figures';O=B/'audit/results';x=pd.read_csv(O/'BOOTSTRAP_ALL_CORRECTED.csv');plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,name):fig.savefig(F/(name+'.pdf'));fig.savefig(F/(name+'.png'),dpi=450);plt.close(fig)
def pick(period,group):return x[(x.period==period)&(x.group==group)&(x.block_hours==72)].iloc[0]
fig,ax=plt.subplots(figsize=(8.5,3.5));groups=[('validation','E1B vs E0'),('validation','E2_h6 vs E0'),('later','E1B vs E0')]
for i,(period,group) in enumerate(groups):
 r=pick(period,group)
 if period=='validation':ax.plot([r.adjusted_low,r.adjusted_high],[i,i],lw=7,color='#c8ccd0',label='99.5% validation adjustment' if i==0 else None)
 ax.plot([r.ci95_low,r.ci95_high],[i,i],lw=2.5,color='#318763' if 'E2' in group else '#bd642e',label='95% interval' if i==0 else None);ax.plot(r.difference,i,'o',color='black',ms=5)
ax.axvline(0,color='#555',ls='--');ax.set(yticks=range(3),yticklabels=[f'{g}: {p}' for p,g in groups],xlabel='Paired RMSE-risk difference (USDT): negative favours news',title='72-hour paired calendar blocks; 50,000 resamples');ax.invert_yaxis();ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.17);fig.tight_layout();save(fig,'fig06_paired_intervals')
fig,axs=plt.subplots(1,2,figsize=(10,8),sharey=True)
for ax,period in zip(axs,['validation','later']):
 z=x[(x.period==period)&(x.scope=='exploratory conditional')&(x.block_hours==72)].reset_index(drop=True)
 for i,r in z.iterrows():
  if r.interval_estimable:
   ax.plot([r.adjusted_low,r.adjusted_high],[i,i],color='#c8ccd0',lw=3);ax.plot([r.ci95_low,r.ci95_high],[i,i],color='#bd642e',lw=1.2)
  ax.plot(r.difference,i,'o',color='#bd642e',ms=4)
 ax.axvline(0,color='#555',ls='--');ax.set(yticks=range(len(z)),xlabel='E1B − E0 risk RMSE (USDT)',title=period.capitalize());ax.grid(alpha=.17);ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
a=x[(x.period=='validation')&(x.scope=='exploratory conditional')&(x.block_hours==72)];b=x[(x.period=='later')&(x.scope=='exploratory conditional')&(x.block_hours==72)];axs[0].set_yticklabels([f'{g} (n={i}/{j})' for g,i,j in zip(a.group,a.n,b.n)]);axs[1].tick_params(labelleft=False);axs[0].invert_yaxis();fig.suptitle('50,000 draws; n = validation/later; dark = 95%; pale = 99.84375%',fontsize=11);fig.tight_layout(rect=(0,0,1,.96));save(fig,'fig09_conditional_forest')
# Supplementary numerical precision figure (outside the main manuscript figure sequence).
fig,axs=plt.subplots(1,2,figsize=(10,4.2));acf=pd.read_csv(O/'PAIRED_LOSS_AUTOCORRELATION.csv')
for period,c in [('validation','#28638b'),('later','#bd642e'),('HBAR','#318763')]:
 z=acf[acf.period==period];axs[0].plot(z.lag_hours,z.paired_squared_loss_acf,'o-',color=c,label=period,ms=3)
axs[0].axhline(0,color='#555',lw=.7);axs[0].set(xlabel='Lag (hours)',ylabel='Paired squared-loss autocorrelation',title='a  Descriptive serial dependence');axs[0].legend();axs[0].grid(alpha=.17)
zz=x[(x.period=='later')&(x.block_hours==72)&x.group.isin(['High volatility','No current-hour news','No XRP-specific news within 6h','Other trailing24h article count'])].copy()
for i,(_,r) in enumerate(zz.iterrows()):
 axs[1].plot([r.adjusted_low_mc95_low,r.adjusted_low_mc95_high],[i,i],color='#28638b',lw=3);axs[1].plot(r.adjusted_low,i,'o',color='black',ms=4)
axs[1].axvline(0,color='#555',ls='--');axs[1].set(yticks=range(len(zz)),yticklabels=[s.replace('trailing24h','24 h') for s in zz.group],xlabel='Adjusted lower endpoint (USDT)',title='b  Monte Carlo 95% endpoint brackets');axs[1].xaxis.set_major_locator(MaxNLocator(3));axs[1].ticklabel_format(style='sci',axis='x',scilimits=(-3,3));axs[1].grid(alpha=.17);fig.tight_layout();save(fig,'audit_fig01_bootstrap_precision')
print('CORRECTED FIGURES SAVED')
