"""Score genuine reviewer labels; never fill missing labels or generate human evidence.
Usage: python score_human_audit.py --reviewer1 R1.csv --reviewer2 R2.csv [--consensus C.csv] --output NEW_DIRECTORY
Consensus must be a separately adjudicated file preserving original reviewer records.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.metrics import cohen_kappa_score,accuracy_score,f1_score,confusion_matrix
p=argparse.ArgumentParser();p.add_argument('--reviewer1',required=True);p.add_argument('--reviewer2',required=True);p.add_argument('--consensus');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
b=Path(__file__).resolve().parents[1];saved=pd.read_csv(b/'audit/GPT4O_ANNOTATION_AUDIT.csv').set_index('audit_id');r1=pd.read_csv(a.reviewer1).set_index('audit_id');r2=pd.read_csv(a.reviewer2).set_index('audit_id')
assert r1.index.is_unique and r2.index.is_unique and set(r1.index)==set(saved.index)==set(r2.index)
continuous=['sentiment','uncertainty','xrp_specificity'];categorical=['event_type','stance_xrp','reg_access_flag','surprise_flag'];rows=[]
def compare(x,y,field,mask,role,stratum):
 x=x.reindex(saved.index);y=y.reindex(saved.index);valid=mask&x.notna()&y.notna();n=int(valid.sum());z={'comparison':role,'stratum':stratum,'field':field,'n':n,'missing_or_unknown':int(mask.sum())-n}
 if n:
  if field in continuous:
   xx=pd.to_numeric(x[valid],errors='raise');yy=pd.to_numeric(y[valid],errors='raise');z.update(mae=float(abs(xx-yy).mean()),mean_signed_difference=float((xx-yy).mean()))
  else:
   xx=x[valid].astype(str);yy=y[valid].astype(str)
   if field!='event_type':xx=pd.to_numeric(xx).astype(float).astype(str);yy=pd.to_numeric(yy).astype(float).astype(str)
   labels=sorted(set(xx)|set(yy));z.update(exact_agreement=float(accuracy_score(xx,yy)),macro_f1=float(f1_score(xx,yy,average='macro',zero_division=0)),kappa=float(cohen_kappa_score(xx,yy)) if len(labels)>1 else None)
   pd.DataFrame(confusion_matrix(xx,yy,labels=labels),index=labels,columns=labels).to_csv(out/f'{role}_{stratum}_{field}_confusion.csv')
 rows.append(z)
cons=pd.read_csv(a.consensus).set_index('audit_id') if a.consensus else None
for stratum,mask in [('representative',saved.sampling_stratum.eq('representative_random')),('enriched',saved.sampling_stratum.ne('representative_random'))]:
 for field in continuous+categorical:
  compare(r1[field],r2[field],field,mask,'inter_reviewer',stratum)
  if cons is not None:compare(cons[field],saved[field],field,mask,'archive_vs_consensus',stratum)
pd.DataFrame(rows).to_csv(out/'METRICS.csv',index=False);(out/'STATUS.json').write_text(json.dumps({'human_provenance_requires_signed_review_records':True,'consensus_supplied':cons is not None,'nonempty_comparisons':sum(z['n']>0 for z in rows),'continuous_accuracy':'No arbitrary tolerance or discretization used; MAE reported','event_type':'single-label exact category comparison; multi-label labels require a separately frozen scoring rule','repeatability':'not measured by human agreement'},indent=2))
