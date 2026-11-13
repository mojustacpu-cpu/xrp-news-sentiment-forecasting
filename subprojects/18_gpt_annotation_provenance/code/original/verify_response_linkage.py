from pathlib import Path
import json,hashlib
import pandas as pd
B=Path(__file__).resolve().parents[1];O=B/'provenance_v2';d=pd.read_csv(B/'audit/GPT4O_ANNOTATION_AUDIT.csv').set_index('audit_id');fields=['sentiment','stance_xrp','uncertainty','event_type','reg_access_flag','surprise_flag','xrp_specificity'];rows=[]
for line in (O/'MATCHED_REQUESTS_RESPONSES.jsonl').open():
 q=json.loads(line);response=q.get('response');aid=q['audit_id']
 if not response:continue
 raw=response['response']['body']['choices'][0]['message']['content'];v=json.loads(raw)
 for f in fields:
  archived=d.loc[aid,f];returned=v.get(f)
  try:
   same=abs(float(archived)-float(returned))<=1e-9 if f!='event_type' else str(archived).strip()==str(returned).strip()
  except (TypeError,ValueError):same=False
  rows.append({'audit_id':aid,'field':f,'archive_value':archived,'response_value':returned,'matches':same,'custom_id':q['request']['custom_id'],'response_content_sha256':hashlib.sha256(raw.encode()).hexdigest()})
r=pd.DataFrame(rows);r.to_csv(O/'ARCHIVE_RESPONSE_FIELD_COMPARISON.csv',index=False);article=r.groupby('audit_id').matches.all();s={'linked_articles_with_responses':len(article),'all_seven_fields_match':int(article.sum()),'articles_with_mismatch':int((~article).sum()),'field_match_counts':r.groupby('field').matches.sum().astype(int).to_dict(),'mismatched_fields':r.loc[~r.matches,['audit_id','field']].to_dict('records'),'limitation':'Agreement verifies saved-value consistency for matched headlines only; not semantic accuracy, independent human validation or API repeatability.'};(O/'RESPONSE_LINKAGE_VERIFICATION.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
