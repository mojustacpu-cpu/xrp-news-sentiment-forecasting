from pathlib import Path
import pandas as pd,json,re,hashlib
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];O=B/'provenance_v2';p=ROOT.parent/'news_sentiment_bilstm/data/raw/raw/complete_sentiment_raw.csv';d=pd.read_csv(p,low_memory=False);s=pd.read_csv(B/'audit/GPT4O_ANNOTATION_AUDIT.csv');s=s[s.annotation_origin.eq('original_saved_gpt_json')];fields=['sentiment','stance_xrp','uncertainty','event_type','reg_access_flag','surprise_flag','xrp_specificity'];norm=lambda x:re.sub(r'\s+',' ',str(x)).strip().lower();d['key']=pd.to_datetime(d.timestamp,utc=True).astype(str)+'|'+d.headline.map(norm);rows=[];raw=[]
for _,a in s.iterrows():
 key=str(pd.to_datetime(a.published_at_utc,utc=True))+'|'+norm(a.headline);q=d[d.key.eq(key)];record={'audit_id':a.audit_id,'exact_headline_time_matches':len(q)}
 if len(q)==1:
  j=q.iloc[0].gpt_json
  try:
   v=json.loads(j);matches={f:(abs(float(a[f])-float(v[f]))<=1e-9 if f!='event_type' else str(a[f]).strip()==str(v[f]).strip()) for f in fields};record.update(json_valid=True,all_fields_match=all(matches.values()),mismatched_fields=','.join(f for f,x in matches.items() if not x));raw.append({'audit_id':a.audit_id,'source_row':int(q.index[0]),'saved_json':v})
  except Exception as e:record.update(json_valid=False,error=str(e))
 rows.append(record)
r=pd.DataFrame(rows);r.to_csv(O/'LEGACY_JSON_COMPARISON.csv',index=False);(O/'LEGACY_JSON_MATCHES.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in raw));summary={'sample_articles':len(s),'uniquely_matched':int(r.exact_headline_time_matches.eq(1).sum()),'all_seven_fields_match':int(r.get('all_fields_match',pd.Series(dtype=bool)).fillna(False).sum()),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_path':str(p),'limitation':'Saved JSON consistency only. Original request, model version, generation settings and human accuracy not established.'};(O/'LEGACY_JSON_VERIFICATION.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
