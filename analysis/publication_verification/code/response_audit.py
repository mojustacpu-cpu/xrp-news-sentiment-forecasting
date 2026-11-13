from pathlib import Path
import json,hashlib,collections
import pandas as pd,numpy as np
B=Path(__file__).resolve().parents[2];O=B/'audit/results';R=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research');J=R/'subprojects/xrp_sentiment/data/annotation_jobs/20261025T025052Z';req=[];outs=[];files={};issues=[]
for kind,paths,records in [('input',sorted((J/'inputs').glob('*.jsonl')),req),('output',sorted((J/'outputs').glob('*output_file_id.jsonl')),outs)]:
 for p in paths:
  files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  for ln,line in enumerate(p.read_text().splitlines(),1):
   try:q=json.loads(line);q['_source_file']=p.name;records.append(q)
   except Exception as e:issues.append(dict(kind=kind,file=p.name,line=ln,issue='outer_JSON_parse',detail=str(e)))
required=['sentiment','stance_xrp','uncertainty','xrp_specificity'];invalid=[];reasons=collections.Counter();models=collections.Counter();response_ids=collections.Counter();parsed=0;schema=0;rangevalid=0
for q in outs:
 cid=q.get('custom_id');response_ids[cid]+=1
 try:
  rb=q['response']['body'];models[rb.get('model','unknown')]+=1;ch=rb['choices'][0];reasons[ch.get('finish_reason','unknown')]+=1;x=json.loads(ch['message']['content']);parsed+=1
  if not all(k in x for k in required):raise ValueError('missing_primary_field')
  schema+=1
  nums={k:float(x[k]) for k in required}
  if not all(np.isfinite(list(nums.values()))):raise ValueError('nonfinite_primary_field')
  if not(-1<=nums['sentiment']<=1 and nums['stance_xrp'] in [-1,0,1] and 0<=nums['uncertainty']<=1 and 0<=nums['xrp_specificity']<=1):raise ValueError('out_of_primary_range')
  rangevalid+=1
 except Exception as e:invalid.append(dict(custom_id=cid,file=q['_source_file'],issue=str(e)))
ri=collections.Counter(q.get('custom_id') for q in req);oi=collections.Counter(q.get('custom_id') for q in outs);headlines=[' '.join(str(q['body']['messages'][-1]['content']).lower().split()) for q in req]
summary=dict(input_records=len(req),input_unique_custom_ids=len(ri),input_duplicate_custom_id_records=sum(v-1 for v in ri.values()),input_distinct_normalised_headlines=len(set(headlines)),output_records=len(outs),output_unique_custom_ids=len(oi),output_duplicate_custom_id_records=sum(v-1 for v in oi.values()),requests_without_output=sorted(set(ri)-set(oi)),outputs_without_request=sorted(set(oi)-set(ri)),outer_parse_issues=len(issues),response_content_JSON_parsed=parsed,complete_primary_schema=schema,primary_fields_valid_numeric_ranges=rangevalid,invalid_primary_records=len(invalid),finish_reasons=dict(reasons),response_models=dict(models),semantic_accuracy='not measured by structural audit',legacy_scope='This saved batch is not all article annotation history; the95legacy sample records lack established snapshot metadata')
(O/'SAVED_BATCH_RESPONSE_AUDIT.json').write_text(json.dumps(summary,indent=2));(O/'RESPONSE_SOURCE_HASHES.json').write_text(json.dumps(files,indent=2));pd.DataFrame(invalid,columns=['custom_id','file','issue']).to_csv(O/'RESPONSE_ISSUES.csv',index=False);print(json.dumps(summary,indent=2))
