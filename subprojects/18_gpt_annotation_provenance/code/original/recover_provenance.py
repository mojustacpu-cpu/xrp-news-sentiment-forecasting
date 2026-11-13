from pathlib import Path
import json,hashlib,re
import pandas as pd
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];J=ROOT/'subprojects/xrp_sentiment/data/annotation_jobs/20261025T025052Z';O=B/'provenance_v2';O.mkdir(exist_ok=True)
norm=lambda s:re.sub(r'\s+',' ',str(s)).strip().lower()
requests={};settings={}
for p in sorted((J/'inputs').glob('*.jsonl')):
 for line in p.open():
  q=json.loads(line);body=q['body'];headline=body['messages'][-1]['content'].removeprefix('Input: ');requests[norm(headline)]=(q,p.name);key=hashlib.sha256(json.dumps({k:v for k,v in body.items() if k!='messages'},sort_keys=True).encode()+body['messages'][0]['content'].encode()).hexdigest();settings[key]={k:v for k,v in body.items() if k!='messages'};settings[key]['system_prompt']=body['messages'][0]['content']
outputs={};invalid=[]
for p in sorted((J/'outputs').glob('*output_file_id.jsonl')):
 for line in p.open():
  q=json.loads(line);outputs[q['custom_id']]=q
  try:
   v=json.loads(q['response']['body']['choices'][0]['message']['content']);assert all(k in v for k in ['sentiment','stance_xrp','uncertainty','event_type','reg_access_flag','surprise_flag','xrp_specificity'])
  except Exception as e:invalid.append({'custom_id':q.get('custom_id'),'error':str(e)})
s=pd.read_csv(B/'audit/GPT4O_ANNOTATION_AUDIT.csv');rows=[];linked=[]
for _,a in s.iterrows():
 match=requests.get(norm(a.headline));row={'audit_id':a.audit_id,'record_id':a.record_id,'matched_by':'normalised exact headline' if match else 'unresolved','request_found':bool(match),'response_found':False}
 if match:
  q,p=match;out=outputs.get(q['custom_id']);row.update(custom_id=q['custom_id'],input_file=p,model=q['body']['model'],response_found=bool(out));linked.append({'audit_id':a.audit_id,'request':q,'response':out})
 rows.append(row)
pd.DataFrame(rows).to_csv(O/'SAMPLE_REQUEST_LINKAGE.csv',index=False);(O/'MATCHED_REQUESTS_RESPONSES.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in linked));(O/'RECOVERED_SETTINGS.json').write_text(json.dumps(settings,indent=2));(O/'SUMMARY.json').write_text(json.dumps({'request_count':len(requests),'output_count':len(outputs),'sample_count':len(s),'sample_requests_linked':len(linked),'invalid_structural_responses':invalid,'invalid_check_scope':'JSON and required-key presence only; semantic validity not established','linkage_limitation':'exact normalized headline matching, not proof that the final merged record used this response','repeatability':'not run; historical outputs are not repeat calls','human_audit':'pending'},indent=2));print((O/'SUMMARY.json').read_text()[:2000])
# New expanded blinded schema; preserve previous four-field package.
fields=['sentiment','event_type','xrp_specificity','uncertainty','stance_xrp','reg_access_flag','surprise_flag']
for role in ['reviewer_1','reviewer_2']:
 d=s[['audit_id','headline','article_url','published_at_utc']].sample(frac=1,random_state=801)
 for f in fields+['insufficient_information','evidence_quote','reviewer_notes']:d[f]=''
 d.to_csv(O/f'BLINDED_{role}_EXPANDED.csv',index=False)
