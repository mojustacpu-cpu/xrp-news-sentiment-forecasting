from pathlib import Path
import json,hashlib,shutil,re
import numpy as np,pandas as pd,torch
B=Path(__file__).resolve().parents[2];O=B/'audit/results';R=Path('/Users/mahmoudali/Disk/Master 2025/cursor projects/crypto-forecasting-research');P=R/'news-enhanced-crypto-research/phase1';Q=R/'Research_Final_Experiments';seeds=[42,123,2026,7,99]
reuse=json.loads((P/'E0_E1_REUSE.json').read_text())['E0'];rows=[];sources={}
def sha(p):p=Path(p);sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return sources[str(p)]
for model in ['E0','E1B','E2_h3','E2_h6','E2_h12','E2_h24']:
 for seed in seeds:
  if model=='E0':cp=Path(next(x for x in reuse['seeds'] if x['seed']==seed)['checkpoint'])
  else:cp=P/model/'runs'/f'{model}_seed{seed}'/'best.pt'
  hp=cp.parent/'history.csv';h=pd.read_csv(hp);ck=torch.load(cp,map_location='cpu',weights_only=False);param=sum(v.numel() for v in ck['model'].values());selected=int(ck['epoch']);mins=h.loc[h.validation_mse.idxmin()];final=h.iloc[-1];best=h[h.epoch.eq(selected)].iloc[0];assert selected==int(mins.epoch);assert h.epoch.tolist()==list(range(1,len(h)+1));assert int(final.epoch)==selected+15 or int(final.epoch)==200
  rows.append(dict(model=model,seed=seed,parameters=param,selected_epoch=selected,epochs_run=int(final.epoch),checkpoint_matches_history_min=True,patience15_consistent=True,min_validation_mse=float(mins.validation_mse),first_training_mse=float(h.iloc[0].training_mse),selected_training_mse=float(best.training_mse),last_training_mse=float(final.training_mse),last_validation_mse=float(final.validation_mse),validation_increase_after_selected_percent=float(100*(final.validation_mse/mins.validation_mse-1)),training_reduction_after_selected_percent=float(100*(1-final.training_mse/best.training_mse)),validation_improvement_steps=int((h.validation_mse<h.validation_mse.cummin().shift(1).fillna(np.inf)).sum()),checkpoint=str(cp),checkpoint_sha256=sha(cp),history=str(hp),history_sha256=sha(hp)))
pd.DataFrame(rows).to_csv(O/'TRAINING_STABILITY_ALL_30_RUNS.csv',index=False)
f=pd.DataFrame(rows);summary=f.groupby('model').agg(selected_epoch_min=('selected_epoch','min'),selected_epoch_max=('selected_epoch','max'),epochs_run_min=('epochs_run','min'),epochs_run_max=('epochs_run','max'),validation_worse_at_stop=('validation_increase_after_selected_percent',lambda x:int((x>0).sum())),training_lower_after_best=('training_reduction_after_selected_percent',lambda x:int((x>0).sum())),parameters=('parameters','first'));summary.to_csv(O/'TRAINING_STABILITY_SUMMARY.csv')
# Verify audit package and human label completeness; do not populate labels.
D=Q/'02_GPT4O_Annotation_Validation';dest=B/'audit/annotation_package';fields=['sentiment','stance_xrp','uncertainty','xrp_specificity'];labels=[]
for k in [1,2]:
 src=D/'provenance_v2'/f'BLINDED_reviewer_{k}_EXPANDED.csv';z=pd.read_csv(src);assert len(z)==300 and z.audit_id.is_unique;labels.append(dict(reviewer=k,rows=len(z),nonempty_primary_field_cells=int(z[fields].notna().sum().sum()),complete_primary_rows=int(z[fields].notna().all(1).sum()),source=str(src),sha256=sha(src)))
 # Create a primary-four-field-only blinded file; other historical fields remain in preserved source.
 z[['audit_id','headline','article_url','published_at_utc']+fields+['insufficient_information','evidence_quote','reviewer_notes']].to_csv(dest/f'BLINDED_REVIEWER_{k}.csv',index=False)
saved=pd.read_csv(D/'audit/GPT4O_ANNOTATION_AUDIT.csv');print('AUDIT COLUMNS',saved.columns.tolist(),flush=True)
# Save privileged answer key separately. Reviewers must not see it before labelling.
cols=['audit_id','sampling_stratum']+fields
saved[[x for x in cols if x in saved]].to_csv(dest/'PRIVILEGED_ARCHIVE_KEY_DO_NOT_SHOW_REVIEWERS.csv',index=False)
status=dict(reviewers=labels,semantic_accuracy_status='pending independent human labels',claimed_human_accuracy=None,primary_fields=fields,structural_provenance_semantics_distinct=True)
(O/'ANNOTATION_EVIDENCE_AUDIT.json').write_text(json.dumps(status,indent=2));pd.DataFrame(columns=['comparison','stratum','field','n_paired','n_missing','MAE','signed_bias','Spearman_rho','exact_agreement','macro_F1','Cohen_kappa','notes']).to_csv(dest/'EVALUATION_TEMPLATE_EMPTY.csv',index=False)
# No negative-control fit may be inferred from input masks or unrelated historical runs.
configs={}
for name in ['E1B','E2_h3','E2_h6','E2_h12','E2_h24']:configs[name]=json.loads((P/name/'CONFIG.json').read_text());sha(P/name/'CONFIG.json')
control=dict(inspected_scope='Current frozen phase1 E0/E1B/E2 configs, all30checkpoints and saved comparison manifests',E0_parameters=564353,E1B_parameters=570497,E2_parameters=566401,E1B_extra_parameters=6144,E1B_extra_percent=100*6144/564353,E2_extra_parameters=2048,E2_extra_percent=100*2048/564353,matched_retrained_shuffled_news_found=False,matched_presence_recency_only_found=False,exact_capacity_matched_news_control_found=False,existing_frozen_perturbation_is_retrained_control=False,legacy_context='Other historical frozen permutation studies and near-matched LSTM controls do not supply these current return-study negative controls',new_training=False)
(O/'CONTROL_EVIDENCE_AUDIT.json').write_text(json.dumps(control,indent=2));(O/'AUDITED_SOURCE_HASHES.json').write_text(json.dumps(sources,indent=2));print('EVIDENCE AUDIT COMPLETE',summary.to_dict(),status,flush=True)
