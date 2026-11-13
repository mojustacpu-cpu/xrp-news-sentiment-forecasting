from pathlib import Path
import csv,json,hashlib,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches,Pt
B=Path(__file__).resolve().parents[1]; R=B/'reports'; O=R/'final_model_comparison'; O.mkdir(exist_ok=True)
source=R/'CODEX_XRP_CONTROLLED_EXPERIMENT_REPORT.docx'
sha=hashlib.sha256(source.read_bytes()).hexdigest()
(O/'artifact.md').write_text(f'''# Comparison report contract
Reference: {source}
SHA256: {sha}
Reference has nine pages, one portrait section, 8.3 by 11.7 inches, top/bottom .65 inches, left/right .7 inches. Reference pages already rendered and inspected in reports/word_qa.
Preserve original nine pages and all tables, figures, references and methodological disclosures, except title wording. Calibri normal 10.5pt, black titles/headings, 7pt after normal paragraphs, captions 9pt italic; light blue table headers with gray borders. Figures 6.7 inches wide. Reuse actual source document and its styles.
Editable slots: first Title paragraph subject; add two new comparison explanation pages using existing Heading 1, body and caption styles. New pages extend the report with metric graphics and model rationale; no numerical changes. Preserve section properties, styles, numbering, original media and relationships.
''')
rows={r['model']:r for r in csv.DictReader(open(R/'PRICE_SUMMARY.csv')) if r['fold']=='main'}
keys=['bilstm_btc','bilstm_no_btc','lstm_btc','lstm_matched_btc','arimax','persistence']
labels=['BiLSTM + BTC','BiLSTM no BTC','LSTM 128 + BTC','LSTM 210 + BTC','ARIMAX','Last price']
colors=['#176b87','#df923c','#648bca','#9380b1','#47967c','#929292']
fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
for ax,(field,title,mult) in zip(axes.flat,[('rmse_mean','RMSE in USDT — lower is better',1),('mae_mean','MAE in USDT — lower is better',1),('r2_mean','R² — higher is better',1),('direction_mean','Direction accuracy in percent',100)]):
 vals=[float(rows[k][field])*mult for k in keys]
 ax.barh(labels,vals,color=colors);ax.invert_yaxis();ax.set_title(title,fontsize=11)
 ax.set_xlim(0,max(vals)*1.22);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 for i,v in enumerate(vals):ax.text(v+max(vals)*.015,i,f'{v:.2f}' if mult==100 else f'{v:.4f}',va='center',fontsize=9)
 ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'all_price_metrics.png',dpi=200);plt.close(fig)
d=Document(source)
d.paragraphs[0].text='XRP forecasting model comparison after methodological corrections'
def p(s):d.add_paragraph(s)
def page(s):d.add_page_break();d.add_heading(s,1)
def cap(s):
 x=d.add_paragraph(s)
 for r in x.runs:r.italic=True;r.font.size=Pt(9)
page('Comparing all corrected price forecasting results')
p('This report uses only the completed controlled study. It compares the final fitted models on the same 5,262 hourly forecast origins and exact 24 hour targets. Neural values are averages across five random seeds; ARIMAX has one validation-selected fit. No original-method scores are mixed into these figures.')
d.add_picture(str(O/'all_price_metrics.png'),width=Inches(6.7))
cap('Figure 5. Four views of the same main evaluation. Neural bars show the mean of five individual scores, not an ensemble prediction. Last price predicts no change: its direction score counts unchanged closes and does not measure up/down forecasting skill.')
p('Read the first two panels as forecast errors: shorter bars are better. ARIMAX has the smallest errors among fitted models, while BiLSTM with Bitcoin is the best neural configuration by mean RMSE and MAE. The R² panel confirms the same level-error ranking. The direction panel measures a different objective: LSTM 128 has slightly higher sign accuracy than BiLSTM with Bitcoin despite its larger price errors.')
p('The Bitcoin comparison removes only BTC_Close from BiLSTM. Its inclusion reduces mean RMSE by 19.13%. BiLSTM with Bitcoin reduces mean RMSE by 5.70% relative to LSTM 128 and 10.61% relative to the near parameter-matched LSTM 210. ARIMAX reduces RMSE by 30.58% relative to BiLSTM with Bitcoin. These are measured differences, not significance tests or guaranteed future gains.')
page('Why these models were selected and what was corrected')
d.add_heading('The purpose of each model',2)
p('LSTM is the recurrent reference because its gated memory can learn temporal dependencies within the observed 30 hour sequence (Hochreiter and Schmidhuber, 1997). The standard LSTM tests a simpler recurrent model. The larger LSTM gives a near parameter-matched comparison, reducing the chance that an apparent BiLSTM benefit is explained only by having more weights.')
p('BiLSTM processes the same observed sequence in both directions (Schuster and Paliwal, 1997). It can represent relationships across that historical window, but it receives no future candles. The Bitcoin ablation tests whether a market price covariate adds predictive information in this specification. Bitcoin close is not a direct measure of liquidity. The results support its inclusion here without proving a causal explanation for XRP movements.')
p('ARIMAX provides a statistical comparator: regression on Bitcoin and the nine indicators with ARIMA errors. It tests whether a relatively simple dynamic specification can compete with recurrent networks. Its 24 hour lagged external variables avoid using realised future inputs. Its different target transformation and predictor timing mean that this comparison tests complete forecasting specifications, not architecture alone. Orders are selected using validation only; see the statsmodels reference.')
p('GARCH models changing conditional variance and clustered volatility (Bollerslev, 1986). It answers how large movements may be, not whether price will rise or fall. It therefore has its own realised variance target, QLIKE and variance errors. Its results cannot be put in the price R² ranking, and no claim is made that it improved a price model.')
d.add_heading('Corrections that define this report',2)
p('The corrected study uses an exact +24 hour target instead of the archived effective 25 hour alignment. It fits both input and target transformations on training data only; selects epochs and ARIMAX orders using a separate chronological validation block; and purges overlapping labels at split boundaries. Earlier rolling evaluations refit models and transformations using only earlier information. The original 30 hour input window and nine indicators are retained. No sentiment features or attention are added.')
p('These corrections change the experimental protocol, so score changes relative to the archived reports cannot be attributed to one correction alone. Within this corrected study, matched evaluation timestamps make the model comparisons interpretable. The ranking is supported by two earlier rolling periods, although those periods use only one neural seed and the larger LSTM is tested only in the main period.')
d.add_heading('Research conclusion',2)
p('BiLSTM with Bitcoin is the strongest neural configuration by average price error in this study. ARIMAX is the strongest fitted price model and was preferred using validation performance alone. However, none of the fitted configurations improves on last price in mean evaluation RMSE. The appropriate conclusion is conditional evidence for BiLSTM over LSTM and for including Bitcoin, not that a complex model necessarily beats a simple price forecast. GARCH remains a complementary volatility experiment. All results are retrospective historical evidence.')
output=O/'XRP_FINAL_MODEL_COMPARISON_REPORT.docx';d.save(output)
assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
with zipfile.ZipFile(source) as a,zipfile.ZipFile(output) as z:
 for name in a.namelist():
  if name.startswith('word/media/') or name in ['word/styles.xml','word/numbering.xml']:
   assert a.read(name)==z.read(name),name
(O/'BUILD_VERIFICATION.json').write_text(json.dumps({'reference_preserved':True,'original_media_styles_numbering_preserved':True,'source_sha256':sha,'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_results':'../PRICE_SUMMARY.csv','scope':'Corrected controlled results only; not archived original-method scores'},indent=2))
print(output)
