from pathlib import Path
import json,csv
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
B=Path(__file__).resolve().parents[1];R=B/'results';O=B/'reports';O.mkdir(exist_ok=True)
d=Document();s=d.sections[0];s.page_height=Inches(11.7);s.page_width=Inches(8.3);s.top_margin=s.bottom_margin=Inches(.65);s.left_margin=s.right_margin=Inches(.7)
for name in ['Normal','Title','Heading 1','Heading 2']:
 st=d.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)

for st in d.styles:
 for border in list(st.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
d.styles['Normal'].font.size=Pt(10.5);d.styles['Normal'].paragraph_format.space_after=Pt(7)
def p(t):d.add_paragraph(t)
def h(t):d.add_heading(t,1)
def page(t):d.add_page_break();h(t)
def table(headers,rows,widths=None):
 t=d.add_table(rows=1, cols=len(headers));t.autofit=False
 if widths:
  for col,w in zip(t.columns,widths):col.width=Inches(w)
 for i,v in enumerate(headers):t.rows[0].cells[i].text=str(v)
 for row in rows:
  for c,v in zip(t.add_row().cells,row):c.text=str(v)
 for ri,row in enumerate(t.rows):
  for ci,c in enumerate(row.cells):
   if widths:c.width=Inches(widths[ci])
   tc=c._tc.get_or_add_tcPr();bd=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');bd.append(e)
   tc.append(bd)
   if ri==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'DFEAF2');tc.append(sh)
   for par in c.paragraphs:
    par.paragraph_format.space_after=Pt(5);par.paragraph_format.space_before=Pt(5)
    for run in par.runs:run.font.size=Pt(9);run.bold=ri==0
  pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
 t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
 d.add_paragraph().paragraph_format.space_after=Pt(0)
def pic(path):d.add_picture(str(path),width=Inches(6.7))
def read(path):return list(csv.DictReader(open(path)))

protocol=json.loads((B/'configs/protocol.json').read_text());audit=json.loads((O/'FINAL_NUMERICAL_VERIFICATION.json').read_text());assert audit['status']=='passed'
summary=read(O/'PRICE_SUMMARY.csv');runs=read(O/'PRICE_RESULTS_ALL_RUNS.csv');main=[r for r in summary if r['fold']=='main'];sel=json.loads((O/'VALIDATION_SELECTION.json').read_text());selection=next(x for x in sel if x['fold']=='main')
labels={'bilstm_btc':'BiLSTM with BTC','bilstm_no_btc':'BiLSTM without BTC','lstm_btc':'LSTM 128 with BTC','lstm_matched_btc':'LSTM 210 with BTC','arimax':'ARIMAX','persistence':'Last price'}
def f(x,n=5):
 try:return f"{float(x):.{n}f}" if x and str(x)!='nan' else '—'
 except:return '—'
def caption(t):
 par=d.add_paragraph(t)
 for r in par.runs:r.italic=True;r.font.size=Pt(9)
def mean(name):return float(next(r['rmse_mean'] for r in main if r['model']==name))
def comparison(a,b):
 change=100*(1-mean(a)/mean(b));return f"{labels[a]} has {abs(change):.2f}% {'lower' if change>0 else 'higher'} mean RMSE than {labels[b]}."
d.add_heading('Codex XRP indicators recommendations',0)
p('Controlled technical forecasting using archived hourly data through 25 October 2026. This study compares BiLSTM, LSTM, a near parameter-matched LSTM, and ARIMAX on exactly 24 hour price targets. Last-price predictions use the same observations. GARCH is reported separately for volatility. No attention or GPT sentiment features are used.')
h('Measured conclusions')
p(comparison('bilstm_btc','lstm_btc')+' '+comparison('bilstm_btc','lstm_matched_btc'))
p(comparison('bilstm_no_btc','bilstm_btc')+' Including Bitcoin reduced mean RMSE by '+f(100*(1-mean('bilstm_btc')/mean('bilstm_no_btc')),2)+'% relative to omitting it. Bitcoin also improved the seed42 BiLSTM in both earlier rolling periods. This supports keeping Bitcoin in this tested configuration, not a causal market claim.')
p(comparison('arimax','bilstm_btc')+' '+comparison('arimax','bilstm_no_btc')+' '+comparison('arimax','lstm_btc'))
p('The preferred fitted configuration selected using validation only is '+labels[selection['selected_configuration']]+', with mean validation RMSE '+f(selection['mean_validation_rmse'])+' USDT. Final evaluation is used to describe performance, not to revise this selection. No model is forced to win.')
p('ARIMAX had 0.26% higher RMSE than last price in the main period; none of the fitted configurations beat last price on mean RMSE. ARIMAX did have slightly lower MAE and higher directional accuracy. These distinctions prevent treating one score as evidence of superiority on every forecasting objective.')
h('Main evaluation scores')
table(['Model','Runs','RMSE mean','RMSE SD','MAE mean','R² mean'],[[labels[r['model']],r['runs'],f(r['rmse_mean']),f(r['rmse_sd']),f(r['mae_mean']),f(r['r2_mean'])] for r in main],[1.8,.45,1.1,1.1,1.1,1.15])
caption('Table 1. Five independently initialised fits per neural configuration; one ARIMAX fit and one deterministic last-price calculation. RMSE and MAE are in USDT. Scores are means of individual model metrics, not ensemble metrics. SD is variation across seeds, not a confidence interval over market outcomes.')
p('The main evaluation period has been examined in previous research. Training and model selection are now separated correctly, but this remains retrospective historical evidence, not a completely unseen prospective test. The original 25 hour experiments are preserved and not pooled with these exact24hour results.')
page('Fixed architecture and chronological design')
table(['Model','Inputs','Layers and width','Parameters'],[[labels[c['name']],19 if c['btc'] else 18,('BiLSTM ' if c['bidirectional'] else 'LSTM ')+str(c['hidden'])+' × 2; dense64',f"{c['parameters']:,}"] for c in protocol['neural']],[1.9,.55,2.95,1.3])
caption('Table 2. Two recurrent layers; bidirectional width is per direction. All neural networks use the inherited last sequence output readout, dense64 ReLU, dropout0.2 and one scalar price output. LSTM210 differs from BiLSTM128 by 2,264 parameters, about0.4%, so parameter matching is approximate.')
p('Inputs are OHLCV; EMA20, MACD Signal, SMA20, SMA10, EMA10, Volatility10, MACD, MFI and OBV; return, rolling return volatility, relative price to SMA10 and SMA20; plus real BTC_Close except in the ablation. Bitcoin close is a market price covariate, not a direct liquidity measurement.')
p('Every neural input window contains the 30 completed hourly candles ending at origin t. The target is close at t+24. Bidirectional processing operates within the observed window; it does not provide future candles. RobustScaler fits unique training-window rows only and MinMaxScaler fits training targets only. Scalers are frozen for validation and evaluation.')
p('Adam learning rate0.001, batch32, seed set42,123,2026,7,99, maximum200epochs and early stopping patience15 are fixed before evaluation. The saved checkpoint minimises chronological validation MSE. Configuration selection uses mean validation price RMSE over all five neural seeds or validation-selected ARIMAX RMSE. The representative seed is selected on validation only; the report retains every seed.')
for fold in ['main','rolling_1','rolling_2']:
 z=protocol['folds'][fold];p(f"{fold}: {z['train']['n']:,} training samples; {z['validation']['n']:,} validation samples; {z['evaluation']['n']:,} evaluation origins from {z['evaluation']['first_origin']} to {z['evaluation']['last_origin']}.")
p('Training targets end strictly before the first validation origin; validation targets end strictly before the first evaluation origin. Each rolling fold has freshly trained weights and newly fitted scalers, using only information available before that fold. Later-period weights are never reused. Evaluation origins match the archived comparison; target times are one hour earlier because the former effective horizon was25hours.')
p('ARIMAX regresses log price on Bitcoin plus nine indicators with ARIMA errors. External inputs are lagged24hours, so no future realised inputs are consumed. Orders(0,1,0),(1,1,0),(0,1,1) are selected on validation RMSE, and training-only parameters remain fixed with causal state updates. Log predictions are exponentiated without a mean bias correction.')
page('Direction and comparison with last price')
table(['Model','Direction %','RMSE ratio','RMSE gain %'],[[labels[r['model']],f(float(r['direction_mean'])*100,2),f(r['rmse_relative_to_persistence'],4),f(r['rmse_improvement_pct'],2)] for r in main],[2.7,1.25,1.25,1.5])
caption('Table 3. Ratio = model RMSE / last-price RMSE; below1 is better. Gain =100×(1−ratio); positive values indicate improvement. Direction is exact sign agreement including zero movements. Last price predicts zero change, so its direction score is the frequency of unchanged actual closes, not up/down skill.')
pic(O/'main_comparison.png')
caption('Figure 1. Main price RMSE and one seed standard deviation. Every model uses the same5,262 origins and target timestamps. Single-fit statistical and deterministic results have no seed dispersion estimate.')
p('A model can have high price R² while having weak directional accuracy or failing to beat last price. These measures answer different questions. Lower RMSE and MAE indicate smaller level errors; direction measures movement sign. The same target series is used for every price metric.')
p('Five seeds measure sensitivity to neural initialisation and training order. They are not five independent market datasets. Overlapping 24hour forecast intervals also make errors serially dependent. No significance claim is inferred from the error bars.')
page('Individual neural runs and selected epochs')
rr=[r for r in runs if r['fold']=='main' and r['model'] not in ['arimax','persistence']]
table(['Model','Seed','Epoch','Run epochs','RMSE','MAE','R²'],[[labels[r['model']],r['seed'].replace('.0',''),r['selected_epoch'].replace('.0',''),r['epochs_run'].replace('.0',''),f(r['rmse']),f(r['mae']),f(r['r2'])] for r in rr],[1.8,.55,.55,.75,1.05,1.05,.95])
caption('Table 4. Selected epoch comes only from validation loss; run epochs includes the patience period after the best checkpoint. All five seeds are reported, including poor-performing runs. Detailed directional accuracy, bias, duration and relative errors are saved in PRICE_RESULTS_ALL_RUNS.csv.')
p('Each run folder contains best.pt, scalers.pkl, history.csv, a training and validation loss chart, validation predictions and evaluation predictions. Seeds do not alter the dataset split. The original architecture was retained; the preprocessing, target indexing and checkpoint selection protocol were corrected.')
page('Actual prices and predictions')
pic(O/'main_curves_seed42.png')
caption('Figure 2. Seed42 is shown for every neural configuration because it was predefined, not chosen for its evaluation score. ARIMAX has one selected fit. Black lines are actual XRP prices. The curves use a common scale and identical target dates. All five seed metrics remain in Table4.')
page('Earlier rolling evaluations')
rolling=[r for r in summary if r['fold']!='main']
table(['Period','Model','RMSE','R²','Gain %'],[[r['fold'],labels[r['model']],f(r['rmse_mean']),f(r['r2_mean']),f(r['rmse_improvement_pct'],2)] for r in rolling],[1.0,2.45,1.1,1.05,1.1])
caption('Table 5. Earlier periods use freshly trained models with seed42 only, plus newly fitted statistical models. The size-matched LSTM is restricted to the main five-seed comparison. Rolling results test historical period sensitivity, not seed robustness.')
pic(O/'rolling_comparison.png')
caption('Figure 3. Seed42 results across the two earlier periods and the main period. Each period has its own chronological training, validation and evaluation blocks. Differences may reflect price scales and market conditions; RMSE gain relative to each period’s last-price forecast is provided in Table5.')
page('GARCH as a separate volatility experiment')
vr=read(O/'GARCH_RESULTS.csv');table(['Period','Variance RMSE','Variance MAE','QLIKE','Volatility RMSE'],[[r['fold'],f(r['variance_rmse'],4),f(r['variance_mae'],4),f(r['qlike'],4),f(r['volatility_rmse'],4)] for r in vr],[1.1,1.4,1.4,1.2,1.6])
caption('Table 6. Gaussian zero-mean GARCH(1,1), three fitted parameters. Variance units are squared percentage points; volatility RMSE is percentage points. QLIKE is mean log forecast variance plus realised variance divided by forecast variance. GARCH is not ranked by price R² or by a last-price price-error ratio.')
pic(O/'garch_curve.png')
caption('Figure 4. Main-period forecast and realised24hour volatility. The realised target sums24 future squared percentage log returns. Variance forecasts update only with returns available at each origin; no future returns enter the forecast.')
for fold in ['main','rolling_1','rolling_2']:
 v=json.loads((B/'runs'/fold/'garch/COMPLETE.json').read_text());p(f"{fold}: omega={v['parameters']['omega']:.6f}, alpha={v['parameters']['alpha[1]']:.6f}, beta={v['parameters']['beta[1]']:.6f}; alpha+beta={v['alpha_plus_beta']:.6f}. "+('At or very near the stationarity boundary; long-run unconditional variance is not reliably finite.' if v['alpha_plus_beta']>=.999999 else 'The fitted coefficient sum is below one.'))
p('No claim is made that these volatility estimates improve the price models. GARCH uses XRP returns only, with fresh training-only fits at each cutoff. It does not include Bitcoin, technical indicators, sentiment or attention as regressors.')
page('Additional saved fit details')
rroll=[r for r in runs if r['fold']!='main' and r['model'] not in ['arimax','persistence']]
table(['Period','Model','Seed','Selected epoch','Run epochs'],[[r['fold'],labels[r['model']],r['seed'].replace('.0',''),r['selected_epoch'].replace('.0',''),r['epochs_run'].replace('.0','')] for r in rroll],[1.0,2.7,.6,1.2,1.2])
caption('Table 7. Independently trained earlier-fold neural runs. All use seed42 and the same fixed architecture as the main configurations. No checkpoint is carried backward from a later period.')
ar=[]
for fold in ['main','rolling_1','rolling_2']:
 a=json.loads((B/'runs'/fold/'arimax/COMPLETE.json').read_text());order=a['selected']['order'];ar.append([fold,str(order),str(11+order[0]+order[2]),f(a['selected']['rmse']),a['fit_end']])
table(['Period','ARIMA order','Parameters','Validation RMSE','Fit end UTC'],ar,[.9,1.0,.8,1.2,2.8])
caption('Table 8. Statistical order selected by validation price RMSE. Ten regression coefficients plus AR and MA coefficients and one innovation variance parameter; no seasonal terms or deterministic trend.')
p('ARIMAX retains its selected training-only fit; observed-data filtering carries information forward causally without estimating parameters on the evaluation period. The full candidate results and optimiser warnings are retained in each arimax folder. An optimiser convergence flag is required but does not prove global optimality.')
p('GARCH has no neural selected epoch; its three parameter estimates appear in the preceding section. Last price has no fitted parameters or training epoch. Forecast duration and neural training duration are recorded in COMPLETE.json files and are execution logs rather than controlled computational benchmarks.')
page('Interpretation and reproducibility')
p('The primary conclusions must be read together: recurrent directionality is assessed by BiLSTM versus LSTM128 and the near size-matched LSTM210; Bitcoin is assessed by removing only BTC_Close; ARIMAX tests a different statistical specification. All price comparisons share an exact24hour target, but ARIMAX uses different predictor timing and a log-price target, so the comparison does not isolate architecture alone.')
p('Last-price scores establish whether greater model complexity improves this particular forecast task. A lower validation score does not guarantee a lower final evaluation error. The selected configuration is fixed by validation; all evaluation results, including any failure to improve, are retained. No attention or sentiment modelling is part of this experiment.')
p('The archived data and original source projects are unchanged. This study does not reclassify previously examined2026 observations as pristine unseen data. Truly prospective evidence would require freezing the chosen model and scoring newly arriving future observations. No such prospective run is claimed here.')
p('Reproduction: code/prepare.py creates immutable input copies and the split protocol once; code/check_setup.py checks causal alignment and network shapes; code/pipeline.py executes fits sequentially; code/summarise.py recomputes metrics and verifies scalers, target alignment and checkpoint hashes. code/build_report.py creates this Word report after numerical verification. See README.md for exact runtime commands. The pipeline refuses silent retraining of incomplete runs.')
h('References')
for t in ['Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation9(8),1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735','Schuster, M., and Paliwal, K. K. (1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing45(11),2673–2681. https://doi.org/10.1109/78.650093','Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics31(3),307–327. https://doi.org/10.1016/0304-4076(86)90063-1','statsmodels. SARIMAX documentation. https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html','arch. Volatility forecasting documentation. https://arch.readthedocs.io/en/latest/univariate/forecasting.html']:p(t)
import re
for par in d.paragraphs:
 for run in par.runs:
  if run.text:run.text=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])', ' ', run.text)
d.save(O/'CODEX_XRP_CONTROLLED_EXPERIMENT_REPORT.docx');print(O/'CODEX_XRP_CONTROLLED_EXPERIMENT_REPORT.docx')
