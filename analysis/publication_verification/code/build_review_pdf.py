from pathlib import Path
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
B=Path(__file__).resolve().parents[2];out=B/'FINAL_SCIENTIFIC_AUDIT_REPORT.pdf';styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyAudit',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=6));styles.add(ParagraphStyle(name='CellAudit',fontName='Helvetica',fontSize=8,leading=10,wordWrap='CJK'));styles.add(ParagraphStyle(name='HeadCellAudit',fontName='Helvetica-Bold',fontSize=8,leading=10,textColor=colors.white));styles['Heading1'].fontSize=18;styles['Heading1'].leading=22;styles['Heading2'].fontSize=12;styles['Heading2'].leading=15
W=A4[0]-90;story=[]
def fmt(t):return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',html.escape(t))
def para(t,style='BodyAudit'):return Paragraph(fmt(t),styles[style])
def render_md(path):
 lines=path.read_text().splitlines();i=0;buf=[]
 def flush():
  if buf:story.append(para(' '.join(buf)));buf.clear()
 while i<len(lines):
  l=lines[i].strip()
  if not l:flush();i+=1;continue
  if l.startswith('|'):
   flush();rows=[]
   while i<len(lines) and lines[i].strip().startswith('|'):
    cells=[v.strip() for v in lines[i].strip().strip('|').split('|')]
    if not all(re.fullmatch(r'[:\- ]+',v or '-') for v in cells):rows.append(cells)
    i+=1
   n=len(rows[0]);fracs=([.06,.08,.17,.20,.49] if n==5 else [.12,.16,.30,.42] if n==4 else [.10,.16,.15,.14,.14,.14,.17] if n==7 else [.10,.20,.14,.23,.23,.10] if n==6 else [1/n]*n);total=sum(fracs);cw=[W*f/total for f in fracs];data=[[para(c,'HeadCellAudit' if r==0 else 'CellAudit') for c in row] for r,row in enumerate(rows)];t=Table(data,colWidths=cw,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#285674')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#c6cdd2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f7')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);continue
  if l.startswith('# '):flush();story.extend([para(l[2:],'Heading1'),Spacer(1,6)])
  elif l.startswith('## '):flush();story.append(para(l[3:],'Heading2'))
  elif l.startswith('- '):flush();story.append(para('• '+l[2:]))
  elif re.match(r'^\d+\.',l):flush();story.append(para(l))
  else:buf.append(l)
  i+=1
 flush()
for j,name in enumerate(['SCIENTIFIC_REVIEW_REPORT.md','NUMERICAL_VERIFICATION_REPORT.md','REPRODUCIBILITY_CHECKLIST.md','CHANGE_LOG.md','OUTSTANDING_WORK.md']):
 if j:story.append(PageBreak())
 render_md(B/name)
 if name=='NUMERICAL_VERIFICATION_REPORT.md':
  story.extend([Spacer(1,10),para('Supporting numerical diagnostic','Heading2')]);story.append(Image(str(B/'figures/audit_fig01_bootstrap_precision.png'),width=W,height=W*4.2/10));story.append(para('Descriptive serial dependence and simulation uncertainty of selected adjusted lower endpoints. Endpoint brackets describe Monte Carlo precision, not an independent sample of market outcomes.'))
def page(c,d):
 c.setStrokeColor(colors.HexColor('#c6cdd2'));c.line(45,35,A4[0]-45,35);c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#53616a'));c.drawString(45,24,'XRP scientific audit | 8 November 2026 | No model retraining');c.drawRightString(A4[0]-45,24,str(d.page))
SimpleDocTemplate(str(out),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=48,title='XRP Final Scientific Review and Numerical Audit',author='Critical assistant review for Mahmoud Ali and co-authors').build(story,onFirstPage=page,onLaterPages=page)
print(out)
