from pathlib import Path
import sys,os,time,json,subprocess,traceback,hashlib
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'reference'))
from common import save,sha
legacy=P.parents[1]/'subprojects/xrp-hybrid-news-feature-comparison-oct-2026'
def main():
 with (P/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),started=time.time()),f)
 identity=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','lstart=,command='],text=True).strip()
 while True:
  status=json.loads((legacy/'LATEST_RUN.json').read_text())
  if (legacy/'reports/FINAL_NUMERICAL_VERIFICATION.json').exists():
   break
  if 'fail' in status.get('status','').lower():raise RuntimeError('Legacy pipeline failure; Phase1 not started')
  active=subprocess.run(['ps','-p',str(status.get('pid',0)),'-o','command='],capture_output=True,text=True).stdout
  if 'pipeline.py' not in active:raise RuntimeError('Legacy process absent without verified completion; Phase1 not started')
  save(P/'STATUS.json',dict(status='waiting_for_legacy',pid=os.getpid(),identity=identity,legacy=status))
  time.sleep(30)
 for j,name in enumerate(['E1B','E2_h3','E2_h6','E2_h12','E2_h24']):
  save(P/'STATUS.json',dict(status='training',pid=os.getpid(),identity=identity,candidate=name,completed_candidates=j,total_candidates=5,progress_file=str(P/name/'runs/STATUS.json')))
  with (P/name/'execution.log').open('x') as log:
   subprocess.run([sys.executable,'-u',str(P/'train_candidate.py'),name],stdout=log,stderr=subprocess.STDOUT,check=True)
 save(P/'STATUS.json',dict(status='25_fits_complete_reports_pending',pid=os.getpid(),completed_fits=25,evaluation_performed=False,E3_E4_started=False))
if __name__=='__main__':
 try:main()
 except Exception:
  save(P/'FAILURE.json',dict(error=traceback.format_exc(),pid=os.getpid()))
  save(P/'STATUS.json',dict(status='failed',error=traceback.format_exc()))
  raise
