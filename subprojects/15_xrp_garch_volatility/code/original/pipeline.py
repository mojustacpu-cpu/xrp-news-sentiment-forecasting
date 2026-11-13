from common import *
import sys,time,traceback,subprocess,fcntl
import neural,statistical
lock=open(B/'runs/pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
protocol=json.loads((B/'configs/protocol.json').read_text());protocol_hash=sha(B/'configs/protocol.json');started=str(pd.Timestamp.now(tz='UTC'))
def status(phase,**kw):
 assert sha(B/'configs/protocol.json')==protocol_hash,'Frozen protocol changed'
 save(B/'LATEST_RUN.json',dict(status=phase,pid=os.getpid(),started_utc=started,command='code/pipeline.py',protocol_sha256=protocol_hash,**kw))
try:
 for fold in ['main','rolling_1','rolling_2']:
  configs=protocol['neural'] if fold=='main' else [c for c in protocol['neural'] if c['name'] in protocol['rolling_models']]
  seeds=protocol['main_seeds'] if fold=='main' else protocol['rolling_seeds']
  for c in configs:
   for seed in seeds:
    status('training',fold=fold,model=c['name'],seed=seed,progress_file=f"runs/{fold}/{c['name']}_seed{seed}/PROGRESS.json")
    neural.train(fold,c,seed,protocol)
  status('statistical_fitting',fold=fold);statistical.train(fold)
  checks={str(p.relative_to(B)):sha(p) for p in (B/'runs'/fold).glob('*/best.pt')}
  checks.update({str(p.relative_to(B)):sha(p) for p in (B/'runs'/fold).glob('*/COMPLETE.json')})
  save(B/'runs'/fold/'SELECTION_LOCK.json',dict(protocol_sha256=protocol_hash,checkpoints_and_selections=checks,locked_before_evaluation=True,utc=str(pd.Timestamp.now(tz='UTC'))))
  status('evaluating_frozen_models',fold=fold)
  for c in configs:
   for seed in seeds:neural.evaluate(fold,c,seed)
  statistical.evaluate(fold)
 status('summarising');subprocess.run([sys.executable,str(B/'code/summarise.py')],check=True)
 status('analysis_complete_report_pending');print('ALL EXPERIMENTS COMPLETE; REPORT PENDING',flush=True)
except BaseException as e:
 status('failed',error=repr(e),checkpoints_preserved=True);traceback.print_exc();raise
