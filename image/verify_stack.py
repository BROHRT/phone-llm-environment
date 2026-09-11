"""CPU-only in-image verification. No model retrieval and no GPU checks."""
import hashlib,importlib,importlib.metadata as metadata,json,pathlib,platform,subprocess,sys,time
ROOT=pathlib.Path('/opt/phone-llm');normalize=lambda x:x.lower().replace('_','-').replace('.','-')
def require(value,message):
 if not value:raise RuntimeError(message)
def verify(*,build_checks=True):
 expected=json.loads((ROOT/'expected.json').read_text())
 require(platform.system()=='Linux' and platform.machine()=='x86_64','Linux amd64 image required')
 require(platform.python_version()==expected['python'],'exact Python pin mismatch')
 require(pathlib.Path(sys.prefix)==ROOT/'venv','preinstalled environment not selected')
 require(hashlib.sha256((ROOT/'source-install-report.json').read_bytes()).hexdigest()==expected['source_install_report_sha256'],'source inventory drift')
 require(hashlib.sha256(pathlib.Path('/start.sh').read_bytes()).hexdigest()==expected['base_start_sha256'],'proven startup script changed')
 source=json.loads((ROOT/'source-install-report.json').read_text());built=json.loads((ROOT/'pip-install-report.json').read_text())
 wheel=lambda row:row['download_info']['archive_info']['hashes']['sha256']
 source_wheels={normalize(x['metadata']['name']):(x['metadata']['version'],wheel(x)) for x in source['install']}
 built_wheels={normalize(x['metadata']['name']):(x['metadata']['version'],wheel(x)) for x in built['install']}
 require(len(source_wheels)==55 and source_wheels==built_wheels,'build wheel set differs from successful A9 install')
 installed={name:metadata.version(name) for name in expected['all_packages']}
 require(installed==expected['all_packages'],'installed distribution pin mismatch')
 imports={}
 for name,version in expected['pins'].items():
  module=importlib.import_module(name);imports[name]=getattr(module,'__version__',None)
  require(metadata.version(name)==version,'top-level package pin mismatch')
 import torch
 # No CUDA availability or version admission check on this CPU build runner.
 # Tiny CPU operation validates compiled torch extension import; no model or CUDA.
 require((torch.ones(2,device='cpu')+1).tolist()==[2.,2.],'CPU tensor smoke failed')
 pip_result='NOT_RUN_RUNTIME_VERIFY_ONLY'
 if build_checks:
  check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=120)
  require(check.returncode==0,'pip check failed: '+check.stdout+check.stderr)
  pip_result=check.stdout.strip()
 ssh=subprocess.run(['/usr/sbin/sshd','-V'],capture_output=True,text=True,timeout=10)
 require(ssh.returncode==0 and 'OpenSSH' in ssh.stderr+ssh.stdout,'preinstalled sshd missing')
 return {'status':'PASS','issued_unix':time.time(),'python':platform.python_version(),'architecture':platform.machine(),'installed':installed,'imported_versions':imports,'torch_cuda_build':torch.version.cuda,'pip_check':pip_result,'runtime_verify_only':not build_checks,'sshd_version':(ssh.stderr+ssh.stdout).strip(),'base_start_sha256':expected['base_start_sha256'],'source_install_report_sha256':expected['source_install_report_sha256'],'built_install_report_sha256':hashlib.sha256((ROOT/'pip-install-report.json').read_bytes()).hexdigest(),'model_weights_loaded':False,'gpu_checks_executed':False,'cpu_tensor_check':True,'heldout_content_read':False,'pip_install_during_verification':False,'venv_creation_during_verification':False}
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path);ap.add_argument('--runtime',action='store_true');args=ap.parse_args();record=verify(build_checks=not args.runtime)
 if args.output:
  with args.output.open('x') as file:json.dump(record,file,sort_keys=True);file.write('\n')
 print(json.dumps(record,sort_keys=True))
