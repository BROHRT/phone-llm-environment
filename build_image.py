"""CPU-only image build/verification. Optional push requires an explicit repository.
No Runpod calls, model weights, experiment payload, approval or launch capability.
"""
import argparse,hashlib,json,pathlib,re,shutil,subprocess,time,uuid
import build_process
P=pathlib.Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--publish',action='store_true');a=ap.parse_args()
 if not re.fullmatch(r'[a-z0-9][a-z0-9./_-]*:[A-Za-z0-9_.-]+',a.tag) or '@' in a.tag:raise RuntimeError('explicit credential-free repository:tag required')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);docker=shutil.which('docker')
 if docker is None:
  (out/'BLOCKED.json').write_text(json.dumps({'status':'BLOCKED','reason':'No Docker runtime; image not built, executed, verified or published','paid_execution':False},indent=2)+'\n');raise RuntimeError('Docker CPU builder required')
 counter=0
 def run(args,timeout=120,check=True):
  nonlocal counter
  counter+=1;started=time.time();policy=json.loads((P/'storage-calculation.json').read_text());q=build_process.run([docker,*args],out/('%02d-stream'%counter),timeout,P,policy['stop_free_bytes'])
  (out/('%02d-command.json'%counter)).write_text(json.dumps({'argv':[docker,*args],'started_unix':started,'finished_unix':time.time(),'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'failure_classification':q.failure_classification,'stdout_file':q.stdout_file,'stderr_file':q.stderr_file,'stdout_bytes':q.stdout_bytes,'stderr_bytes':q.stderr_bytes},indent=2)+'\n')
  if check and q.returncode:raise RuntimeError('Docker command failed; preserved command'+str(counter))
  return q
 info=json.loads(run(['info','--format','{{json .}}']).stdout)
 if info.get('OSType')!='linux' or info.get('Architecture') not in ('x86_64','amd64'):raise RuntimeError('native Linux amd64 CPU builder required; no emulated wheel-install experiment')
 expected=json.loads((P/'image/expected.json').read_text());base=expected['base_image']
 run(['pull','--platform','linux/amd64',base],1800)
 run(['build','--platform','linux/amd64','--network=default','--progress=plain','--tag',a.tag,str(P/'image')],10800)
 def inspect(ref):return json.loads(run(['image','inspect',ref]).stdout)[0]
 run(['builder','prune','--all','--force'],120)
 before=inspect(base);built=inspect(a.tag)
 for key in ('Entrypoint','Cmd','WorkingDir','Env','User'):
  if before['Config'].get(key)!=built['Config'].get(key):raise RuntimeError('inherited base configuration changed: '+key)
 def cpu_verify(ref,label):
  q=run(['run','--rm','--network','none','--entrypoint','/opt/phone-llm/venv/bin/python',ref,'-I','/opt/phone-llm/verify_stack.py'],300)
  v=json.loads(q.stdout);assert v['status']=='PASS' and not v['gpu_checks_executed'] and not v['model_weights_loaded'];(out/(label+'.json')).write_text(json.dumps(v,indent=2)+'\n');return v
 inventory=cpu_verify(a.tag,'local-in-image-verification')
 # Separate actual sshd listener smoke in an isolated CPU container; no published
 # host port and no claim about provider endpoint publication.
 name='phone-llm-image-check-'+uuid.uuid4().hex
 try:
  run(['run','-d','--name',name,'--network','none','--entrypoint','/bin/bash',a.tag,'-euc','mkdir -p /run/sshd; ssh-keygen -A; exec /usr/sbin/sshd -D -e -p 2222'])
  code="import socket,time;end=time.monotonic()+30\nwhile True:\n try:\n  s=socket.create_connection(('127.0.0.1',2222),timeout=2);print(s.recv(255).decode().strip());s.close();break\n except OSError:\n  if time.monotonic()>=end:raise\n  time.sleep(.2)"
  q=run(['exec',name,'/opt/phone-llm/venv/bin/python','-c',code],45);assert q.stdout.startswith('SSH-2.0-OpenSSH')
  (out/'container-listener.json').write_text(json.dumps({'status':'PASS','banner':q.stdout.strip(),'in_container_listener_observed':True,'provider_endpoint_checked':False,'dedicated_cpu_test_container':True},indent=2)+'\n')
 finally:
  run(['logs',name],check=False);removed=run(['rm','--force',name],check=False)
  if removed.returncode:raise RuntimeError('CPU verification container cleanup unconfirmed: '+name)
 receipt={'status':'LOCAL_IMAGE_VERIFIED','tag':a.tag,'image_config_id':built['Id'],'config_id_is_not_registry_manifest_digest':True,'base_digest':base,'cpu_only':True,'provider_calls':0,'model_weights_loaded':False,'heldout_content_read':False,'build_context_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((P/'image').iterdir()) if p.is_file()},'issued_unix':time.time()}
 if a.publish:
  if '/' not in a.tag:raise RuntimeError('publication requires explicit registry/repository')
  run(['push',a.tag],3600);digests=inspect(a.tag).get('RepoDigests',[]);repository=a.tag.rsplit(':',1)[0];matches=[x for x in digests if x.startswith(repository+'@sha256:')]
  if len(matches)!=1:raise RuntimeError('unique pushed repository digest not resolved')
  digest=matches[0];run(['pull','--platform','linux/amd64',digest],1800);post=cpu_verify(digest,'digest-pulled-in-image-verification')
  if post['installed']!=inventory['installed'] or inspect(digest)['Id']!=built['Id']:raise RuntimeError('published digest differs from tested image')
  receipt.update(status='DIGEST_PINNED_IMAGE_VERIFIED',image=digest,registry_manifest_digest=digest.split('@')[1],digest_pull_reverified=True)
 (out/'IMAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
