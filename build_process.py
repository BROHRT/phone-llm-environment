"""Bound a Docker build command while preserving logs and free-space diagnostics."""
import json,os,pathlib,shutil,signal,subprocess,time,types

def stop_daemon():
    # Only the disposable Actions build host. Killing the Docker client alone
    # does not establish that its daemon-side build has stopped.
    q=subprocess.run(['sudo','-n','systemctl','stop','--no-block','docker.service','docker.socket'],capture_output=True,text=True,timeout=15)
    return {'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'stop_requested':q.returncode==0}

def run(args,prefix,timeout,storage_path,reserve,free_fn=None,stop_fn=None):
    prefix=pathlib.Path(prefix);free_fn=free_fn or (lambda:shutil.disk_usage(storage_path).free);stop_fn=stop_fn or stop_daemon
    stdout=prefix.with_suffix('.stdout.log');stderr=prefix.with_suffix('.stderr.log');started=time.monotonic();failure=None;stop_result=None
    with stdout.open('xb') as so,stderr.open('xb') as se:
        p=subprocess.Popen(args,stdout=so,stderr=se,start_new_session=True)
        try:
            while p.poll() is None:
                free=free_fn()
                if free<=reserve or time.monotonic()-started>=timeout:
                    failure='disk_reserve_reached' if free<=reserve else 'command_timeout'
                    record={'classification':failure,'free_bytes':free,'reserve_bytes':reserve,'elapsed_seconds':time.monotonic()-started,'argv':args}
                    prefix.with_suffix('.abort.json').write_text(json.dumps(record,indent=2)+'\n')
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
                    try:stop_result=stop_fn()
                    except Exception as e:stop_result={'stop_requested':False,'error_type':type(e).__name__,'error':str(e)}
                    record['daemon_stop']=stop_result;prefix.with_suffix('.abort.json').write_text(json.dumps(record,indent=2)+'\n')
                    break
                time.sleep(.5)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
    # Full streams are retained separately, including timeout/abort output.
    def text(path):
        with path.open('rb') as f:return f.read(2*1024**2).decode('utf-8',errors='replace')
    return types.SimpleNamespace(returncode=p.returncode if not failure else (p.returncode or 1),stdout=text(stdout),stderr=text(stderr),failure_classification=failure,stdout_file=str(stdout),stderr_file=str(stderr),stdout_bytes=stdout.stat().st_size,stderr_bytes=stderr.stat().st_size)
