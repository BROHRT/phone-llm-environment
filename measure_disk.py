"""Sample actual build filesystem occupancy; do not call it a continuous peak."""
import argparse,json,pathlib,shutil,signal,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--path',required=True,type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path);p.add_argument('--interval',type=float,default=1);a=p.parse_args()
    if a.interval<=0:raise ValueError('positive interval required')
    a.out.mkdir(parents=True,exist_ok=True);running=True;samples=[]
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (a.out/'disk-samples.ndjson').open('x') as f:
        def sample():
            row={'issued_unix':time.time(),**shutil.disk_usage(a.path)._asdict()};samples.append(row);f.write(json.dumps(row)+'\n');f.flush()
        while running:
            sample();time.sleep(a.interval)
        sample()
    result={'path':str(a.path.resolve()),'sample_interval_seconds':a.interval,'sample_count':len(samples),'initial_used_bytes':samples[0]['used'],'maximum_observed_used_bytes':max(x['used'] for x in samples),'minimum_observed_free_bytes':min(x['free'] for x in samples),'maximum_observed_increase_bytes':max(x['used'] for x in samples)-samples[0]['used'],'observed_filesystem_peak_not_continuous_peak':True,'includes_other_process_disk_use':True,'first_unix':samples[0]['issued_unix'],'last_unix':samples[-1]['issued_unix']}
    (a.out/'disk-peak.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
