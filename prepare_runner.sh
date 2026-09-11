#!/usr/bin/env bash
# Disposable GitHub-hosted CPU runner only. No experiment or account data.
set -euo pipefail
mkdir -p evidence
test "$(uname -m)" = x86_64
test "$(uname -s)" = Linux
df -B1 / /mnt > evidence/disk-before.txt
df -h > evidence/all-mounts-human.txt
df -B1 > evidence/all-mounts-bytes.txt
findmnt > evidence/all-mounts-findmnt.txt
findmnt -T /mnt > evidence/mnt-mount.txt
lsblk -b > evidence/block-devices.txt
/usr/bin/python3 - <<'PYBEFORE'
import json,pathlib,shutil,time
pathlib.Path('evidence/reclaim-before.json').write_text(json.dumps({'issued_unix':time.time(),**shutil.disk_usage('/')._asdict()})+'\n')
PYBEFORE
sudo du -x -B1 -d 2 /usr/local /usr/share /opt /var/lib/docker > evidence/toolchain-disk-before.txt 2>&1 || true
# Only unused toolchains on this disposable runner; retain Docker, Python and gh.
sudo rm -rf /usr/local/lib/android /usr/share/dotnet /usr/local/.ghcup /opt/hostedtoolcache /usr/local/share/powershell /usr/local/swift /usr/share/swift /opt/ghc /usr/local/share/boost /usr/local/share/chromium /opt/az /usr/share/miniconda
docker image prune --all --force
df -B1 / /mnt > evidence/disk-after-cleanup.txt
df -B1 > evidence/all-mounts-after-cleanup.txt
# One measured root filesystem. No capacity is attributed to directory moves.
BUILD_ROOT=/var/lib/phone-llm-build
/usr/bin/python3 - <<'PYADMIT'
import json,pathlib,shutil,time,hashlib
policy=json.loads(pathlib.Path('storage-calculation.json').read_text())
assert policy['wheel_metadata_sha256']==hashlib.sha256(pathlib.Path('wheel-storage-metadata.json').read_bytes()).hexdigest()
assert sum(x['bytes'] for x in policy['items'])==policy['sum_bytes']
assert policy['minimum_free_bytes']==((policy['sum_bytes']+1024**3-1)//1024**3)*1024**3
before=json.loads(pathlib.Path('evidence/reclaim-before.json').read_text());now=shutil.disk_usage('/')
record={'issued_unix':time.time(),'filesystem':'/','before':before,'after':now._asdict(),'recovered_free_bytes':now.free-before['free'],'minimum_free_bytes':policy['minimum_free_bytes'],'admission_is_estimate_not_measured_peak':True}
pathlib.Path('evidence/storage-admission.json').write_text(json.dumps(record,indent=2)+'\n')
pathlib.Path('evidence/storage-calculation.json').write_text(json.dumps(policy,indent=2)+'\n')
if now.free<policy['minimum_free_bytes']:raise SystemExit('Calculated build footprint exceeds measured capacity; do not build')
PYADMIT
sudo mkdir -p "$BUILD_ROOT/docker" "$BUILD_ROOT/tmp" "$BUILD_ROOT/context"
sudo chown "$USER:$(id -gn)" "$BUILD_ROOT" "$BUILD_ROOT/tmp" "$BUILD_ROOT/context"
sudo systemctl stop docker.service docker.socket
sudo /usr/bin/python3 - "$BUILD_ROOT" <<'PY'
import json,pathlib,sys
root=sys.argv[1];p=pathlib.Path('/etc/docker/daemon.json')
config=json.loads(p.read_text()) if p.exists() else {}
config['data-root']=root+'/docker'
# Use the classic Docker image store, so image layers do not remain under
# containerd's default root on another filesystem.
config.setdefault('features',{})['containerd-snapshotter']=False
p.write_text(json.dumps(config,indent=2)+'\n')
d=pathlib.Path('/etc/systemd/system/docker.service.d');d.mkdir(parents=True,exist_ok=True)
(d/'phone-llm-tmp.conf').write_text('[Service]\nEnvironment="DOCKER_TMPDIR='+root+'/tmp" "TMPDIR='+root+'/tmp"\n')
PY
sudo systemctl daemon-reload
sudo systemctl start docker.service
docker info --format '{{json .}}' > evidence/docker-info.json
/usr/bin/python3 - "$BUILD_ROOT" <<'PY'
import json,pathlib,platform,shutil,sys,time
root=pathlib.Path(sys.argv[1]);info=json.loads(pathlib.Path('evidence/docker-info.json').read_text())
assert pathlib.Path(info['DockerRootDir']).resolve()==(root/'docker').resolve()
assert not any('containerd.snapshotter' in str(row) for row in info.get('DriverStatus',[]))
free=shutil.disk_usage(root/'docker').free
record={'issued_unix':time.time(),'system':platform.system(),'architecture':platform.machine(),'build_root':str(root),'docker_root':info['DockerRootDir'],'free_bytes':free,'minimum_free_bytes':json.loads(pathlib.Path('storage-calculation.json').read_text())['minimum_free_bytes'],'gpu_requested':False,'disk_requirement_is_preflight_not_measured_peak':True}
pathlib.Path('evidence/runner.json').write_text(json.dumps(record,indent=2)+'\n')
if free<record['minimum_free_bytes']:raise SystemExit('Actual Docker filesystem falls below calculated storage admission')
PY
findmnt -T "$BUILD_ROOT/docker" > evidence/docker-root-mount.txt
df -B1 "$BUILD_ROOT/docker" "$BUILD_ROOT/context" "$BUILD_ROOT/tmp" > evidence/disk-after.txt
cp build_image.py build_process.py storage-calculation.json "$BUILD_ROOT/context/"
cp -R image "$BUILD_ROOT/context/"
printf 'BUILD_ROOT=%s\nTMPDIR=%s/tmp\n' "$BUILD_ROOT" "$BUILD_ROOT" >> "$GITHUB_ENV"
