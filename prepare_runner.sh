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
sudo du -x -B1 -d 2 /usr/local /usr/share /opt /var/lib/docker > evidence/toolchain-disk-before.txt 2>&1 || true
# Only unused toolchains on this disposable runner; retain Docker, Python and gh.
sudo rm -rf /usr/local/lib/android /usr/share/dotnet /usr/local/.ghcup /opt/hostedtoolcache /usr/local/share/powershell /usr/local/swift /usr/share/swift /opt/ghc /usr/local/share/boost /usr/local/share/chromium /opt/az /usr/share/miniconda
docker image prune --all --force
df -B1 / /mnt > evidence/disk-after-cleanup.txt
df -B1 > evidence/all-mounts-after-cleanup.txt
# Prefer /mnt, but do not mistake a small separate mount for extra root capacity.
BUILD_VOLUME=$(/usr/bin/python3 - <<'PY'
import json,pathlib,shutil,time
minimum=55*1024**3
volumes={p:shutil.disk_usage(p)._asdict() for p in ('/mnt','/')}
selected=next((p for p in ('/mnt','/') if volumes[p]['free']>=minimum),None)
pathlib.Path('evidence/volume-selection.json').write_text(json.dumps({'issued_unix':time.time(),'volumes':volumes,'selected':selected,'minimum_free_bytes':minimum,'disk_requirement_is_preflight_not_measured_peak':True},indent=2)+'\n')
if selected is None:raise SystemExit('Insufficient disk on both /mnt and /: require 55 GiB; threshold unchanged')
print(selected)
PY
)
if test "$BUILD_VOLUME" = /mnt; then BUILD_ROOT=/mnt/phone-llm-build; else BUILD_ROOT=/var/lib/phone-llm-build; fi
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
record={'issued_unix':time.time(),'system':platform.system(),'architecture':platform.machine(),'build_root':str(root),'docker_root':info['DockerRootDir'],'free_bytes':free,'minimum_free_bytes':55*1024**3,'gpu_requested':False,'disk_requirement_is_preflight_not_measured_peak':True}
pathlib.Path('evidence/runner.json').write_text(json.dumps(record,indent=2)+'\n')
if free<record['minimum_free_bytes']:raise SystemExit('Docker filesystem requires 55 GiB before pull/build')
PY
findmnt -T "$BUILD_ROOT/docker" > evidence/docker-root-mount.txt
df -B1 "$BUILD_ROOT/docker" "$BUILD_ROOT/context" "$BUILD_ROOT/tmp" > evidence/disk-after.txt
cp build_image.py "$BUILD_ROOT/context/"
cp -R image "$BUILD_ROOT/context/"
printf 'BUILD_ROOT=%s\nTMPDIR=%s/tmp\n' "$BUILD_ROOT" "$BUILD_ROOT" >> "$GITHUB_ENV"
