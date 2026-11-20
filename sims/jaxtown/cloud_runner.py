"""Train on Cloud TPU machines (e.g. a TPU Research Cloud grant) from this PC, and keep
going when a machine is taken back (spot) or stops.

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/cloud_runner.py \\
        --name towns --zone us-central2-b --type v4-8 --spot --machines 2 \\
        --bucket gs://<bucket> -- --runs <8 runs> --river hughes --steps 1e8 \\
        --out-root out/cloud/towns

USE EXACTLY THE ZONE, TYPE AND KIND (--spot or --on-demand) FROM THE TRC EMAIL: Cloud
TPUs are free only within the granted quota, and anything else is billed. So none of
the three has a default.

Everything after `--` goes to train_jax.py and must include --out-root, a folder on this
PC. Needs `gcloud` (logged in, a project set), a Cloud Storage bucket in the machines'
region, and this repo's current commit pushed to GitHub: each machine clones the public
repo at that commit.

How it works:
- the runs are dealt out over --machines machines (<name>-0, <name>-1, ...); on each,
  train_jax.py spreads its runs over the machine's chips;
- the bucket holds each run's copy (folder gs://<bucket>/<out-root's name>/): each
  machine puts its runs there every --vm-sync-minutes (sims/jaxtown/cloud_vm.py), so a
  machine that's taken back is replaced and the new one carries on from there;
- at the start, a run whose copy here is further along than the bucket's goes up first
  (so runs started on the GPU here can carry on in the cloud);
- every --sync-minutes, this PC checks each machine (replacing it if it's gone, relaunching
  training if it stopped) and copies the small files home: config.json, metrics.csv, the
  new lines of episodes.jsonl, each machine's train log (in _machines/). Checkpoints come
  home when a machine's runs finish (each copy out of Google Cloud is charged; the
  bucket keeps them safe meanwhile);
- a machine whose runs have all finished is deleted at once; when all have, it's done.
  Every machine is deleted on the way out, Ctrl-C included (that first asks each one to
  save and put its copies in the bucket); run the same command again to carry on.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cloud_vm  # noqa: E402
import colab_runner as cr  # noqa: E402  (shared: clean, remote_argv, pushed_commit, signals)

train_jax, town_train = cr.train_jax, cr.town_train
Store = cloud_vm.Store

# Software image for each TPU kind (cloud.google.com/tpu/docs/runtimes, read 2026-11-19).
VERSIONS = {"v4": "tpu-ubuntu2204-base", "v5litepod": "v2-alpha-tpuv5-lite",
            "v5p": "v2-alpha-tpuv5", "v6e": "v2-alpha-tpuv6e"}
# Machine states (Cloud TPU API Node.State) that mean: still on its way, wait.
WAITING = ("CREATING", "STARTING", "RESTARTING", "REPAIRING", "UNKNOWN")
UV = "0.13.8"   # the uv we use here, to make the machine's Python and venv
BACKOFF = (60, 120, 300, 600, 1200, 1800)   # waits between tries for a machine


class MachineLost(Exception):
    pass


def tpu_version(accel):
    kind, _, size = accel.partition("-")
    if kind not in VERSIONS or not size.isdigit():
        raise SystemExit(f"cloud_runner: don't know TPU type {accel!r} (kinds: "
                         f"{', '.join(VERSIONS)}; e.g. v4-8, v5litepod-8, v6e-8)")
    if int(size) > 8:
        raise SystemExit(f"cloud_runner: {accel} spans several hosts; use several single-host "
                         "machines instead (--machines, with a -8 or smaller type)")
    return VERSIONS[kind]


class TPUVMs:
    """`gcloud compute tpus tpu-vm`: start, look at, run commands on and delete machines."""

    def __init__(self, zone, accel, spot, project=None, version=None):
        self.zone, self.accel, self.spot = zone, accel, spot
        self.version = version or tpu_version(accel)
        self.where = ["--zone", zone] + (["--project", project] if project else [])

    def _g(self, args, timeout):
        try:
            p = subprocess.run(["gcloud", "compute", "tpus", "tpu-vm", *args, *self.where],
                               capture_output=True, text=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"gcloud tpu-vm {args[0]}: no answer in {timeout}s"

    def create_args(self, name):
        return ["create", name, "--accelerator-type", self.accel, "--version", self.version,
                *(["--spot"] if self.spot else []), "--quiet"]

    def create(self, name):
        return self._g(self.create_args(name), 1800)

    def state(self, name):
        """READY, CREATING, PREEMPTED, ... or None if there's no such machine."""
        rc, out, err = self._g(["describe", name, "--format", "value(state)"], 300)
        if rc == 0:
            return out.strip() or "UNKNOWN"
        return None if ("NOT_FOUND" in err or "not found" in err.lower()) else "UNKNOWN"

    def delete(self, name):
        return self._g(["delete", name, "--quiet"], 1800)

    def ssh(self, name, command, timeout=600):
        return self._g(["ssh", name, "--command", command, "--strict-host-key-checking=no",
                        "--quiet"], timeout)


def same_region(location, zone):
    """Is a bucket at `location` (e.g. US-CENTRAL2) in zone's region (us-central2-b)?"""
    return location.strip().lower() == zone.rsplit("-", 1)[0]


def check_bucket(bucket, zone):
    """The bucket must be in the machines' region: copies between regions are charged."""
    p = subprocess.run(["gcloud", "storage", "buckets", "describe", bucket, "--format",
                        "value(location)"], capture_output=True, text=True, timeout=300)
    if p.returncode:
        raise SystemExit(f"cloud_runner: can't see bucket {bucket}: {cr.clean(p.stderr, 300)}")
    if not same_region(p.stdout, zone):
        region = zone.rsplit("-", 1)[0]
        raise SystemExit(f"cloud_runner: bucket {bucket} is in {p.stdout.strip()}, the "
                         f"machines in {region}: make the bucket in {region} (copies between "
                         "regions are charged), or pass --any-region")


def without_runs(argv):
    """train_jax arguments without --runs and its run specs."""
    out, skipping = [], False
    for a in argv:
        if skipping and not a.startswith("-"):
            continue
        skipping = False
        if a == "--runs":
            skipping = True
            continue
        if not a.startswith("--runs="):
            out.append(a)
    return out


def spec_text(spec):
    return ":".join(str(x) for x in spec)


class Slot:
    """One machine of the fleet and the runs it trains."""

    def __init__(self, index, name, specs, runs, argv):
        self.index, self.name, self.specs, self.runs, self.argv = index, name, specs, runs, argv
        self.alive = self.done = False
        self.machines = self.crashes = self.fails = self.start_failures = 0
        self.retry_at = self.launched_at = 0.0
        self.launch = None


class Runner:
    def __init__(self, vms, store, train_argv, name, machines=1, sync_minutes=10.0,
                 vm_sync_minutes=5.0, max_creates=100, max_crashes=3, commit=None, setup=True,
                 remote_python="venv/bin/python", log=None, stop_wait_s=900):
        self.vms, self.store = vms, store
        self.sync_s, self.vm_sync = 60 * sync_minutes, vm_sync_minutes
        self.max_creates, self.max_crashes = max_creates, max_crashes
        self.commit, self.do_setup, self.py = commit, setup, remote_python
        self.stop_wait_s = stop_wait_s
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", name):
            raise SystemExit("cloud_runner: --name: lowercase letters, digits and -, "
                             "starting with a letter")
        self.args = train_jax.parse(train_argv)
        if self.args.out:
            raise SystemExit("cloud_runner: use --out-root (a folder of runs), not --out")
        self.local = self.args.out_root
        self.root = os.path.basename(os.path.normpath(self.local))
        self.runs = dict(train_jax.run_identity(s, self.args) for s in self.args.specs)
        self.target = int(self.args.steps)
        machines = max(1, min(machines, len(self.args.specs)))
        base = without_runs(train_argv)
        self.slots = []
        for i in range(machines):
            specs = self.args.specs[i::machines]
            names = [train_jax.run_identity(s, self.args)[0] for s in specs]
            argv = cr.remote_argv(base + ["--runs", *map(spec_text, specs)], f"out/{self.root}")
            self.slots.append(Slot(i, f"{name}-{i}", specs, names, argv))
        self.creates = self.launches = 0
        os.makedirs(self.local, exist_ok=True)
        self.log_f = open(os.path.join(self.local, "runner.log"), "a")
        self.echo = log or print

    # --- bookkeeping ------------------------------------------------------------------

    def say(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        self.echo(line)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def run_dir(self, name):
        return os.path.join(self.local, name)

    def local_steps(self, name):
        path = os.path.join(self.run_dir(name), "ckpt.pkl")
        return cloud_vm.ckpt_steps(path) if os.path.exists(path) else (0, 0)

    def mirror_json(self, key):
        """A small file from the bucket, as last copied here."""
        try:
            with open(os.path.join(self.local, *key.split("/"))) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def progress(self, name):
        p = self.mirror_json(f"{name}/progress.json")
        return p["env_steps"] if p else 0

    # --- the start: what's here vs what's in the bucket -----------------------------------

    def send_ahead(self):
        """A run whose copy here is further along than the bucket's goes up (after the
        same folder check train_jax makes). Runs the bucket has further along are only
        mirrored here, so they aren't checked here: each machine checks its own."""
        args = argparse.Namespace(**{**vars(self.args), "resume": True})
        for name, settings in self.runs.items():
            there = (self.store.read_json(f"{name}/progress.json") or {}).get("env_steps", 0)
            here, update = self.local_steps(name)
            if there and there >= here:
                self.say(f"  {name}: carries on from the bucket's copy ({there:,} steps)")
                continue
            town_train.check_folder(self.run_dir(name), args, settings,
                                    train_jax.SAME_ON_RESUME)
            if not here:
                continue                       # new run: starts on its machine
            town_train.trim_logs(self.run_dir(name), update, name)
            d = self.run_dir(name)
            logs = [os.path.join(d, f) for f in cloud_vm.LOGS if os.path.exists(os.path.join(d, f))]
            if not (self.store.put_many(logs, name) and
                    self.store.put(os.path.join(d, "ckpt.pkl"), f"{name}/ckpt.pkl") and
                    self.store.write_json(f"{name}/progress.json",
                                          {"env_steps": here, "update": update,
                                           "time": time.time(), "machine": "this PC"})):
                raise SystemExit(f"cloud_runner: couldn't put {name} in the bucket")
            self.say(f"  {name}: sent this PC's copy up ({here:,} steps)")

    # --- copies home ------------------------------------------------------------------------

    def pull(self, checkpoints=()):
        """Copy the bucket's small files here, the new lines of each episodes.jsonl, and
        the checkpoints of the runs named in `checkpoints` (those first)."""
        for name in checkpoints:
            if self.store.get(f"{name}/ckpt.pkl", os.path.join(self.run_dir(name), "ckpt.pkl")):
                self.say(f"  {name}: checkpoint copied here")
        self.store.sync_down(self.local, skip=("ckpt.pkl", "episodes.jsonl"))
        sizes = self.store.sizes() or {}
        for name in self.runs:
            self.pull_episodes(name, sizes.get(f"{name}/episodes.jsonl"))

    def pull_episodes(self, name, size):
        """episodes.jsonl grows by whole lines, so copy only the new part, checking that
        the part before it is what's here (after a resume trimmed it, it may not be)."""
        local = os.path.join(self.run_dir(name), "episodes.jsonl")
        have = os.path.getsize(local) if os.path.exists(local) else 0
        if size is None:
            return
        if size == have:
            if not os.path.exists(local):         # no game finished yet: an empty log
                os.makedirs(os.path.dirname(local), exist_ok=True)
                open(local, "ab").close()
            return
        if 0 < have < size:
            k = min(have, 256)
            data = self.store.read(f"{name}/episodes.jsonl", have - k)
            with open(local, "rb") as f:
                f.seek(have - k)
                tail = f.read()
            if data is not None and data[:k] == tail:
                new = data[k:]
                with open(local, "ab") as f:
                    f.write(new[:new.rfind(b"\n") + 1])
                return
        self.store.get(f"{name}/episodes.jsonl", local)

    # --- one machine ---------------------------------------------------------------------

    def ssh(self, slot, command, marker, timeout=600):
        rc, out, err = self.vms.ssh(slot.name, command, timeout)
        for line in out.splitlines():
            if line.startswith(marker + " "):
                return line[len(marker) + 1:]
        raise MachineLost(f"{marker}: {cr.clean(out + err, 300)}")

    def setup(self, slot):
        pins = [f"jax[tpu]=={md.version('jax')}", f"jaxlib=={md.version('jaxlib')}",
                *cr.PINS, f"numpy=={md.version('numpy')}"]
        pyver = ".".join(map(str, sys.version_info[:2]))
        cmd = f"""set -e
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/{UV}/install.sh | env UV_NO_MODIFY_PATH=1 sh
[ -x venv/bin/python ] || uv venv -q --python {pyver} venv
uv pip install -q --python venv/bin/python {' '.join(shlex.quote(p) for p in pins)}
[ -d new-melting-pot ] || git clone -q {cr.GITHUB} new-melting-pot
git -C new-melting-pot fetch -q origin
git -C new-melting-pot checkout -q {self.commit}
echo ok > setup_ok.txt
gcloud storage cp -q setup_ok.txt {self.store.url}/_machines/{slot.name}/setup_ok.txt
venv/bin/python -c "import jax; d = jax.devices(); print('SETUP jax', jax.__version__, len(d), d[0].device_kind)"
"""
        self.say(f"  {slot.name} set up: " + self.ssh(slot, cmd, "SETUP", timeout=2400))

    def launch(self, slot):
        """Start cloud_vm.py (which starts the training) in the background, after asking
        an earlier one on this machine to save and stop."""
        self.launches += 1
        slot.launch = f"{time.strftime('%m%d%H%M%S')}-{slot.index}-{self.launches}"
        vm = f"out/{self.root}/_vm"
        argv = [self.py, "new-melting-pot/sims/jaxtown/cloud_vm.py", "--store", self.store.url,
                "--machine", slot.name, "--launch", slot.launch, "--sync-minutes",
                str(self.vm_sync), "--commit", str(self.commit), "--", *slot.argv]
        cmd = f"""cd "$HOME"
if [ -f {vm}/cloud_vm.pid ]; then
  p=$(cat {vm}/cloud_vm.pid); kill -TERM "$p" 2>/dev/null || true
  for i in $(seq 300); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$p" 2>/dev/null || true
fi
mkdir -p {vm}
setsid nohup {shlex.join(argv)} > {vm}/cloud_vm_{slot.launch}.out 2>&1 < /dev/null &
echo LAUNCHED ok
"""
        self.ssh(slot, cmd, "LAUNCHED", timeout=600)
        slot.launched_at = time.time()
        self.say(f"  {slot.name}: training launched ({len(slot.runs)} runs, launch {slot.launch})")

    def ask_to_stop(self, slot):
        rc, out, err = self.vms.ssh(
            slot.name, f'cd "$HOME"; kill -TERM "$(cat out/{self.root}/_vm/cloud_vm.pid)" '
            '&& echo ASKED ok', 300)
        return "ASKED ok" in out

    def start(self, slot):
        if self.creates >= self.max_creates:
            raise SystemExit(f"cloud_runner: {self.creates} machines started, stopping "
                             "(--max-creates)")
        self.creates += 1
        rc, out, err = self.vms.create(slot.name)
        if rc != 0:
            wait = BACKOFF[min(slot.fails, len(BACKOFF) - 1)]
            slot.fails += 1
            slot.retry_at = time.time() + wait
            self.say(f"  {slot.name}: no machine ({cr.clean(err or out, 300)}); trying again in "
                     f"{wait // 60} min")
            return
        slot.fails, slot.alive = 0, True
        slot.machines += 1
        self.say(f"{slot.name}: machine {slot.machines} started ({self.vms.accel}, "
                 f"{'spot' if self.vms.spot else 'on-demand'})")
        try:
            if self.do_setup:
                self.setup(slot)
            self.launch(slot)
            slot.start_failures = 0
        except MachineLost as e:
            self.say(f"{slot.name}: failed while starting ({e})")
            self.delete(slot)
            slot.start_failures += 1
            if slot.start_failures >= 3:
                raise SystemExit(f"cloud_runner: {slot.name} failed to start training three "
                                 "times in a row; see runner.log") from None

    def delete(self, slot):
        rc, out, err = self.vms.delete(slot.name)
        self.say(f"{slot.name}: machine deleted" +
                 ("" if rc == 0 else f" (delete said: {cr.clean(err or out, 200)})"))
        slot.alive = False

    def finished(self, slot):
        return all(self.progress(n) >= self.target for n in slot.runs)

    def tend(self, slot):
        """Look at one machine and do what it needs."""
        if slot.done or time.time() < slot.retry_at:
            return
        state = self.vms.state(slot.name)
        if state in WAITING:
            return
        if state != "READY":
            if slot.alive or state:
                self.say(f"{slot.name}: machine {state or 'gone'}; replacing it")
            if state:
                self.delete(slot)
            slot.alive = False
            self.start(slot)
            return
        if not slot.alive:            # READY but not ours (can't happen after the start)
            self.delete(slot)
            return
        st = self.mirror_json(f"_machines/{slot.name}/status.json")
        if not st or st.get("launch") != slot.launch:
            if time.time() - slot.launched_at > max(900, 3 * self.sync_s):
                self.say(f"{slot.name}: launched but never reported; launching again")
                self.crash(slot, "never reported")
            return
        if st["state"] in ("starting", "running"):
            if time.time() - st["time"] > max(900, 4 * 60 * self.vm_sync + self.sync_s):
                self.crash(slot, f"no report for {(time.time() - st['time']) / 60:.0f} min")
            return
        if st["state"] == "exit 0" and self.finished(slot):
            self.say(f"{slot.name}: its runs finished")
            self.pull(checkpoints=slot.runs)
            self.delete(slot)
            slot.done = True
            return
        self.crash(slot, st["state"])

    def crash(self, slot, why):
        slot.crashes += 1
        self.say(f"{slot.name}: training stopped ({why}); last log lines:")
        log = os.path.join(self.local, "_machines", slot.name, f"train_{slot.launch}.log")
        if os.path.exists(log):
            for line in open(log, errors="replace").read().splitlines()[-8:]:
                self.say("    " + cr.clean(line, 200))
        if slot.crashes > self.max_crashes:
            raise SystemExit(f"cloud_runner: {slot.name}'s training stopped {slot.crashes} "
                             "times; giving up (see the log above)")
        try:
            self.launch(slot)
        except MachineLost as e:
            self.say(f"{slot.name}: couldn't relaunch ({e}); replacing the machine")
            self.delete(slot)

    # --- the loop ---------------------------------------------------------------------------

    def report(self):
        self.say("  copied here: " + ", ".join(f"{n} {self.progress(n):,}" for n in self.runs))

    def stop_all(self):
        """Ask every machine to save and put its copies in the bucket, wait for them,
        copy everything home."""
        asked = [s for s in self.slots if s.alive and not s.done and self.ask_to_stop(s)]
        self.say(f"stopping: asked {len(asked)} machines to save and stop")
        until = time.time() + self.stop_wait_s
        while asked and time.time() < until:
            self.store.sync_down(self.local, skip=("ckpt.pkl", "episodes.jsonl"))
            if all(self.exited(s) for s in asked):
                break
            time.sleep(min(10.0, self.sync_s))
        self.pull(checkpoints=[n for s in self.slots if not s.done for n in s.runs])
        self.report()

    def bring_home(self):
        """Every run's copy from the bucket, checkpoint first (no machine is started):
        afterwards train_jax.py --resume carries on here."""
        self.pull(checkpoints=list(self.runs))
        self.report()
        self.log_f.close()

    def exited(self, slot):
        st = self.mirror_json(f"_machines/{slot.name}/status.json") or {}
        return st.get("launch") == slot.launch and st.get("state", "").startswith("exit")

    def run(self):
        self.say(f"runner: {len(self.runs)} runs to {self.target:,} steps each on "
                 f"{len(self.slots)} x {self.vms.accel} "
                 f"({'spot' if self.vms.spot else 'on-demand'}), commit {self.commit}, "
                 f"bucket {self.store.url}, copies in {self.local}")
        # Machines left by an earlier runner (say this PC restarted) keep training
        # unwatched: delete them (before anything goes up, so they can't overwrite it).
        # Their work since their last copy in the bucket (at most --vm-sync-minutes, plus
        # --ckpt-minutes) is redone.
        for slot in self.slots:
            if self.vms.state(slot.name) is not None:
                self.say(f"{slot.name}: left over from an earlier runner")
                self.delete(slot)
        self.send_ahead()
        try:
            while True:
                for slot in self.slots:
                    self.tend(slot)
                if all(s.done for s in self.slots):
                    self.report()
                    self.say("all runs finished")
                    return {n: self.progress(n) for n in self.runs}
                time.sleep(self.sync_s)
                self.pull()
                self.report()
        except KeyboardInterrupt:
            self.say("stop asked: saving on the machines before deleting them")
            try:
                self.stop_all()
            except (OSError, MachineLost) as e:
                self.say(f"  couldn't copy everything home ({e}); the bucket has it")
            raise
        finally:
            for slot in self.slots:
                if slot.alive or (not slot.done and self.vms.state(slot.name) is not None):
                    self.delete(slot)
            self.log_f.close()


def main(argv=None):
    cr.stop_on_signals()
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: cloud_runner.py [options] -- <train_jax.py arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="machines are called <name>-0, <name>-1, ...")
    ap.add_argument("--zone", required=True, help="exactly the zone of the TRC grant")
    ap.add_argument("--type", required=True, help="exactly the TPU type, e.g. v4-8")
    kind = ap.add_mutually_exclusive_group(required=True)
    kind.add_argument("--spot", action="store_true", help="the grant's spot/preemptible quota")
    kind.add_argument("--on-demand", action="store_true", help="the grant's on-demand quota")
    ap.add_argument("--machines", type=int, default=1, help="machines at once (runs dealt out)")
    ap.add_argument("--bucket", required=True, help="gs://<bucket>, in the zone's region")
    ap.add_argument("--any-region", action="store_true", help="skip the bucket region check")
    ap.add_argument("--project", default=None, help="Google Cloud project (default: gcloud's)")
    ap.add_argument("--version", default=None, help="TPU software image (default by type)")
    ap.add_argument("--sync-minutes", type=float, default=10.0, help="how often to look here")
    ap.add_argument("--vm-sync-minutes", type=float, default=5.0,
                    help="how often each machine puts its runs in the bucket")
    ap.add_argument("--max-creates", type=int, default=100)
    ap.add_argument("--max-crashes", type=int, default=3)
    ap.add_argument("--bring-home", action="store_true",
                    help="only copy every run (checkpoints too) from the bucket here, start no "
                         "machine: e.g. to carry on on the GPU here when the grant ends")
    opts = ap.parse_args(argv[:i])
    bucket = opts.bucket.rstrip("/")
    if not bucket.startswith("gs://"):
        raise SystemExit("cloud_runner: --bucket is gs://<bucket>")
    if not opts.any_region:
        check_bucket(bucket, opts.zone)
    train_argv = argv[i + 1:]
    root = os.path.basename(os.path.normpath(train_jax.parse(train_argv).out_root))
    vms = TPUVMs(opts.zone, opts.type, opts.spot, opts.project, opts.version)
    runner = Runner(vms, Store(f"{bucket}/{root}"), train_argv, opts.name, opts.machines,
                    opts.sync_minutes, opts.vm_sync_minutes, opts.max_creates, opts.max_crashes,
                    commit=None if opts.bring_home else cr.pushed_commit())
    if opts.bring_home:
        runner.bring_home()
    else:
        runner.run()


if __name__ == "__main__":
    main()
