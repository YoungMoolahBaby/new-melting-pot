"""Train on a Colab machine (TPU by default) from this PC, and keep going when Colab takes it.

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python sims/jaxtown/colab_runner.py \\
        --session towns -- --runs neighbours:selfish neighbours:fair --river hughes \\
        --steps 1e8 --out-root out/colab/towns

Everything after `--` goes to train_jax.py and must include --out-root, a folder on this
PC. Needs the `colab` command (google-colab-cli, logged in once by hand) and this repo's
current commit pushed to GitHub: the machine clones the public repo at that commit.

What it does:
- starts a Colab machine, installs the library versions we use, clones our code at this
  commit, uploads the latest copy of each run (if any), and starts train_jax.py there in
  the background with --resume;
- in between, runs a tiny bit of Python on the machine every --ping-seconds: Colab keeps a
  machine only while its notebook is active, and training runs in a separate process;
- every --sync-minutes, copies each run back to --out-root here: the checkpoint first,
  then config.json, metrics.csv, the new lines of episodes.jsonl, and the machine's
  train log (so the logs are never older than the checkpoint; --resume drops the extra);
- if the machine is lost (Colab ended it, or it stopped answering), trims the logs here
  back to the checkpoint here, starts a new machine and resumes there;
- when train_jax.py has finished every run, copies everything back and stops the
  machine. The machine is always stopped on the way out, Ctrl-C included; run the same
  command again to carry on from the copies here.

Colab doesn't publish its limits (people report ~12 h sessions, ~90 min idle). Training
saves a checkpoint every --ckpt-minutes (5 unless given), so a lost machine costs at most
that much training, plus the few minutes a new one takes to start.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))

import train_jax  # noqa: E402  (light: JAX only loads when training starts)

town_train = train_jax.town_train

GITHUB = "https://github.com/YoungMoolahBaby/new-melting-pot"
# The library versions we train with on this PC (Colab's own flax can be older than its
# JAX and fail). JAX itself stays as the machine has it, matched to its TPU or GPU.
PINS = ("flax==0.12.12", "optax==0.2.9")
WHOLE = ("config.json", "metrics.csv")   # small files, copied whole each time
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def clean(text, limit=600):
    """CLI output for our log, without anyone's email address."""
    return EMAIL.sub("<account>", text or "").strip()[-limit:]


class MachineLost(Exception):
    pass


class ColabCLI:
    """The `colab` command: each call authenticates, does one thing and exits."""

    def __init__(self, session, accel):
        self.session, self.accel = session, list(accel)   # e.g. ["--tpu", "v5e1"]

    def _run(self, args, timeout, stdin=None):
        try:
            p = subprocess.run(["colab", *args], input=stdin, capture_output=True, text=True,
                               timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"colab {args[0]}: no answer in {timeout}s"

    def new(self):
        return self._run(["new", "-s", self.session, *self.accel], 900)

    def stop(self):
        return self._run(["stop", "-s", self.session], 300)

    def exec(self, code, timeout=180, silence=60):
        return self._run(["exec", "-s", self.session, "--timeout", str(silence)], timeout,
                         stdin=code)

    def upload(self, local, remote):
        return self._run(["upload", "-s", self.session, local, remote], 1800)

    def download(self, remote, local):
        return self._run(["download", "-s", self.session, remote, local], 1800)


def remote_argv(train_argv, remote_out):
    """train_jax.py's arguments for the machine: its own --out-root, always --resume, and
    checkpoints every 5 minutes unless asked otherwise."""
    out, skip = [], False
    for a in train_argv:
        if skip:
            skip = False
            continue
        if a == "--out-root":
            skip = True
            continue
        if a.startswith("--out-root="):
            continue
        out.append(a)
    if not any(a == "--ckpt-minutes" or a.startswith("--ckpt-minutes=") for a in out):
        out += ["--ckpt-minutes", "5"]
    return out + ["--out-root", remote_out, "--resume"]


class Runner:
    def __init__(self, cli, train_argv, sync_minutes=10.0, max_machines=20, max_crashes=3,
                 remote_root="/content", commit=None, setup=True, log=None, ping_seconds=60.0):
        self.cli, self.sync_s, self.ping_s = cli, 60 * sync_minutes, ping_seconds
        self.max_machines, self.max_crashes = max_machines, max_crashes
        self.args = train_jax.parse(train_argv)
        if self.args.out or self.args.folders:
            raise SystemExit("colab_runner: use --runs and --out-root (a folder of runs), "
                             "not --out or --folders")
        self.local = self.args.out_root
        self.runs = dict(train_jax.run_identity(s, self.args) for s in self.args.specs)
        self.target = int(self.args.steps)
        self.r_repo = f"{remote_root}/new-melting-pot"
        self.r_out = f"{remote_root}/out/{os.path.basename(os.path.normpath(self.local))}"
        self.r_sync = f"{remote_root}/_sync"
        self.r_argv = remote_argv(train_argv, self.r_out)
        self.commit, self.do_setup = commit, setup
        # Machine logs are named after the session, so two runners can share --out-root.
        self.log_prefix = f"train_{cli.session}_" if getattr(cli, "session", None) else "train_"
        self.machines = self.crashes = 0
        self.alive, self.last_active = False, 0.0   # last time the notebook ran anything
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

    def machine_log(self):
        """This PC's copy of the current machine's train log."""
        return os.path.join(self.local, f"{self.log_prefix}machine{self.machines}.log")

    def local_steps(self, name):
        """Steps in this PC's copy of the run's checkpoint (0 if none)."""
        path = os.path.join(self.run_dir(name), "ckpt.pkl")
        if not os.path.exists(path):
            return 0, 0
        with open(path, "rb") as f:
            state = pickle.load(f)
        return state["env_steps"], state["update"]

    def check_local(self):
        """The copies here must be runs these settings may resume (else stop before
        renting a machine): train_jax's own folder check, with --resume."""
        args = argparse.Namespace(**{**vars(self.args), "resume": True})
        for name, settings in self.runs.items():
            town_train.check_folder(self.run_dir(name), args, settings,
                                    train_jax.SAME_ON_RESUME)

    # --- talking to the machine --------------------------------------------------------------

    def py(self, code, marker, timeout=180, silence=60, tries=3):
        """Run Python on the machine; return the text after `marker` on its output.
        Raises MachineLost when the machine stops answering."""
        for attempt in range(tries):
            rc, out, err = self.cli.exec(code, timeout=timeout, silence=silence)
            self.last_active = time.time()
            for line in out.splitlines():
                if line.startswith(marker + " "):
                    return line[len(marker) + 1:]
            self.say(f"  machine didn't answer ({marker}, try {attempt + 1}): "
                     f"{clean(out + err, 300)}")
            if attempt + 1 < tries:
                time.sleep(min(20, self.sync_s / 4))
        raise MachineLost(marker)

    def setup_machine(self):
        code = f"""
import importlib.metadata as md, os, subprocess, sys
jv, jlv = md.version("jax"), md.version("jaxlib")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *{list(PINS)!r},
                f"jax=={{jv}}", f"jaxlib=={{jlv}}"], check=True)
repo = {self.r_repo!r}
if not os.path.isdir(repo):
    subprocess.run(["git", "clone", "-q", {GITHUB!r}, repo], check=True)
else:
    subprocess.run(["git", "-C", repo, "fetch", "-q", "origin"], check=True)
subprocess.run(["git", "-C", repo, "checkout", "-q", {self.commit!r}], check=True)
print("SETUP jax", jv, "flax", md.version("flax"), "optax", md.version("optax"))
"""
        self.say("  set up: " + self.py(code, "SETUP", timeout=1200, silence=900, tries=1))

    def upload_runs(self):
        """Trim this PC's logs back to its checkpoints, then upload every run that has a
        checkpoint (a run without one starts afresh on the machine)."""
        dirs = [f"{self.r_out}/{n}" for n in self.runs] + [self.r_sync]
        self.py(f"import os\nfor d in {dirs!r}: os.makedirs(d, exist_ok=True)\nprint('DIRS ok')",
                "DIRS")
        for name in self.runs:
            steps, update = self.local_steps(name)
            if not steps:
                continue
            town_train.trim_logs(self.run_dir(name), update, name)
            for f in ("config.json", "ckpt.pkl", "metrics.csv", "episodes.jsonl"):
                path = os.path.join(self.run_dir(name), f)
                if os.path.exists(path):
                    rc, out, err = self.cli.upload(path, f"{self.r_out}/{name}/{f}")
                    if rc != 0:
                        raise MachineLost(f"upload {name}/{f}: {clean(err)}")
                    self.keep_alive()
            self.say(f"  uploaded {name} at {steps:,} steps")

    def launch(self):
        """Start train_jax.py in the background on the machine (with the machine's own
        Python, where setup installed the libraries); a small shell wrapper records its
        process id and, when it ends, its exit code."""
        code = f"""
import os, shlex, subprocess, sys
out, repo = {self.r_out!r}, {self.r_repo!r}
for f in ("train.pid", "train.exit"):
    if os.path.exists(os.path.join(out, f)):
        os.remove(os.path.join(out, f))
cmd = shlex.join([sys.executable, repo + "/sims/jaxtown/train_jax.py", *{self.r_argv!r}])
q = shlex.quote(out)
script = (f"echo $$ > {{q}}/train.pid; cd {{shlex.quote(repo)}} && "
          f"{{cmd}} >> {{q}}/train.log 2>&1; echo $? > {{q}}/train.exit")
subprocess.Popen(["bash", "-c", script], start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("LAUNCHED ok")
"""
        self.py(code, "LAUNCHED")
        self.say("  training started on the machine")
        self.starting = 0

    def status(self):
        files = [f"{n}/{f}" for n in self.runs for f in ("ckpt.pkl", *WHOLE, "episodes.jsonl")]
        code = f"""
import json, os
out = {self.r_out!r}
st = {{}}
ex, pid = os.path.join(out, "train.exit"), os.path.join(out, "train.pid")
if os.path.exists(ex):
    st["state"] = "exit " + open(ex).read().strip()
elif os.path.exists(pid):
    try:
        p = int(open(pid).read().strip())
        zombie = open(f"/proc/{{p}}/stat").read().split(")")[-1].split()[0] == "Z"
        st["state"] = "vanished" if zombie else "running"
    except (OSError, ValueError, IndexError):
        st["state"] = "vanished"
else:
    st["state"] = "none"
st["files"] = {{f: list(os.stat(os.path.join(out, f))[6:9:2]) for f in {files!r}
               if os.path.exists(os.path.join(out, f))}}
print("STATUS " + json.dumps(st))
"""
        return json.loads(self.py(code, "STATUS"))

    def fetch(self, remote, local):
        """Download to a temporary name, then put it in place (never a half file here)."""
        os.makedirs(os.path.dirname(local), exist_ok=True)
        tmp = local + ".part"
        rc, out, err = self.cli.download(remote, tmp)
        self.keep_alive()
        if rc != 0 or not os.path.exists(tmp):
            self.say(f"  download failed {remote}: {clean(err, 200)}")
            return False
        os.replace(tmp, local)
        return True

    def sync(self, st):
        """Copy each run back: checkpoint first, then the logs."""
        files = st["files"]
        for name in self.runs:
            d, r = self.run_dir(name), f"{self.r_out}/{name}"
            ck = files.get(f"{name}/ckpt.pkl")
            if not ck:
                continue          # no checkpoint yet: nothing worth keeping
            if self.seen.get(f"{name}/ckpt.pkl") != ck:
                if not self.fetch(f"{r}/ckpt.pkl", os.path.join(d, "ckpt.pkl")):
                    continue
                self.seen[f"{name}/ckpt.pkl"] = ck
            for f in WHOLE:
                meta = files.get(f"{name}/{f}")
                if meta and self.seen.get(f"{name}/{f}") != meta:
                    if self.fetch(f"{r}/{f}", os.path.join(d, f)):
                        self.seen[f"{name}/{f}"] = meta
        self.sync_episodes()
        self.fetch(f"{self.r_out}/train.log",
                   self.machine_log())
        done = {n: self.local_steps(n)[0] for n in self.runs}
        self.say("  copied here: " + ", ".join(f"{n} {s:,}" for n, s in done.items()))
        return done

    def sync_episodes(self):
        """episodes.jsonl only grows during a run (by whole lines), so copy only what's new:
        the machine cuts the part past this PC's copy into a chunk file, we append it."""
        offsets = {}
        for name in self.runs:
            if os.path.exists(os.path.join(self.run_dir(name), "ckpt.pkl")):
                p = os.path.join(self.run_dir(name), "episodes.jsonl")
                offsets[name] = os.path.getsize(p) if os.path.exists(p) else 0
        if not offsets:
            return
        code = f"""
import json, os
res = {{}}
for name, off in {offsets!r}.items():
    p = os.path.join({self.r_out!r}, name, "episodes.jsonl")
    if not os.path.exists(p):
        res[name] = 0
        continue
    if os.path.getsize(p) < off:       # cut back (a resume trimmed it): copy it whole
        res[name] = -1
        continue
    with open(p, "rb") as f:
        f.seek(off)
        data = f.read()
    data = data[:data.rfind(b"\\n") + 1]
    with open(os.path.join({self.r_sync!r}, name + ".chunk"), "wb") as f:
        f.write(data)
    res[name] = len(data)
print("CHUNKS " + json.dumps(res))
"""
        res = json.loads(self.py(code, "CHUNKS"))
        for name, n in res.items():
            local = os.path.join(self.run_dir(name), "episodes.jsonl")
            if n == -1:
                self.fetch(f"{self.r_out}/{name}/episodes.jsonl", local)
            elif n > 0:
                tmp = local + ".chunk"
                rc, out, err = self.cli.download(f"{self.r_sync}/{name}.chunk", tmp)
                self.keep_alive()
                if rc == 0 and os.path.exists(tmp) and os.path.getsize(tmp) == n:
                    with open(tmp, "rb") as src, open(local, "ab") as dst:
                        dst.write(src.read())
                if os.path.exists(tmp):
                    os.remove(tmp)

    # --- the loop ----------------------------------------------------------------------------

    def start_machine(self):
        if self.machines >= self.max_machines:
            raise SystemExit(f"colab_runner: {self.machines} machines used, stopping "
                             "(--max-machines)")
        for wait in (0, 120, 300, 600, 1200, 1800):
            time.sleep(wait)
            rc, out, err = self.cli.new()
            if rc == 0:
                break
            self.say(f"  no machine ({clean(err or out, 300)}); trying again in a while")
        else:
            raise SystemExit("colab_runner: Colab gave no machine after several tries")
        self.machines += 1
        self.alive = True
        self.seen = {}
        self.say(f"machine {self.machines} started ({' '.join(self.cli.accel) or 'CPU'})")
        if self.do_setup:
            self.setup_machine()
        self.upload_runs()
        self.launch()

    def stop_machine(self):
        if self.alive:
            rc, out, err = self.cli.stop()
            self.say(f"machine {self.machines} stopped" +
                     ("" if rc == 0 else f" (stop said: {clean(err or out, 200)})"))
            self.alive = False

    def keep_alive(self):
        """Run a tiny bit of Python on the machine if nothing has run there for
        --ping-seconds. Colab keeps a machine only while its notebook is active, and
        training runs in a separate process, so without this the notebook looks idle and
        Colab takes the machine (measured 2026-11-19: gone within 20 quiet minutes).
        Copying files doesn't count as activity, so long copies call this between files.
        Returns False if the machine didn't answer."""
        if time.time() - self.last_active < self.ping_s:
            return True
        rc, out, err = self.cli.exec("print('PING ok')", timeout=120, silence=60)
        self.last_active = time.time()
        if "PING ok" not in out:
            self.say(f"  ping got no answer: {clean(out + err, 200)}")
            return False
        return True

    def wait_and_ping(self, until):
        """Wait for the next copy, keeping the machine active meanwhile. A ping that gets
        no answer ends the wait early, so a lost machine is noticed at once."""
        while time.time() < until:
            time.sleep(max(0.0, min(5.0, until - time.time(),
                                    self.last_active + self.ping_s - time.time())))
            if time.time() < until and not self.keep_alive():
                return

    def finished(self, done):
        return all(s >= self.target for s in done.values())

    def run(self):
        self.check_local()
        self.say(f"runner: {len(self.runs)} runs to {self.target:,} steps each, "
                 f"commit {self.commit}, copies in {self.local}")
        # A machine left under this session name by an earlier runner (say the PC
        # restarted) would block a new one and keep running unwatched: stop it first.
        # Its work since the last copy here is redone. (So: one runner per session name.)
        self.cli.stop()
        start_failures = 0
        try:
            while True:
                if not self.alive:
                    try:
                        self.start_machine()
                        start_failures = 0
                    except MachineLost as e:
                        self.say(f"machine {self.machines} failed while starting ({e})")
                        self.stop_machine()
                        start_failures += 1
                        if start_failures >= 2:
                            raise SystemExit("colab_runner: two machines in a row failed to "
                                             "start training; see runner.log") from None
                        continue
                self.wait_and_ping(time.time() + self.sync_s)
                try:
                    st = self.status()
                    done = self.sync(st)
                except MachineLost as e:
                    self.say(f"machine {self.machines} lost ({e}); starting another")
                    self.stop_machine()           # in case it's half alive
                    continue
                state = st["state"]
                if state == "running":
                    continue
                if state == "none" and self.starting < 3:
                    self.starting += 1            # the wrapper hasn't written its pid yet
                    continue
                if state == "exit 0" and self.finished(done):
                    self.say("all runs finished")
                    return done
                self.crashes += 1
                self.say(f"training stopped on the machine ({state}); last log lines:")
                log = self.machine_log()
                if os.path.exists(log):
                    for line in open(log).read().splitlines()[-8:]:
                        self.say("    " + clean(line, 200))
                if self.crashes > self.max_crashes:
                    raise SystemExit(f"colab_runner: training stopped {self.crashes} times; "
                                     "giving up (see the log above)")
                self.launch()                     # resume on the same machine
        except KeyboardInterrupt:
            if self.alive:
                self.say("Ctrl-C: copying the latest back before stopping the machine")
                try:
                    self.sync(self.status())
                except (MachineLost, OSError):
                    pass
            raise
        finally:
            was_alive = self.alive
            self.stop_machine()
            if not was_alive:
                self.cli.stop()   # stopped while a new machine was starting: don't leave it
            self.log_f.close()


def pushed_commit():
    """This repo's commit, if the code the machine will clone is exactly what's here."""
    git = ["git", "-C", REPO]
    head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain", "--", "sims"], capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit("colab_runner: sims/ has changes not committed; the machine runs "
                         "the commit from GitHub, so commit and push first:\n" + dirty)
    subprocess.run(git + ["fetch", "-q", "origin"], check=True)
    if subprocess.run(git + ["merge-base", "--is-ancestor", "HEAD", "origin/main"]).returncode:
        raise SystemExit(f"colab_runner: commit {head[:7]} isn't on GitHub yet; push first")
    return head


def stop_on_signals():
    """Ctrl-C and `kill` stop the runner the careful way (copy back, stop the machine).
    Needed when it runs in the background: a shell starts background jobs with Ctrl-C
    ignored, and plain `kill` would end it without stopping the Colab machine."""
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def main(argv=None):
    stop_on_signals()
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: colab_runner.py [options] -- <train_jax.py arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", required=True, help="a name for the Colab machine")
    ap.add_argument("--tpu", default="v5e1", help="Colab TPU type (v5e1, v6e1)")
    ap.add_argument("--gpu", default=None, help="a Colab GPU type instead of a TPU")
    ap.add_argument("--sync-minutes", type=float, default=10.0)
    ap.add_argument("--ping-seconds", type=float, default=60.0,
                    help="how often to keep the machine's notebook active between copies")
    ap.add_argument("--max-machines", type=int, default=20)
    ap.add_argument("--max-crashes", type=int, default=3)
    opts = ap.parse_args(argv[:i])
    accel = ["--gpu", opts.gpu] if opts.gpu else ["--tpu", opts.tpu]
    runner = Runner(ColabCLI(opts.session, accel), argv[i + 1:], opts.sync_minutes,
                    opts.max_machines, opts.max_crashes, commit=pushed_commit(),
                    ping_seconds=opts.ping_seconds)
    runner.run()


if __name__ == "__main__":
    main()
