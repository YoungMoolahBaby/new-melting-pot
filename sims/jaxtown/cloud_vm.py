"""The machine's half of cloud_runner.py: runs on a Cloud TPU machine, next to the training.

    venv/bin/python new-melting-pot/sims/jaxtown/cloud_vm.py --store gs://bucket/towns \\
        --machine towns-0 --launch <id> -- <train_jax.py arguments, with --out-root>

- fetches each of its runs from the store (a Cloud Storage folder) unless this machine
  already holds a copy (a relaunch on the same machine), into train_jax's --out-root;
- starts train_jax.py on them (with --resume), its output to <out-root>/_vm/train_<id>.log;
- every --sync-minutes, and when training ends, puts each run back in the store: the
  logs first, then the checkpoint, then progress.json (the checkpoint's step count), so
  the store's logs are never older than its checkpoint (train_jax --resume trims logs
  that are newer);
- keeps _machines/<machine>/status.json in the store up to date (state, time, launch id):
  that's how cloud_runner.py on the PC follows this machine;
- on SIGTERM: asks the training to stop (it saves first), puts everything in the store,
  then exits.

Never imports JAX (a TPU takes one process, and that's the training's).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "town"))

import train_jax  # noqa: E402  (light: JAX only loads when training starts)

LOGS = ("config.json", "metrics.csv", "episodes.jsonl")
GROWING = ("metrics.csv", "episodes.jsonl")   # appended to while training runs


class Store:
    """Where the runs' copies live: a Cloud Storage folder (gs://bucket/path, through the
    `gcloud storage` command), or, for the tests, a plain folder (file:///path)."""

    def __init__(self, url):
        self.url = url.rstrip("/")
        self.dir = self.url[len("file://"):] if self.url.startswith("file://") else None

    def _gcloud(self, args, timeout=1800):
        try:
            p = subprocess.run(["gcloud", "storage", *args], capture_output=True,
                               timeout=timeout)
            return p.returncode, p.stdout, p.stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            return 124, b"", f"gcloud storage {args[0]}: no answer in {timeout}s"

    def _path(self, key):
        return os.path.join(self.dir, *key.split("/"))

    def put_many(self, paths, prefix):
        """Copy local files into the store folder `prefix` (each object appears whole)."""
        if not paths:
            return True
        if self.dir:
            os.makedirs(self._path(prefix), exist_ok=True)
            for p in paths:
                dst = self._path(f"{prefix}/{os.path.basename(p)}")
                shutil.copyfile(p, dst + ".part")
                os.replace(dst + ".part", dst)
            return True
        rc, _, err = self._gcloud(["cp", *paths, f"{self.url}/{prefix}/"])
        if rc:
            print(f"store: upload to {prefix} failed: {err[-300:]}", flush=True)
        return rc == 0

    def put(self, path, key):
        prefix, name = key.rsplit("/", 1)
        if os.path.basename(path) == name:
            return self.put_many([path], prefix)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copyfile(path, os.path.join(tmp, name))
            return self.put_many([os.path.join(tmp, name)], prefix)

    def get(self, key, local):
        """Copy one object here (to a temporary name, then into place); False if missing."""
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        tmp = local + ".part"
        if self.dir:
            if not os.path.exists(self._path(key)):
                return False
            shutil.copyfile(self._path(key), tmp)
        else:
            rc, _, _ = self._gcloud(["cp", f"{self.url}/{key}", tmp])
            if rc or not os.path.exists(tmp):
                return False
        os.replace(tmp, local)
        return True

    def read(self, key, start=0):
        """The object's bytes from `start` on (None if it's missing)."""
        if self.dir:
            if not os.path.exists(self._path(key)):
                return None
            with open(self._path(key), "rb") as f:
                f.seek(start)
                return f.read()
        rng = ["-r", f"{start}-"] if start else []
        rc, out, _ = self._gcloud(["cat", *rng, f"{self.url}/{key}"], timeout=600)
        return out if rc == 0 else None

    def read_json(self, key):
        data = self.read(key)
        try:
            return json.loads(data) if data else None
        except ValueError:
            return None

    def write_json(self, key, obj):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, key.rsplit("/", 1)[-1])
            with open(path, "w") as f:
                json.dump(obj, f)
            return self.put(path, key)

    def sizes(self):
        """{key: size} of every object in the store."""
        if self.dir:
            found = {}
            for d, _, files in os.walk(self.dir):
                for f in files:
                    if not f.endswith(".part"):
                        p = os.path.join(d, f)
                        found[os.path.relpath(p, self.dir).replace(os.sep, "/")] = \
                            os.path.getsize(p)
            return found
        rc, out, err = self._gcloud(["ls", "-l", f"{self.url}/**"], timeout=600)
        if rc:   # an empty store "matched no objects"; anything else: don't know
            return {} if "matched no objects" in err else None
        found, head = {}, self.url + "/"
        for line in out.decode(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[-1].startswith(head):
                found[parts[-1][len(head):]] = int(parts[0])
        return found

    def sync_down(self, local, skip):
        """Make the store's files (all but those named in `skip`) the same here."""
        if self.dir:
            for key, size in (self.sizes() or {}).items():
                if key.rsplit("/", 1)[-1] in skip:
                    continue
                dst = os.path.join(local, *key.split("/"))
                if not os.path.exists(dst) or os.path.getsize(dst) != size or \
                        open(dst, "rb").read() != open(self._path(key), "rb").read():
                    self.get(key, dst)
            return True
        pattern = "|".join(rf"(^|.*/){n.replace('.', '[.]')}$" for n in skip)
        os.makedirs(local, exist_ok=True)
        rc, _, err = self._gcloud(["rsync", "-r", "-x", pattern, self.url, local])
        if rc:
            print(f"store: copying here failed: {err[-300:]}", flush=True)
        return rc == 0


class _Anything:
    """Stands in for every class a checkpoint names (optax's states, ...), so reading its
    step count needs no JAX."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _StepsOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("builtins", "collections", "copyreg", "numpy", "_codecs",
                                    "ml_dtypes"):
            return super().find_class(module, name)
        return _Anything


def ckpt_steps(path):
    """(env_steps, update) of a checkpoint, without loading JAX."""
    with open(path, "rb") as f:
        state = _StepsOnly(f).load()
    return int(state["env_steps"]), int(state["update"])


def copy_whole_lines(src, dst):
    """Copy a log that's still being written, up to its last complete line."""
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data[:data.rfind(b"\n") + 1])


class Machine:
    def __init__(self, store, machine, launch, train_argv, sync_minutes, commit=""):
        self.store, self.machine, self.launch = store, machine, launch
        self.sync_s = 60 * sync_minutes
        self.commit = commit
        args = train_jax.parse(train_argv)
        self.work = os.path.abspath(args.out_root)
        self.train_argv = [a for a in train_argv]
        i = self.train_argv.index("--out-root") if "--out-root" in self.train_argv else None
        if i is None:
            raise SystemExit("cloud_vm: the training arguments need --out-root")
        self.train_argv[i + 1] = self.work
        self.names = [train_jax.run_identity(s, args)[0] for s in args.specs]
        self.vm = os.path.join(self.work, "_vm")
        self.snap = os.path.join(self.vm, "snap")
        os.makedirs(self.snap, exist_ok=True)
        self.sent = {}    # run -> (inode, mtime, size) of the checkpoint last put in the store
        self.proc = None

    def status(self, state, **extra):
        self.store.write_json(f"_machines/{self.machine}/status.json", {
            "state": state, "time": time.time(), "launch": self.launch, "runs": self.names,
            "commit": self.commit, **extra})

    def fetch(self):
        """Each run from the store, unless this machine already has its checkpoint."""
        for name in self.names:
            d = os.path.join(self.work, name)
            if os.path.exists(os.path.join(d, "ckpt.pkl")):
                print(f"{name}: carrying on from this machine's copy", flush=True)
                continue
            if self.store.get(f"{name}/ckpt.pkl", os.path.join(d, "ckpt.pkl.store")):
                for f in LOGS:
                    self.store.get(f"{name}/{f}", os.path.join(d, f))
                os.replace(os.path.join(d, "ckpt.pkl.store"), os.path.join(d, "ckpt.pkl"))
                print(f"{name}: fetched from the store at {ckpt_steps(os.path.join(d, 'ckpt.pkl'))[0]:,} "
                      "steps", flush=True)
            else:
                print(f"{name}: not in the store yet, starts afresh", flush=True)

    def upload(self):
        """Each run into the store: logs, then checkpoint, then its step count."""
        for name in self.names:
            d, s = os.path.join(self.work, name), os.path.join(self.snap, name)
            ck = os.path.join(d, "ckpt.pkl")
            if not os.path.exists(ck):
                continue          # nothing worth keeping yet (logs alone can't resume)
            os.makedirs(s, exist_ok=True)
            # The checkpoint as it is now (train_jax replaces it whole, never in place),
            # then the logs, which are at least as new.
            snap_ck = os.path.join(s, "ckpt.pkl")
            with open(ck, "rb") as src, open(snap_ck + ".tmp", "wb") as dst:
                st = os.fstat(src.fileno())
                shutil.copyfileobj(src, dst)
            os.replace(snap_ck + ".tmp", snap_ck)
            logs = []
            for f in LOGS:
                if os.path.exists(os.path.join(d, f)):
                    if f in GROWING:
                        copy_whole_lines(os.path.join(d, f), os.path.join(s, f))
                    else:              # written once, whole (and without a last newline)
                        shutil.copyfile(os.path.join(d, f), os.path.join(s, f))
                    logs.append(os.path.join(s, f))
            steps, update = ckpt_steps(snap_ck)
            if not self.store.put_many(logs, name):
                continue
            sig = (st.st_ino, st.st_mtime_ns, st.st_size)
            if self.sent.get(name) != sig:
                if not self.store.put(snap_ck, f"{name}/ckpt.pkl"):
                    continue
                self.sent[name] = sig
            self.store.write_json(f"{name}/progress.json",
                                  {"env_steps": steps, "update": update, "time": time.time(),
                                   "machine": self.machine})
        log = os.path.join(self.vm, f"train_{self.launch}.log")
        if os.path.exists(log):
            self.store.put(log, f"_machines/{self.machine}/train_{self.launch}.log")

    def run(self):
        with open(os.path.join(self.vm, "cloud_vm.pid"), "w") as f:
            f.write(str(os.getpid()))
        self.status("starting")
        stop = {"asked": False}

        def on_signal(signum, frame):
            stop["asked"] = True
            if self.proc and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)    # train_jax saves, then stops

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        self.fetch()
        if stop["asked"]:
            self.status("exit stopped")
            return 0
        log = open(os.path.join(self.vm, f"train_{self.launch}.log"), "ab")
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "train_jax.py"), *self.train_argv],
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.status("running")
        while True:
            try:
                code = self.proc.wait(timeout=self.sync_s)
            except subprocess.TimeoutExpired:
                code = None
            if code is not None:
                log.close()
            self.upload()
            if code is not None:
                self.status(f"exit {code}")
                return code
            self.status("running")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: cloud_vm.py [options] -- <train_jax.py arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", required=True, help="gs://bucket/folder (or file:///path)")
    ap.add_argument("--machine", required=True)
    ap.add_argument("--launch", required=True, help="id of this launch (from the PC)")
    ap.add_argument("--sync-minutes", type=float, default=5.0)
    ap.add_argument("--commit", default="")
    opts = ap.parse_args(argv[:i])
    m = Machine(Store(opts.store), opts.machine, opts.launch, argv[i + 1:], opts.sync_minutes,
                opts.commit)
    sys.exit(m.run())


if __name__ == "__main__":
    main()
