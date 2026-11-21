"""The Cloud TPU runner keeps runs going when machines are taken back, against a pretend
Google Cloud: each "machine" is a folder here and its commands run on this PC, and the
"bucket" is a folder too. Training runs on the CPU (JAX_PLATFORMS=cpu).

    cd /mnt/c/dev/new-melting-pot && JAX_PLATFORMS=cpu ~/venvs/meltingpot/bin/python -m pytest \\
        sims/jaxtown/tests/test_cloud_runner.py -q
"""
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"        # never the GPU, whatever else is training there
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import cloud_runner as cl  # noqa: E402
import cloud_vm  # noqa: E402

UPD = 4 * 128
TINY = ["--envs", "4", "--minibatches", "2", "--rollout", "128", "--ckpt-minutes", "0"]
TRAIN = os.path.join(HERE, "train_jax.py")


class FakeTPUs:
    """Pretend Cloud TPU machines. `preempt_at`: the first time anyone looks at machine
    `victim` after one of its runs has put update `preempt_at` (or later) in the bucket,
    the machine is taken back (training killed, its disk gone, state PREEMPTED).
    `interrupt_at`: likewise, but Ctrl-C is pressed on the PC instead."""
    accel, spot = "v4-8", True

    def __init__(self, root, bucket, victim=None, preempt_at=None, interrupt_at=None,
                 leftover=()):
        self.root, self.bucket = root, bucket
        self.victim, self.preempt_at, self.interrupt_at = victim, preempt_at, interrupt_at
        self.states, self.created, self.deleted, self.preempted = {}, [], [], 0
        for name in leftover:
            self.create(name)
        self.created.clear()

    def _dir(self, name):
        return os.path.join(self.root, name)

    def _update(self, name):
        ups = [0]
        for d in os.listdir(self.bucket) if os.path.isdir(self.bucket) else []:
            p = os.path.join(self.bucket, d, "progress.json")
            if os.path.exists(p):
                prog = json.load(open(p))
                if prog.get("machine") == name:
                    ups.append(prog["update"])
        return max(ups)

    def _kill(self, name):
        for d, _, files in os.walk(self._dir(name)):
            if "cloud_vm.pid" in files:
                try:
                    os.killpg(int(open(os.path.join(d, "cloud_vm.pid")).read()), signal.SIGKILL)
                except (OSError, ValueError):
                    pass

    def create(self, name):
        self.created.append(name)
        os.makedirs(self._dir(name))
        os.symlink(REPO, os.path.join(self._dir(name), "new-melting-pot"))
        self.states[name] = "READY"
        return 0, "", ""

    def state(self, name):
        st = self.states.get(name)
        if st == "READY" and name == self.victim:
            if self.preempt_at and not self.preempted and self._update(name) >= self.preempt_at:
                self._kill(name)
                shutil.rmtree(self._dir(name))
                self.states[name] = st = "PREEMPTED"
                self.preempted += 1
            if self.interrupt_at and self._update(name) >= self.interrupt_at:
                self.interrupt_at = None
                raise KeyboardInterrupt
        return st

    def delete(self, name):
        self._kill(name)
        shutil.rmtree(self._dir(name), ignore_errors=True)
        self.states.pop(name, None)
        self.deleted.append(name)
        return 0, "", ""

    def ssh(self, name, command, timeout=600):
        if self.states.get(name) != "READY":
            return 255, "", "ssh: connect to host: Connection refused"
        p = subprocess.run(["bash", "-c", command], cwd=self._dir(name), capture_output=True,
                           text=True, timeout=timeout, env={**os.environ, "HOME": self._dir(name)})
        return p.returncode, p.stdout, p.stderr


def make_runner(tmp_path, fake, runs, steps, machines=1, **kw):
    local = str(tmp_path / "here" / "runs")
    store = cl.Store("file://" + str(tmp_path / "bucket" / "runs"))
    argv = ["--out-root", local, *TINY, "--runs", *runs, "--steps", str(steps * UPD)]
    r = cl.Runner(fake, store, argv, "t", machines, sync_minutes=0.03, vm_sync_minutes=0.02,
                  commit="test", setup=False, remote_python=sys.executable,
                  log=lambda line: None, **kw)
    return r, local


def train_here(local, runs, steps, extra=()):
    subprocess.run([sys.executable, TRAIN, "--out-root", local, *TINY, "--runs", *runs,
                    "--steps", str(steps * UPD), "--resume", *extra], check=True,
                   capture_output=True)


def updates(path):
    return [int(r.split(",")[0]) for r in open(path).read().splitlines()[1:]]


def test_a_taken_machine_is_replaced_and_its_runs_carry_on_from_the_bucket(tmp_path):
    bucket = str(tmp_path / "bucket" / "runs")
    fake = FakeTPUs(str(tmp_path / "cloud"), bucket, victim="t-0", preempt_at=13,
                    leftover=["t-1"])
    runs = ["neighbours:selfish", "neighbours:fair", "strangers:selfish"]
    r, local = make_runner(tmp_path, fake, runs, 14, machines=2)
    # Games last 1,100+ frames (8.6 updates). The machine is taken back once one of its
    # runs has update 13 in the bucket; its other run trains alongside, 2 updates behind,
    # so it has reached 11 by then and its first games (update 9) are in the bucket with
    # updates to spare. (At 11, a copy still on its way could miss them, and the test
    # failed now and then.) The new machine's resume trims its game log back to the
    # checkpoint, and the copies here must follow. One run already trained 2 updates here
    # (as on the GPU): it goes up and carries on.
    train_here(local, runs[:1], 2)
    here_first = open(os.path.join(local, "neighbours_selfish_s0", "metrics.csv")).read()
    done = r.run()
    assert fake.deleted[0] == "t-1"                        # the leftover, before anything
    assert fake.preempted == 1 and fake.created.count("t-0") == 2
    assert fake.created.count("t-1") == 1 and not fake.states   # all deleted at the end
    assert done == {n: 14 * UPD for n in
                    ("neighbours_selfish_s0", "neighbours_fair_s0", "strangers_selfish_s0")}
    games = {}
    for name in done:
        d = os.path.join(local, name)
        assert cloud_vm.ckpt_steps(os.path.join(d, "ckpt.pkl")) == (14 * UPD, 14)
        assert updates(os.path.join(d, "metrics.csv")) == list(range(1, 15))
        ups = [json.loads(ln)["update"] for ln in open(os.path.join(d, "episodes.jsonl"))]
        games[name] = len(ups)
        assert ups == sorted(ups) and all(u <= 14 for u in ups)
        assert json.load(open(os.path.join(d, "config.json")))["town"] == name.split("_")[0]
        # Here and in the bucket: the same files.
        for f in ("ckpt.pkl", "metrics.csv", "episodes.jsonl", "config.json"):
            assert open(os.path.join(d, f), "rb").read() == \
                open(os.path.join(bucket, name, f), "rb").read(), (name, f)
    # (Each resume starts new games, so a run resumed twice may not finish one: the taken
    # machine's runs together must have, for the copying through its loss to be tested.)
    assert games["strangers_selfish_s0"] + games["neighbours_selfish_s0"] > 0
    # The run trained here kept its first two updates.
    rows = open(os.path.join(local, "neighbours_selfish_s0", "metrics.csv")).read()
    assert rows.startswith(here_first)
    logs = os.listdir(os.path.join(local, "_machines", "t-0"))
    assert sum(f.startswith("train_") for f in logs) == 2   # one per launch, both copied


def test_training_that_keeps_failing_gives_up_and_deletes_the_machine(tmp_path):
    bucket = tmp_path / "bucket" / "runs" / "neighbours_selfish_s0"
    bucket.mkdir(parents=True)
    # A bucket copy made with other settings (a different learning rate): each machine's
    # train_jax refuses to resume it.
    with open(bucket / "ckpt.pkl", "wb") as f:
        pickle.dump({"env_steps": 99 * UPD, "update": 99}, f)
    json.dump({"town": "neighbours", "players": "selfish", "seed": 0, "lr": 1.0},
              open(bucket / "config.json", "w"))
    json.dump({"env_steps": 99 * UPD, "update": 99}, open(bucket / "progress.json", "w"))
    fake = FakeTPUs(str(tmp_path / "cloud"), str(tmp_path / "bucket" / "runs"))
    r, _ = make_runner(tmp_path, fake, ["neighbours:selfish"], 100, max_crashes=1)
    with pytest.raises(SystemExit, match="giving up"):
        r.run()
    assert fake.created == ["t-0"] and fake.deleted == ["t-0"] and not fake.states


def test_ctrl_c_saves_on_the_machine_copies_home_then_deletes_it(tmp_path):
    bucket = str(tmp_path / "bucket" / "runs")
    fake = FakeTPUs(str(tmp_path / "cloud"), bucket, victim="t-0", interrupt_at=2)
    r, local = make_runner(tmp_path, fake, ["neighbours:fair"], 1000)
    with pytest.raises(KeyboardInterrupt):
        r.run()
    assert fake.deleted == ["t-0"] and not fake.states
    d = os.path.join(local, "neighbours_fair_s0")
    steps, update = cloud_vm.ckpt_steps(os.path.join(d, "ckpt.pkl"))
    assert update >= 2
    assert updates(os.path.join(d, "metrics.csv"))[-1] >= update   # logs as new or newer
    st = json.load(open(os.path.join(local, "_machines", "t-0", "status.json")))
    assert st["state"] == "exit 0"                         # it saved and stopped cleanly
    assert open(os.path.join(d, "ckpt.pkl"), "rb").read() == \
        open(os.path.join(bucket, "neighbours_fair_s0", "ckpt.pkl"), "rb").read()


def test_spreading_runs_over_chips_trains_exactly_as_on_one(tmp_path):
    runs = ["neighbours:selfish", "isolating:fair"]
    out = {}
    for chips in (1, 2):
        root = str(tmp_path / f"chips{chips}")
        env = {**os.environ, "XLA_FLAGS": f"--xla_force_host_platform_device_count={chips}"}
        p = subprocess.run([sys.executable, TRAIN, "--out-root", root, *TINY, "--runs", *runs,
                            "--steps", str(2 * UPD)], env=env, capture_output=True, text=True,
                           check=True)
        assert f"of {chips} cpu chips" in p.stdout
        out[chips] = {n: pickle.load(open(os.path.join(root, n, "ckpt.pkl"), "rb"))
                      for n in ("neighbours_selfish_s0", "isolating_fair_s0")}
    import jax
    for name in out[1]:
        a, b = out[1][name], out[2][name]
        for x, y in zip(jax.tree.leaves((a["params"], a["opt"])),
                        jax.tree.leaves((b["params"], b["opt"]))):
            assert np.array_equal(x, y), name
        assert np.array_equal(a["key"], b["key"])


def test_machines_use_exactly_the_granted_kind_and_get_their_share_of_runs(tmp_path):
    vms = cl.TPUVMs("us-central2-b", "v4-8", spot=True, project="p")
    assert vms.create_args("t-0") == ["create", "t-0", "--accelerator-type", "v4-8",
                                      "--version", "tpu-ubuntu2204-base", "--spot", "--quiet"]
    assert vms.where == ["--zone", "us-central2-b", "--project", "p"]
    assert "--spot" not in cl.TPUVMs("europe-west4-a", "v6e-8", spot=False).create_args("x")
    assert cl.tpu_version("v5litepod-8") == "v2-alpha-tpuv5-lite"
    with pytest.raises(SystemExit, match="several hosts"):
        cl.tpu_version("v4-32")
    assert cl.same_region("US-CENTRAL2\n", "us-central2-b")
    assert not cl.same_region("US", "us-central2-b")
    fake = FakeTPUs(str(tmp_path / "cloud"), str(tmp_path / "bucket"))
    r, _ = make_runner(tmp_path, fake, ["neighbours:selfish", "neighbours:fair",
                                        "strangers:fair"], 10, machines=2)
    assert [s.runs for s in r.slots] == [["neighbours_selfish_s0", "strangers_fair_s0"],
                                         ["neighbours_fair_s0"]]
    argv = r.slots[0].argv
    assert argv[argv.index("--runs") + 1:argv.index("--runs") + 3] == [
        "neighbours:selfish:0:melting_pot", "strangers:fair:0:melting_pot"]
    assert argv[-3:] == ["--out-root", "out/runs", "--resume"]
    assert "neighbours:fair:0:melting_pot" not in argv
