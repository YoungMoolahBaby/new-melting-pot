"""The Colab runner keeps a run going when the machine is lost, against a pretend Colab:
the "machine" is a folder here, and its Python (training included) runs on this PC.

    cd /mnt/c/dev/new-melting-pot && ~/venvs/meltingpot/bin/python -m pytest sims/jaxtown/tests -q
"""
import json
import os
import shutil
import signal
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import colab_runner as cr  # noqa: E402

# Tiny training: 4 games, 128-step rollouts, a checkpoint after every update. Games last
# 1,099+ frames, so 30 updates (3,840 frames a game) finish a few games in each run.
UPD = 4 * 128
TINY = ["--envs", "4", "--minibatches", "2", "--rollout", "128", "--ckpt-minutes", "0"]


class FakeColab:
    """Pretend Colab: one machine at a time, a folder `root`. `lose_at`: the first time the
    machine is asked anything after every run has logged this update, it is lost (its
    training killed, its files gone), as when Colab takes a machine back."""

    def __init__(self, root, lose_at=None, broken_launch=False):
        self.root, self.lose_at, self.broken_launch = root, lose_at, broken_launch
        self.accel, self.up, self.started, self.lost = [], False, 0, 0

    def new(self):
        self.started += 1
        self.up = True
        os.makedirs(self.root, exist_ok=True)
        link = os.path.join(self.root, "new-melting-pot")
        if not os.path.exists(link):
            os.symlink(REPO, link)
        return 0, "", ""

    def kill_training(self):
        for pid_file in self._find("train.pid"):
            try:
                os.killpg(int(open(pid_file).read().strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass

    def _find(self, fname):
        out = os.path.join(self.root, "out")
        return [os.path.join(d, fname) for d, _, files in os.walk(out) if fname in files]

    def stop(self):
        self.kill_training()
        self.up = False
        return 0, "", ""

    def lose(self):
        self.kill_training()
        for d in ("out", "_sync"):
            shutil.rmtree(os.path.join(self.root, d), ignore_errors=True)
        self.up, self.lost = False, self.lost + 1

    def _logged_update(self):
        ups = []
        for path in self._find("metrics.csv"):
            lines = open(path).read().splitlines()
            ups.append(int(lines[-1].split(",")[0]) if len(lines) > 1 else 0)
        return min(ups) if ups else 0

    def exec(self, code, timeout=180, silence=60):
        if self.up and self.lose_at and not self.lost and self._logged_update() >= self.lose_at:
            self.lose()
        if not self.up:
            return 1, "", "session not found"
        if self.broken_launch and "LAUNCHED" in code:
            return 1, "", "kernel died"
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def upload(self, local, remote):
        if not self.up:
            return 1, "", "session not found"
        os.makedirs(os.path.dirname(remote), exist_ok=True)
        shutil.copyfile(local, remote)
        return 0, "", ""

    def download(self, remote, local):
        if not self.up or not os.path.exists(remote):
            return 1, "", "not found"
        shutil.copyfile(remote, local)
        return 0, "", ""


def runner(tmp_path, fake, extra, **kw):
    local = str(tmp_path / "here" / "runs")
    argv = ["--out-root", local, *TINY, *extra]
    return cr.Runner(fake, argv, sync_minutes=0.03, remote_root=str(tmp_path / "machine"),
                     commit="test", setup=False, log=lambda line: None, **kw), local


def updates(path):
    rows = open(path).read().splitlines()[1:]
    return [int(r.split(",")[0]) for r in rows]


def test_a_lost_machine_is_replaced_and_the_runs_resume_without_gaps(tmp_path):
    fake = FakeColab(str(tmp_path / "machine"), lose_at=12)
    r, local = runner(tmp_path, fake, ["--runs", "neighbours:selfish", "neighbours:fair:0:capped",
                                       "--steps", str(30 * UPD)])
    done = r.run()
    assert fake.started == 2 and fake.lost == 1 and not fake.up      # replaced, then stopped
    assert set(done) == {"neighbours_selfish_s0", "neighbours_fair_s0_capped"}
    for name in done:
        assert done[name] >= 30 * UPD
        assert updates(os.path.join(local, name, "metrics.csv")) == list(range(1, 31))
        eps = [json.loads(ln) for ln in open(os.path.join(local, name, "episodes.jsonl"))]
        ups = [e["update"] for e in eps]
        assert eps and ups == sorted(ups) and ups[-1] <= 30
        cfg = json.load(open(os.path.join(local, name, "config.json")))
        assert cfg["out_root"].startswith(str(tmp_path / "machine"))   # written on the machine
    assert os.path.exists(os.path.join(local, "train_machine1.log"))
    assert os.path.exists(os.path.join(local, "train_machine2.log"))


def test_the_machine_is_always_stopped_and_bad_starts_give_up(tmp_path):
    fake = FakeColab(str(tmp_path / "machine"), broken_launch=True)
    r, _ = runner(tmp_path, fake, ["--steps", str(2 * UPD)])
    with pytest.raises(SystemExit, match="two machines in a row"):
        r.run()
    assert fake.started == 2 and not fake.up


def test_runs_here_that_cannot_resume_stop_it_before_renting_a_machine(tmp_path):
    fake = FakeColab(str(tmp_path / "machine"))
    r, local = runner(tmp_path, fake, ["--steps", str(2 * UPD)])
    os.makedirs(os.path.join(local, "neighbours_selfish_s0"))
    open(os.path.join(local, "neighbours_selfish_s0", "metrics.csv"), "w").write("update\n")
    with pytest.raises(SystemExit, match="logs but no checkpoint"):
        r.run()
    assert fake.started == 0


def test_the_machine_gets_its_own_folder_resume_and_5_minute_checkpoints():
    argv = ["--runs", "neighbours:fair", "--out-root", "out/colab/x", "--steps", "1e6"]
    assert cr.remote_argv(argv, "/content/out/x") == [
        "--runs", "neighbours:fair", "--steps", "1e6", "--ckpt-minutes", "5",
        "--out-root", "/content/out/x", "--resume"]
    assert "5" not in cr.remote_argv(["--ckpt-minutes=2", "--out-root=a"], "/b")


def test_logs_never_show_an_email_address():
    assert cr.clean("[colab] signed in as some.one+x@example.com ok") == (
        "[colab] signed in as <account> ok")
