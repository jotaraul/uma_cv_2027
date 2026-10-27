"""Checks a Part 1 submission the way we run it, before you submit it.

    python check_submission.py team_<name>/ --data fruits_part1/train
    python check_submission.py team_<name>/ --data fruits_part1/train --device cpu      # no GPU

Steps:
  1. checks the folder: required files, size, things that must not be there;
  2. installs requirements.txt in a new virtual environment with Python 3.12
     (with your Python if 3.12 is not installed; then it warns you);
  3. runs `python main.py detect` on a few images of the train set (10 by
     default, chosen at random from --data) and measures the time;
  4. validates the predictions file and scores it against the annotations of
     those images with evaluate.py.

It works on Linux, macOS and Windows. It does not disable the internet, so check
yourself that main.py does not download anything: on our machine it runs
without internet access.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate  # noqa: E402

REQUIRED = ["README.md", "requirements.txt", "main.py"]
FORBIDDEN_DIRS = {".venv", "venv", "env", "__pycache__", ".git", ".ipynb_checkpoints"}
MAX_SIZE_MB = 500
TIME_LIMIT_S = 20 * 60          # for the whole test set
TEST_IMAGES = 200               # for the time estimate: as many images as the train set
PYTHON = "3.12"


# ----------------------------------------------------------------------------- 1. folder

def check_folder(sub):
    """Returns (errors, warnings) about the structure of the submission folder."""
    errors, warnings = [], []
    for f in REQUIRED:
        if not (sub / f).is_file():
            errors.append(f"missing {f}")
    size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file()) / 1e6
    if size > MAX_SIZE_MB:
        errors.append(f"the folder takes {size:.0f} MB, more than {MAX_SIZE_MB} MB")
    for d in sub.rglob("*"):
        if d.is_dir() and d.name in FORBIDDEN_DIRS:
            warnings.append(f"remove {d.relative_to(sub)}/ from the submission")
    images = [f for f in sub.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(images) > 50:
        warnings.append(f"{len(images)} images in the folder: do not include the dataset")
    req = sub / "requirements.txt"
    if req.is_file():
        for line in req.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line and not line.startswith("-") and "==" not in line and "@" not in line:
                warnings.append(f"requirements.txt: pin an exact version for '{line}' (package==version)")
    return errors, warnings, size


# ----------------------------------------------------------------------------- inputs

def make_input(data, out, n=None, seed=0):
    """Input folder for main.py: images/ and images.json, without annotations. Returns the image ids."""
    meta = json.loads((data / "images.json").read_text())
    images = meta["images"]
    if n is not None and n < len(images):
        images = random.Random(seed).sample(images, n)
    (out / "images").mkdir(parents=True)
    for im in images:
        shutil.copy2(data / "images" / im["file_name"], out / "images" / im["file_name"])
    (out / "images.json").write_text(json.dumps({"categories": meta["categories"], "images": images}))
    return [im["id"] for im in images]


# ----------------------------------------------------------------------------- 2-3. local virtual environment

def find_python():
    """Command that runs Python 3.12 if it is installed, else the current interpreter (with a warning)."""
    candidates = [[f"python{PYTHON}"], ["python3"], ["python"]]
    if shutil.which("py"):                                   # the Python launcher of Windows
        candidates.insert(0, ["py", f"-{PYTHON}"])
    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        r = subprocess.run(cmd + ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() == PYTHON:
            return cmd, None
    v = "%d.%d" % sys.version_info[:2]
    return [sys.executable], f"Python {PYTHON} is not installed: using your Python {v}. We run submissions " \
                             f"with Python {PYTHON}, so check that your requirements also install there"


def venv_python(env_dir):
    """The Python of a virtual environment, on Windows or on Linux and macOS."""
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def run_logged(cmd, log, timeout=None, cwd=None, env=None, show=False):
    """Runs a command, appending its output to log (and showing it too with show=True).
    Returns (return code or None on timeout, seconds)."""
    t0 = time.time()
    sys.stdout.flush()                       # our own messages before those of the command
    with open(log, "ab") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n".encode())
        f.flush()
        if not show:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd, env=env).returncode
            except subprocess.TimeoutExpired:
                rc = None
            return rc, time.time() - t0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env)

        def pump():
            for chunk in iter(lambda: proc.stdout.read1(4096), b""):
                f.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = None
        reader.join(5)
    return rc, time.time() - t0


def install_venv(sub, env_dir, python, log):
    """Creates a virtual environment with the Python command `python` and installs requirements.txt in it."""
    rc, _ = run_logged(python + ["-m", "venv", str(env_dir)], log)
    if rc != 0:
        return False
    py = str(venv_python(env_dir))
    run_logged([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], log)
    rc, _ = run_logged([py, "-m", "pip", "install", "-r", str(sub / "requirements.txt")], log)
    return rc == 0


def run_local(sub, env_dir, input_dir, output, device, timeout, log, show=False, prefix=()):
    """Runs main.py detect in the virtual environment. prefix: command to run it with (for example,
    one that disables the network). Returns (return code or None on timeout, seconds)."""
    cmd = list(prefix) + [str(venv_python(env_dir)), "main.py", "detect", "--input", str(input_dir),
                          "--output", str(output), "--device", device]
    return run_logged(cmd, log, timeout=timeout, cwd=sub, show=show)


# ----------------------------------------------------------------------------- 4. predictions

def score(gt_path, pred_path, img_ids, verbose=False):
    """Validates and scores a predictions file. Returns (metrics or None, error, warnings)."""
    if not pred_path.is_file():
        return None, "main.py did not write the predictions file", []
    gt = evaluate.load_gt(gt_path)
    try:
        preds, warnings = evaluate.validate(json.loads(pred_path.read_text()), gt)
    except (evaluate.InvalidPredictions, json.JSONDecodeError) as e:
        return None, f"invalid predictions file: {e}", []
    missing = set(img_ids) - {p["image_id"] for p in preds}
    if missing:
        warnings.append(f"no detections in {len(missing)} of {len(img_ids)} images")
    return evaluate.evaluate(gt, preds, img_ids, verbose), None, warnings


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("submission", type=Path, help="your submission folder, team_<name>/")
    p.add_argument("--data", type=Path, required=True, help="the train set folder, fruits_part1/train")
    p.add_argument("--n", type=int, default=10, help="number of images to run on (default 10)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--keep", action="store_true", help="keep the work folder (environment, logs, predictions)")
    args = p.parse_args()
    sub = args.submission.resolve()
    work = Path(tempfile.mkdtemp(prefix="check_submission_"))
    log = work / "log.txt"
    problems, warnings = [], []

    print(f"1. Folder {sub}")
    errors, w, size = check_folder(sub)
    warnings += w
    print(f"   {size:.0f} MB" + "".join(f"\n   ERROR: {e}" for e in errors) + "".join(f"\n   warning: {x}" for x in w))
    if errors:
        sys.exit("The submission is not valid: fix the errors above.")

    input_dir = work / "input"
    img_ids = make_input(args.data, input_dir, args.n)
    pred = work / "out" / "predictions.json"
    pred.parent.mkdir()
    python, w = find_python()
    if w:
        warnings.append(w)
        print(f"   warning: {w}")
    print(f"2. Installing requirements.txt in a new virtual environment ({' '.join(python)}); "
          "this may take a few minutes")
    if not install_venv(sub, work / "env", python, log):
        sys.exit(f"   ERROR: the installation failed, see {log}")
    print(f"3. Running main.py detect on {len(img_ids)} images")
    rc, secs = run_local(sub, work / "env", input_dir, pred, args.device, TIME_LIMIT_S, log, show=True)
    if rc is None:
        problems.append(f"main.py did not finish in {TIME_LIMIT_S // 60} minutes")
    elif rc != 0:
        problems.append(f"main.py ended with exit code {rc}, see {log}")
    estimate = secs * TEST_IMAGES / len(img_ids)
    print(f"   {secs:.0f} s, {secs / len(img_ids):.1f} s per image (including loading the model)")
    if estimate > TIME_LIMIT_S:
        warnings.append(f"at this speed, a test set as large as the train set ({TEST_IMAGES} images) would take "
                        f"about {estimate / 60:.0f} min, more than the limit of {TIME_LIMIT_S // 60} min")
        print(f"   warning: {warnings[-1]}")

    if not problems:
        print("4. Validating and scoring the predictions")
        metrics, error, w = score(args.data / "annotations.json", pred, img_ids, verbose=True)
        warnings += w
        for x in w:
            print(f"   warning: {x}")
        if error:
            problems.append(error)
        else:
            print(evaluate.format_metrics(metrics, indent=3))

    print()
    if problems:
        print("The submission FAILS:\n" + "\n".join(f"  - {x}" for x in problems))
    else:
        print("The submission passes the checks" + (f", with {len(warnings)} warnings." if warnings else "."))
    if args.keep or problems:
        print(f"Work folder (logs, predictions): {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
        print("The work folder (logs, predictions) was deleted; use --keep to keep it.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
