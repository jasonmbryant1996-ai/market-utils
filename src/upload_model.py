"""
upload_model.py
================
Run this ONCE per model, locally, before deploying that model.

Stages the given .pth/.pkl under model/<model_name>/, temporarily commits
them (force-added, bypassing .gitignore) so the upload_model.yml workflow
can see them, triggers that workflow with the matching model_name input,
waits for it to finish, then removes the files from git again. The files
never permanently live in git history — they end up as a private GitHub
Actions artifact named "regime-model-<model_name>".

This version shells out to the `gh` CLI for everything GitHub-API-related
(triggering the workflow, watching the run) instead of raw `requests`
calls — gh has its own robust retry/reconnect handling built in, and
reuses your existing `gh auth login` session, so there's no --token to
pass and no hand-rolled polling loop to break on a network hiccup.

It's also safe to re-run after a partial failure: Step 2 checks git status
first and skips committing if the model files are already staged/pushed
from a previous attempt, then just continues on to trigger the workflow.

USAGE:
  python src/upload_model.py \
    --pth /path/to/best_regime_transformer.pth \
    --pkl /path/to/scaler_X.pkl \
    --model-name bear_6h \
    --repo YOUR_USERNAME/btc-regime-monitor

Run it again with --model-name bull_48h and that model's files to deploy
the second model.

REQUIREMENTS:
  GitHub CLI (gh) installed and logged in: https://cli.github.com/
  Run `gh auth login` once beforehand if you haven't already.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def run(cmd: list, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if check and result.returncode != 0:
        print(f"FAILED (exit {result.returncode}):\n{result.stderr}")
        sys.exit(1)
    return result


def run_with_retry(cmd: list, attempts: int = 3, delay: int = 5, **kwargs) -> subprocess.CompletedProcess:
    """Retries a subprocess call a few times before giving up — network
    hiccups against the GitHub API are common and usually transient."""
    last_result = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result
        last_result = result
        if attempt < attempts:
            print(f"  (attempt {attempt}/{attempts} failed, retrying in {delay}s: {result.stderr.strip()[:200]})")
            time.sleep(delay)
    print(f"FAILED after {attempts} attempts:\n{last_result.stderr}")
    sys.exit(1)


def stage_and_commit(paths: list, message: str) -> bool:
    """
    Stages and commits the given paths, returning True if a new commit was
    actually created. Rather than pre-checking git status (which can
    misfire across platforms/path-quoting quirks and either force an
    unnecessary re-commit or wrongly skip a real one), this just attempts
    the commit and treats git's own "nothing to commit" response as the
    signal that these exact bytes are already committed — no guessing.
    """
    subprocess.run(["git", "add", "-f"] + paths, check=True)
    result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
    if result.returncode == 0:
        return True
    combined = (result.stdout + result.stderr).lower()
    if "nothing to commit" in combined or "nothing added to commit" in combined:
        print("  Nothing new to commit — these exact files are already committed. Continuing.")
        return False
    print(f"FAILED to commit:\n{result.stderr}")
    sys.exit(1)


def trigger_workflow(repo: str, model_name: str) -> int:
    """Dispatches upload_model.yml via gh, then returns the new run's databaseId."""
    print("  Dispatching workflow via gh...")
    run_with_retry([
        "gh", "workflow", "run", "upload_model.yml",
        "--repo", repo,
        "-f", "confirm=upload",
        "-f", f"model_name={model_name}",
    ])

    print("  Waiting for the new run to register...")
    time.sleep(8)   # gh needs a moment before the run shows up in `gh run list`

    for _ in range(10):
        result = run_with_retry([
            "gh", "run", "list",
            "--repo", repo,
            "--workflow=upload_model.yml",
            "--limit", "1",
            "--json", "databaseId,status,createdAt",
        ])
        runs = json.loads(result.stdout or "[]")
        if runs:
            run_id = runs[0]["databaseId"]
            print(f"  Run detected: {run_id}")
            return run_id
        time.sleep(3)

    print("  WARNING: Could not detect the new run via `gh run list` — check the Actions tab manually")
    return -1


def wait_for_run(repo: str, run_id: int) -> bool:
    """gh run watch blocks until the run finishes and exits non-zero on failure."""
    print(f"  Watching run {run_id} (this blocks until it finishes)...")
    result = subprocess.run(
        ["gh", "run", "watch", str(run_id), "--repo", repo, "--exit-status"],
        text=True,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth",        required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--pkl",        required=True, help="Path to scaler .pkl file")
    parser.add_argument("--model-name", required=True, help="Model key, e.g. bear_6h or bull_48h")
    parser.add_argument("--repo",       required=True, help="GitHub repo: USER/REPONAME")
    args = parser.parse_args()

    if not os.path.exists(args.pth):
        print(f"ERROR: .pth file not found: {args.pth}")
        sys.exit(1)
    if not os.path.exists(args.pkl):
        print(f"ERROR: .pkl file not found: {args.pkl}")
        sys.exit(1)

    # Confirm gh is installed and authenticated before doing anything else
    auth_check = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if auth_check.returncode != 0:
        print("ERROR: gh CLI is not installed or not logged in. Run `gh auth login` first.")
        sys.exit(1)

    model_dir = f"model/{args.model_name}"
    os.makedirs(model_dir, exist_ok=True)
    dest_pth = f"{model_dir}/best_regime_transformer.pth"
    dest_pkl = f"{model_dir}/scaler_X.pkl"

    print(f"\n=== Step 1: Stage files under {model_dir}/ ===")
    shutil.copy(args.pth, dest_pth)
    shutil.copy(args.pkl, dest_pkl)
    print(f"  {dest_pth}: {os.path.getsize(dest_pth):,} bytes")
    print(f"  {dest_pkl}: {os.path.getsize(dest_pkl):,} bytes")

    print("\n=== Step 2: Temporarily add model files to git (force — bypasses .gitignore) ===")
    committed = stage_and_commit(
        [dest_pth, dest_pkl],
        f"temp: add {args.model_name} model files for upload",
    )
    # Push regardless of whether a new commit was made — if this is a repeat
    # run after an earlier crash, there may be a local commit that was never
    # pushed. `git push` is a safe no-op ("Everything up-to-date") if not.
    run(["git", "push", "origin", "main"])
    if not committed:
        print("  (proceeding to Step 3 with the already-committed files)")

    print(f"\n=== Step 3: Trigger upload_model workflow (model_name={args.model_name}) ===")
    run_id = trigger_workflow(args.repo, args.model_name)

    print("\n=== Step 4: Wait for upload to complete ===")
    if run_id > 0:
        success = wait_for_run(args.repo, run_id)
        if not success:
            print("  WARNING: Upload run finished with a failure — check the Actions tab before continuing.")
            print("  Not removing the model files from git until you confirm the artifact actually uploaded.")
            sys.exit(1)
    else:
        print("  Skipping wait — check the Actions tab manually, then re-run this script to finish Step 5,")
        print("  or run Step 5 by hand (see the git rm commands in the script).")
        sys.exit(0)

    print("\n=== Step 5: Remove model files from git history ===")
    run(["git", "rm", "--cached", dest_pth, dest_pkl])
    os.remove(dest_pth)
    os.remove(dest_pkl)
    run(["git", "commit", "-m", f"chore: remove {args.model_name} model files from git (stored as artifact)"])
    run(["git", "push", "origin", "main"])

    print("\n✅ Done!")
    print(f"   '{args.model_name}' model files are now stored as artifact regime-model-{args.model_name}.")
    print("   They are NOT in git history.")
    print("   The monitor workflow will download them automatically each run.")
    print(f"\n   Verify at: https://github.com/{args.repo}/actions")


if __name__ == "__main__":
    main()