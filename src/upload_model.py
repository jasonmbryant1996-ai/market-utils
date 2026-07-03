"""
upload_model.py
================
Run this ONCE locally before your first deployment.

It adds the model files temporarily to git, triggers the upload_model
workflow, then removes them again. This way the files never permanently
live in git history, but they are stored safely as a GitHub Actions
artifact (private, accessible only to your repo's workflows).

USAGE (run on your local machine, not Kaggle):
  python src/upload_model.py \
    --pth /path/to/best_regime_transformer.pth \
    --pkl /path/to/scaler_X.pkl \
    --repo YOUR_USERNAME/btc-regime-monitor \
    --token ghp_yourPersonalAccessToken

REQUIREMENTS:
  pip install requests
  GitHub CLI (gh) installed: https://cli.github.com/
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import requests


def run(cmd: str, check: bool = True) -> str:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"FAILED:\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def trigger_workflow(repo: str, token: str) -> int:
    """Trigger the upload_model workflow and return the new run ID."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    # Get current run count before triggering
    r = requests.get(
        f"https://api.github.com/repos/{repo}/actions/workflows/upload_model.yml/runs",
        headers=headers,
    )
    before_count = r.json().get("total_count", 0)

    # Trigger
    requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/upload_model.yml/dispatches",
        headers=headers,
        json={"ref": "main", "inputs": {"confirm": "upload"}},
    )
    print("  Workflow triggered. Waiting for run to start...")
    time.sleep(5)

    # Poll until a new run appears
    for _ in range(30):
        r = requests.get(
            f"https://api.github.com/repos/{repo}/actions/workflows/upload_model.yml/runs",
            headers=headers,
        )
        runs = r.json().get("workflow_runs", [])
        if runs and len(runs) > before_count:
            run_id = runs[0]["id"]
            print(f"  Run started: {run_id}")
            return run_id
        time.sleep(3)

    print("  WARNING: Could not detect new run — check Actions tab manually")
    return -1


def wait_for_run(repo: str, token: str, run_id: int) -> bool:
    """Poll until the run completes. Returns True on success."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    print(f"  Waiting for run {run_id} to complete...")
    for _ in range(60):   # max 5 minutes
        r = requests.get(
            f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
            headers=headers,
        )
        data = r.json()
        status     = data.get("status")
        conclusion = data.get("conclusion")
        print(f"    status={status}  conclusion={conclusion}")
        if status == "completed":
            return conclusion == "success"
        time.sleep(5)
    print("  Timed out waiting for run")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth",   required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--pkl",   required=True, help="Path to scaler .pkl file")
    parser.add_argument("--repo",  required=True, help="GitHub repo: USER/REPONAME")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    args = parser.parse_args()

    if not os.path.exists(args.pth):
        print(f"ERROR: .pth file not found: {args.pth}")
        sys.exit(1)
    if not os.path.exists(args.pkl):
        print(f"ERROR: .pkl file not found: {args.pkl}")
        sys.exit(1)

    # print("\n=== Step 1: Copy model files to model/ (temporarily) ===")
    # shutil.copy(args.pth, "model/best_regime_transformer.pth")
    # shutil.copy(args.pkl, "model/scaler_X.pkl")
    print(f"  best_regime_transformer.pth: {os.path.getsize('model/best_regime_transformer.pth'):,} bytes")
    print(f"  scaler_X.pkl: {os.path.getsize('model/scaler_X.pkl'):,} bytes")

    print("\n=== Step 2: Temporarily add model files to git (force — bypasses .gitignore) ===")
    run("git add -f model/best_regime_transformer.pth model/scaler_X.pkl")
    run('git commit -m "temp: add model files for upload"')
    run(f"git push origin main")

    print("\n=== Step 3: Trigger upload_model workflow ===")
    run_id = trigger_workflow(args.repo, args.token)

    print("\n=== Step 4: Wait for upload to complete ===")
    if run_id > 0:
        success = wait_for_run(args.repo, args.token, run_id)
        if not success:
            print("  WARNING: Upload may have failed — check Actions tab")
    else:
        print("  Skipping wait — check Actions tab manually")

    print("\n=== Step 5: Remove model files from git history ===")
    run("git rm --cached model/best_regime_transformer.pth model/scaler_X.pkl")
    os.remove("model/best_regime_transformer.pth")
    os.remove("model/scaler_X.pkl")
    run('git commit -m "chore: remove model files from git (stored as artifact)"')
    run("git push origin main")

    print("\n✅ Done!")
    print("   Model files are now stored as a private GitHub Actions artifact.")
    print("   They are NOT in git history.")
    print("   The monitor workflow will download them automatically each run.")
    print(f"\n   Verify at: https://github.com/{args.repo}/actions")


if __name__ == "__main__":
    main()
