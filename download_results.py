"""
VESSL 실행 완료 후 HuggingFace에서 결과를 다운로드하는 스크립트.

사용법:
    /opt/anaconda3/bin/python3 download_results.py
"""
import os
from huggingface_hub import HfApi, snapshot_download

api = HfApi()
username = api.whoami()["name"]
repo_id = f"{username}/eigenscore-results"
local_dir = os.path.join(os.path.dirname(__file__), "results")

print(f"Downloading from: https://huggingface.co/datasets/{repo_id}")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
)
print(f"Done! Results saved to: {local_dir}")

for root, dirs, files in os.walk(local_dir):
    for f in files:
        if f.endswith(".pkl"):
            fpath = os.path.join(root, f)
            size_mb = os.path.getsize(fpath) / 1e6
            print(f"  {os.path.relpath(fpath, local_dir)} ({size_mb:.1f}MB)")
