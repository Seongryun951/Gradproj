"""Upload eigenscore output .pkl files to HuggingFace Hub."""
import os
import time
from huggingface_hub import HfApi

api = HfApi()
repo_id = "Seongryun951/eigenscore-results"
api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

output_dir = os.path.join(os.path.dirname(__file__), "output")
for shard_dir in sorted(os.listdir(output_dir)):
    shard_path = os.path.join(output_dir, shard_dir)
    if not os.path.isdir(shard_path):
        continue
    for fname in os.listdir(shard_path):
        fpath = os.path.join(shard_path, fname)
        remote_path = f"{shard_dir}/{fname}"
        for attempt in range(3):
            try:
                size_mb = os.path.getsize(fpath) / 1e6
                print(f"Uploading {remote_path} ({size_mb:.1f}MB)...")
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"  OK: {remote_path}")
                break
            except Exception as e:
                print(f"  Retry {attempt+1}/3: {e}")
                time.sleep(5)

print(f"All files uploaded to https://huggingface.co/datasets/{repo_id}")
