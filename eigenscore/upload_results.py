"""Upload eigenscore results tar.gz to a file sharing service."""
import os
import subprocess
import tarfile

output_dir = os.path.join(os.path.dirname(__file__), "output")
tar_path = "/tmp/eigenscore_results.tar.gz"

print("Packaging results...")
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(output_dir, arcname=".")
size_mb = os.path.getsize(tar_path) / 1e6
print(f"Archive: {tar_path} ({size_mb:.1f}MB)")

services = [
    ("0x0.st", ["curl", "-s", "-F", f"file=@{tar_path}", "https://0x0.st"]),
    ("transfer.sh", ["curl", "-s", "--upload-file", tar_path, "https://transfer.sh/eigenscore_results.tar.gz"]),
    ("file.io", ["curl", "-s", "-F", f"file=@{tar_path}", "https://file.io"]),
    ("bashupload", ["curl", "-s", "-T", tar_path, "https://bashupload.com"]),
]

for name, cmd in services:
    print(f"\nTrying {name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = result.stdout.strip()
        err = result.stderr.strip()
        print(f"  stdout: {out}")
        if err:
            print(f"  stderr: {err}")
        if out and ("http" in out.lower()):
            print(f"\n{'='*50}")
            print(f"DOWNLOAD URL ({name}):")
            print(out)
            print(f"{'='*50}")
            break
    except Exception as e:
        print(f"  Error: {e}")
else:
    print("\nAll upload services failed. Trying base64 split output...")
    import base64
    with open(tar_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    chunk_size = 76
    lines = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
    print(f"BASE64_START (total {len(lines)} lines)")
    for line in lines:
        print(line)
    print("BASE64_END")
