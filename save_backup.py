import os
import shutil

src_file = "data/train.jsonl"
backup_dir = "backup"
dst_file = os.path.join(backup_dir, "train.jsonl")

os.makedirs(backup_dir, exist_ok=True)

shutil.copy2(src_file, dst_file)
print(f"Backup created at {dst_file}")
