import os
import shutil
from glob import glob

backup_file_types = {
    "py", "cpp", "cu",
}

backup_dirs = {
    "models",
        "custom_ops",
    "libs",
        "lpips",
    "dev",
    "test_managers",
}

excluded_dirs = {
    "configs",
    "logs",
    "data",
    "lmdb",
    "__pycache__",
    ".build_cache",
    "weights",
}

def backup_files(cur_dir, backup_dir):
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    cur_level_files = [p for p in glob(os.path.join(cur_dir, "*")) if os.path.basename(p) not in excluded_dirs]
    for path in cur_level_files:
        if os.path.isdir(path):
            dir_name = os.path.basename(path)
            if dir_name in backup_dirs:
                backup_files(path, os.path.join(backup_dir, dir_name))
        else:
            file_extension = path.split(".")[-1]
            if file_extension in backup_file_types:
                copy_path = os.path.join(backup_dir, os.path.basename(path))
                shutil.copy2(path, copy_path)

