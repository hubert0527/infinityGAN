import os
import shutil
import socket
import subprocess
from tqdm import tqdm
from glob import glob

import urllib.request

baseline_ckpt_rmt = "http://vllab1.ucmerced.edu/~hubert/shared_files/v19N-ID348B-F128.pth.tar"
baseline_ckpt_stash = "./depricated_logs/v19N-ID348B-F128/ckpt/best_fid.pth.tar"
baseline_ckpt_active = "./logs/v19N-ID348B-F128/ckpt/best_fid.pth.tar"


if socket.gethostname() == "OuO": # Local debugging
    ours_exp = "v19N-ID326-CDropV2-RandY-DivAngZ-F197P101S11-L8C3-VCut3-VC10"
else:
    ours_exp = "v20N-ID354-RandY-DivAngZ-F197P101S11-L4C3SSC256-VCut3-VC10"
baseline_exp = "v19N-ID348B-F128"


GPU_CFG = {
    "baseline.yaml": ["0", baseline_exp],
    "naive.yaml": ["0", ours_exp],
    "parabatch2.yaml": ["0", ours_exp],
    "parabatch4.yaml": ["0", ours_exp],
    "parabatch8.yaml": ["0", ours_exp],
    "parabatch16.yaml": ["0", ours_exp],
    "parabatch32.yaml": ["0,1", ours_exp],
    "parabatch64.yaml": ["0,1,2,3", ours_exp],
    "parabatch128.yaml": ["0,1,2,3,4,5,6,7", ours_exp],
}

if __name__ == "__main__":

    if not os.path.exists(baseline_ckpt_stash) and not os.path.exists(baseline_ckpt_active):
        print(" [*] Downloading checkpoint, may take a while...")
        os.makedirs(os.path.dirname(baseline_ckpt_stash))
        urllib.request.urlretrieve(baseline_ckpt_rmt, baseline_ckpt_stash)

    # Compose paths
    ours_stash_dir = os.path.join("./depricated_logs/", ours_exp)
    ours_active_dir = os.path.join("./logs/", ours_exp)
    baseline_stash_dir = os.path.join("./depricated_logs/", baseline_exp)
    baseline_active_dir = os.path.join("./logs/", baseline_exp)

    # Move stashed ckpt to active
    if os.path.exists(ours_stash_dir):
        shutil.move(ours_stash_dir, ours_active_dir)
    else:
        assert os.path.exists(ours_active_dir)
    if os.path.exists(ours_stash_dir):
        shutil.move(baseline_stash_dir, baseline_active_dir)
    else:
        assert os.path.exists(baseline_active_dir)

    all_configs = sorted(glob("./benchmark_configs/*"))[::-1]
    for benchmark_config in all_configs:

        for k,v in GPU_CFG.items():
            if k in benchmark_config:
                gpu_ids, selected_exp = v
                break
        model_config = os.path.join("./configs/", selected_exp+".yaml")

        job_env = os.environ.copy()
        job_env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        print(" [*] Running new exp: {}".format(benchmark_config))
        # So run subprocess.Popen, which creates legit process, which has access to GPUs
        proc = subprocess.Popen([
            "python", "test.py",
            "--model-config", model_config,
            "--test-config", benchmark_config,
            "--speed-benchmark",
        ], env=job_env) 
        proc.communicate()

    # Move active back to stash
    shutil.move(ours_active_dir, ours_stash_dir)
    shutil.move(baseline_active_dir, baseline_stash_dir)
