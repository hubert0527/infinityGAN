import os
import time
import argparse
import traceback
import subprocess
import urllib.request
import numpy as np
import pickle as pkl
from PIL import Image
from glob import glob
from multiprocessing import Pool

from env_config import REMOTE_CKPT_URL
from utils import FileLock

GPU_MIN_MEM = 300
GPU_IN_USE = "-gpu_in_use.pkl"
ERROR_LOGS = "-error_logs.txt"

CKPTS = {
    "InfinityGAN-IOF": "http://vllab1.ucmerced.edu/~hubert/shared_files/infinityGAN/ckpt/IOF/best_fid.pth.tar",
    "InfinityGAN-IOP": "http://vllab1.ucmerced.edu/~hubert/shared_files/infinityGAN/ckpt/IOP/best_fid.pth.tar",
}


def new_experiment(kwargs):
    args = kwargs["args"]
    sample_idx = str(kwargs["idx"])
    test_name = os.path.basename(args.test_config).split(".yaml")[0]

    while True:
        with FileLock(test_name+GPU_IN_USE, timeout=np.inf, delay=1) as lock:
            gpu_usage = pkl.load(open(test_name+GPU_IN_USE, "rb"))
            found_gpu = False
            for k,v in gpu_usage.items():
                if v < args.exp_per_gpu:
                    gpu_usage[k] += 1
                    gpu_id = k
                    found_gpu = True
                    break
            if found_gpu:
                pkl.dump(gpu_usage, open(test_name+GPU_IN_USE, "wb"))
                break
            else:
                time.sleep(1)
    
    job_env = os.environ.copy()
    job_env["CUDA_VISIBLE_DEVICES"] = gpu_id

    mem_ratio = 1 / args.exp_per_gpu
    print(" [*] Submitting a sample {} to GPU id = {}".format(sample_idx, gpu_id))
    # So run subprocess.Popen, which creates legit process, which has access to GPUs
    FNULL = open(os.devnull, 'w') # suppress all outputs
    proc = subprocess.Popen([
        "python", "test.py",
        "--test-config", args.test_config,
        "--model-config", args.model_config,
        "--inv-start-idx", sample_idx,
        "--try-restrict-memory", str(mem_ratio),
    ], env=job_env, stdout=FNULL, stderr=subprocess.PIPE) 
    output, error = proc.communicate() # Blocking

    if proc.returncode != 0:
        print("[!] Error at sample_id {} (return {}):".format(sample_idx, proc.returncode))
        print(error.decode("utf-8"))
        with FileLock(test_name+ERROR_LOGS, timeout=np.inf, delay=1) as lock:
            with open(test_name+ERROR_LOGS, "a") as f:
                f.write("[!] Error at sample_id {}\n".format(sample_idx))
                f.write(error.decode("utf-8") + "\n")
                f.write("\n")

    with FileLock(test_name+GPU_IN_USE, timeout=np.inf, delay=1) as lock:
        gpu_usage = pkl.load(open(test_name+GPU_IN_USE, "rb"))
        gpu_usage[gpu_id] -= 1
        pkl.dump(gpu_usage, open(test_name+GPU_IN_USE, "wb"))


def create_jobs(args, gpu_id):
    exp_name = os.path.basename(args.model_config).split(".yaml")[0]
    test_name = os.path.basename(args.test_config).split(".yaml")[0]

    if args.idx is not None:
        args.idx = [int(v) for v in args.idx.split(",")]
        all_jobs = [
            {
                "args": args, 
                "idx": str(i),
            } for i in args.idx]
    else:
        save_root = os.path.join("logs", exp_name, "test", test_name, "stats")
        prev_result_files = glob(os.path.join(save_root, "*.pkl"))

        if args.validate:
            for path in prev_result_files:
                try:
                    pkl.load(open(path, "rb"))
                except:
                    os.remove(path)
            prev_result_files = glob(os.path.join(save_root, "*.pkl"))

        completed_exp_ids = [
            int(os.path.basename(path).split(".pkl")[0])
                for path in prev_result_files]

        job_ids = range(args.num) if args.st is None else range(args.st, args.num)

        all_jobs = [
            {
                "args": args, 
                "idx": str(i),
            } for i in job_ids if i not in completed_exp_ids]

    return all_jobs


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used,memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [
        {
            "used": int(x.split(", ")[0]), 
            "free": int(x.split(", ")[1]),
        } for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def check_gpu_availability():
    empty_gpus = []
    mem_map = get_gpu_memory_map()
    is_all_empty = True
    for gpu_id in list(mem_map.keys()):
        #used_ratio = mem_map[gpu_id]["used"] / (mem_map[gpu_id]["free"]+mem_map[gpu_id]["used"]) 
        #if used_ratio > 0.3:
        if mem_map[gpu_id]["used"] > GPU_MIN_MEM: # sometimes GPUs has awkawrd memory reservation
            mem_map.pop(gpu_id)
            is_all_empty = False
        else:
            empty_gpus.append(gpu_id)
    return empty_gpus, is_all_empty


def emission_func(a):
    img = Image.open("assets/lena.png")
    H, W = img.height, img.width
    nH, nW = 10240, 10240
    while True:
        img.resize((nH, nW), resample=Image.BICUBIC)
        img.resize((H, W), resample=Image.BICUBIC)
        time.sleep(1)
        _, is_all_empty = check_gpu_availability()
        if is_all_empty:
            return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str)
    parser.add_argument("--test-config", type=str)

    parser.add_argument("--st", type=int, default=None)
    parser.add_argument("--num", type=int, default=50000)
    parser.add_argument("--idx", type=str, default=None)

    parser.add_argument("--exp-per-gpu", type=int, default=2)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--update-ckpt", action="store_true")
    parser.add_argument("--recur", action="store_true")
    parser.add_argument("--emit-carbon-dioxide", action="store_true",
                        help="GCP automatically closes instances with low CPU usage, which forces me to emit more CO2 than they need to...")
    args = parser.parse_args()

    assert "CUDA_VISIBLE_DEVICES" in os.environ, "Must specify GPUs with CUDA_VISIBLE_DEVICES!"
    gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    n_gpus = len(gpu_ids)

    # # Get ckpt from remote
    exp_name = os.path.basename(args.model_config).split(".yaml")[0]
    exp_ckpt = os.path.join("logs", exp_name, "ckpt", "best_fid.pth.tar")
    if args.update_ckpt or (not os.path.exists(exp_ckpt)):
        # remote_url = os.path.join(REMOTE_CKPT_URL, exp_name+".pth.tar")
        remote_url = CKPTS[exp_name]
        ckpt_dir = os.path.dirname(exp_ckpt)
        if (not os.path.exists(ckpt_dir)):
            os.makedirs(ckpt_dir)
        print(" [*] Downloading checkpoint from {}".format(remote_url))
        urllib.request.urlretrieve(remote_url, exp_ckpt)
        assert os.path.exists(exp_ckpt), \
            "Cannot find ckpt at {}".format(exp_ckpt)

    # Remove prev GPU usage map, and create new GPU usage profile
    test_name = os.path.basename(args.test_config).split(".yaml")[0]
    if os.path.exists(test_name+GPU_IN_USE):
        os.remove(test_name+GPU_IN_USE)
    if os.path.exists(test_name+GPU_IN_USE + ".lock"):
        os.remove(test_name+GPU_IN_USE + ".lock")
    pkl.dump({i: 0 for i in gpu_ids}, open(test_name+GPU_IN_USE, "wb"))

    # Create dir in master thread, avoid conflicts in subprocess
    test_folder = os.path.join("logs", exp_name, "test", test_name)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    if args.emit_carbon_dioxide:
        print(" [*] Emiting CO2 to keep GCP instance alive...")
        with Pool(4) as pool:
            pool.map(emission_func, range(16))

    while True:
        # Create job templates and submit
        all_jobs = create_jobs(args, gpu_ids)
        n_workers = n_gpus * args.exp_per_gpu
        with Pool(n_workers) as pool:
            pool.map(new_experiment, all_jobs, chunksize=1)

        if args.recur:
            if len(all_jobs) == 0: 
                break
        else:
            break

    print(" [*] Complete!")
    
