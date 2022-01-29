import os
import sys
import shutil
from glob import glob

resolution = int(sys.argv[1])
assert "CUDA_VISIBLE_DEVICES" in os.environ, "Must specify cuda devices at run time!"
gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
n_gpus = len(gpu_ids)

# Generate script for resuming previous progress
start_idx = 0

output_script_root = os.path.join("scripts", "quali-permute", "inbetween{}".format(resolution))
if os.path.exists(output_script_root):
    shutil.rmtree(output_script_root)
os.makedirs(output_script_root)

test_cfg = "./test_configs/fused_gen_IOF246_pano_inbetween_256x{}.yaml".format(resolution)
model_cfg = "./configs/v20N-ID378IOF-RandY-DivAngZ-F197P101S11-L4C3SSC256-VCut3-VC10.yaml"
exp_name = os.path.basename(model_cfg)
if ".yaml" in exp_name:
    exp_name = exp_name.split(".yaml")[0]

assert os.path.exists(test_cfg), "Path not exists: "+test_cfg
assert os.path.exists(model_cfg), "Path not exists: "+model_cfg

num_l = 32
inv_cfg_l = "inversion_for_fusedgen_IOF246_256x{}L_256x128".format(resolution)
inv_placement_l = "{:.6f},{:.6f}".format(0.5, (128//2)/resolution)
stats_dir_l = os.path.join("./logs/", exp_name, "test", inv_cfg_l, "stats")
stats_files_l = sorted(glob(os.path.join(stats_dir_l, "*.pkl")))

num_r = 32
inv_cfg_r = "inversion_for_fusedgen_IOF246_256x{}R_256x128".format(resolution)
inv_placement_r = "{:.6f},{:.6f}".format(0.5, (resolution-128//2)/resolution)
stats_dir_r = os.path.join("./logs/", exp_name, "test", inv_cfg_r, "stats")
stats_files_r = sorted(glob(os.path.join(stats_dir_r, "*.pkl")))

n_jobs = num_l * num_r - start_idx
n_jobs_per_gpu = n_jobs // n_gpus + 1


# Generate script for filling in missing files
test_name = os.path.basename(test_cfg).split(".yaml")[0]
scan_root = os.path.join("./logs/", exp_name, "test", test_name)
fused_gen_IOF246_pano_inbetween_256x2150
skip_ids = []
if os.path.exists(scan_root):
    ps = glob(os.path.join(scan_root, "0*.png"))
    for p in ps:
        with open(p, "r") as f:
            for l in f:
                l = l.split("--override-save-idx=")[1]
                if l[-1] == "\n":
                    l = l[:-1]
                skip_ids.append(int(l))
skip_ids = set(skip_ids)

counter = 0
cur_gpu_id = 0
for idx_l in range(num_l):
    for idx_r in range(num_r):

        global_id = idx_l * num_r + idx_r
        if global_id < start_idx:
            continue
        if global_id in skip_ids:
            continue
        
        output_script_path = os.path.join(output_script_root, "{}th-gpu.sh".format(cur_gpu_id))
        with open(output_script_path, "a") as f:

            inv_rec_l = stats_files_l[idx_l]
            inv_rec_r = stats_files_r[idx_r]
            assert os.path.exists(inv_rec_l), "Path not exists: "+inv_rec_l
            assert os.path.exists(inv_rec_r), "Path not exists: "+inv_rec_r

            cmd = " ".join([
                "CUDA_VISIBLE_DEVICES={}".format(gpu_ids[cur_gpu_id]),
                "python test.py",
                "--test-config=" + test_cfg,
                "--model-config=" + model_cfg,
                "--inv-records=" + inv_rec_l + ":" + inv_rec_r,
                "--inv-placements={}:{}".format(inv_placement_l, inv_placement_r),
                "--override-save-idx={}".format(global_id),
            ])
            f.write(cmd + "\n")

        counter += 1
        if counter > n_jobs_per_gpu:
            cur_gpu_id += 1
            cur_gpu_id = min(cur_gpu_id, n_gpus-1)
