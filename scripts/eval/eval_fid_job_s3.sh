GPUs="2"
exp_name="v19N-ID348B-F128"
CUDA_VISIBLE_DEVICES="$GPUs" python eval_spatial_fids.py ./configs/$exp_name.yaml --bs=16 --scale=1

GPUs="2"
exp_name="v20N-ID354-RandY-DivAngZ-F197P101S11-L4C3SSC256-VCut3-VC10"
CUDA_VISIBLE_DEVICES="$GPUs" python eval_spatial_fids.py ./configs/$exp_name.yaml --bs=16 --scale=1


