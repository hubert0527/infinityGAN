GPUs="2,5"

#exp_name="v19N-ID348B-F128"
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=16 --scale=1
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=4 --scale=2
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=4 --scale=4
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=1 --scale=8

exp_name="v20N-ID354-RandY-DivAngZ-F197P101S11-L4C3SSC256-VCut3-VC10"
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=16 --scale=1
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=16 --scale=2
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=16 --scale=4
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=48 --scale=8
CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=128 --scale=16

#exp_name="v20N-ID389S-F197P101S11-StyleGAN2-TFC"
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=16 --scale=1
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=16 --scale=2
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=16 --scale=4
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=64 --scale=8
#CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=128 --scale=16


