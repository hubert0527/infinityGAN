GPUs="0,1,2,3,4,5,6,7"
exp_name="v20N-ID354-RandY-DivAngZ-F197P101S11-L4C3SSC256-VCut3-VC10"

echo "Downloading ckpt..."
mkdir -p logs/${exp_name}/ckpt/
scp hylee@vllab5.ucmerced.edu:/home/hubert/infinityGAN/infinityGAN/logs/${exp_name}/ckpt/best_fid.pth.tar ./logs/${exp_name}/ckpt/best_fid.pth.tar

CUDA_VISIBLE_DEVICES="$GPUs" python eval_fids.py ./configs/$exp_name.yaml --type="scaleinv" --bs=128 --scale=16

