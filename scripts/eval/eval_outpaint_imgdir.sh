infinityGAN_root="~/infinityGAN/infinityGAN"

exp_name="v20N-ID378IOF-RandY-DivAngZ-F197P101S11-L4C3SSC256-VCut3-VC10"
test_name="inversion-eval-IOF_128x128"
python eval_outpaint_imgdir.py \
 --fake-dir=${infinityGAN_root}/logs/${exp_name}/test/${test_name}/imgs/inv_raw/ \
 --real-dir=${infinityGAN_root}/logs/${exp_name}/test/${test_name}/imgs/real_gt

exp_name="v20N-ID379IOP-RandY-DivAngZ-F197P101S11-L4C3SSC256-VCut3-VC10"
test_name="inversion-eval-IOP_128x128"
python eval_outpaint_imgdir.py \
 --fake-dir=${infinityGAN_root}/logs/${exp_name}/test/${test_name}/imgs/inv_raw/ \
 --real-dir=${infinityGAN_root}/logs/${exp_name}/test/${test_name}/imgs/real_gt

echo " [Completed] Please find the results at ./logs-quant/outpaint/"