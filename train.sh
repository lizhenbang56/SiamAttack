cls_weight=1.0
ctr_weight=1.0
reg_weight=1.0
patch_size=64
phase=FFT

source activate pytorch1.8_python3.9
export CUDA_VISIBLE_DEVICES=1,5,6,7
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
cd $SHELL_FOLDER

python train.py --cls_weight $cls_weight \
                     --ctr_weight $ctr_weight \
                     --reg_weight $reg_weight \
                     --patch_size $patch_size \
                     --phase $phase\
                     --config experiments/siamfcpp/train/fulldata/157.yaml \
                     -r models/siamfcpp/siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl