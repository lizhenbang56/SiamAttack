cls_weight=0
ctr_weight=1
reg_weight=0

source activate pytorch1.4_python3.7
export CUDA_VISIBLE_DEVICES=1,2,3
cd /home/etvuz/projects/adversarial_attack/video_analyst

python main/train.py --cls_weight $cls_weight \
                     --ctr_weight $ctr_weight \
                     --reg_weight $reg_weight \
                     --config experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-trn-fulldata.yaml \
                     -r /home/etvuz/projects/adversarial_attack/video_analyst/models/siamfcpp/siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl