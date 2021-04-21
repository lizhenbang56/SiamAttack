do_attack=true
dataset_name="GOT-10k_Val"
backbone_name="googlenet"  # "shufflenetv2x1_0"  #"alexnet"
loop_num=32768
cls_weight=1
ctr_weight=1
reg_weight=1

if [ "$dataset_name" = GOT-10k_Val ];then
  cfg=experiments/siamfcpp/test/got10k/siamfcpp_$backbone_name-got.yaml
fi
echo $cfg

source activate pytorch1.4_python3.7
export PYTHONPATH=/home/etvuz/projects/adversarial_attack/video_analyst:$PYTHONPATH
cd /home/etvuz/projects/adversarial_attack/video_analyst
python main/test.py --config $cfg --dataset_name=$dataset_name --do_attack=$do_attack --loop_num=$loop_num \
                    --cls_weight $cls_weight \
                    --ctr_weight $ctr_weight \
                    --reg_weight $reg_weight
python main/eval.py --dataset_name=$dataset_name --loop_num=$loop_num --backbone_name=siamfcpp_$backbone_name \
                    --cls_weight $cls_weight \
                    --ctr_weight $ctr_weight \
                    --reg_weight $reg_weight
