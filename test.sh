# 配置环境
source activate pytorch1.8_python3.9
SHELL_FOLDER=$(dirname $(readlink -f "$0"))
export PYTHONPATH=$SHELL_FOLDER:$PYTHONPATH
cd $SHELL_FOLDER
# 配置环境

# 设置变量
loop_num=256
patch_size=64
dataset_name="GOT-10k_Val"  # "OTB_2015" "GOT-10k_Val"
backbone="googlenet"  # "shufflenetv2x1_0"
phase="FFT"
# 设置变量

# 测试
python main/test.py --loop_num=$loop_num --patch_size=$patch_size --dataset_name=$dataset_name --backbone=$backbone --phase=$phase
# 测试

# # # 评估
# python eval.py --loop_num=$loop_num --dataset_name=$dataset_name --tracker_name=$phase --backbone_name=$backbone
# # # 评估