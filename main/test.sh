# 配置环境
source activate pytorch1.4_python3.7
export PYTHONPATH=/home/etvuz/projects/adversarial_attack/video_analyst:$PYTHONPATH
cd /home/etvuz/projects/adversarial_attack/video_analyst
# 配置环境

# 设置变量
loop_num=2048
patch_size=64
dataset_name="GOT-10k_Val"  # "OTB_2015" "GOT-10k_Val"
backbone="googlenet"  # "shufflenetv2x1_0"
phase="OURS"
# 设置变量

# 测试
python main/test.py --loop_num=$loop_num --patch_size=$patch_size --dataset_name=$dataset_name --backbone=$backbone --phase=$phase
# 测试

# # 评估
# python main/eval.py --loop_num=$loop_num --dataset_name=$dataset_name --tracker_name=$patch_size --backbone_name=$backbone
# # 评估