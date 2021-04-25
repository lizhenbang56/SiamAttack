# 配置环境
source activate pytorch1.4_python3.7
export PYTHONPATH=/home/etvuz/projects/adversarial_attack/video_analyst:$PYTHONPATH
cd /home/etvuz/projects/adversarial_attack/video_analyst
# 配置环境

# 设置变量
loop_num=8192
patch_size=64
dataset_name="OTB_2015"  # "OTB_2015" "GOT-10k_Val"
# 设置变量

# 测试
python main/test.py --loop_num=$loop_num --patch_size $patch_size --dataset_name=$dataset_name
# 测试

# 评估
python main/eval_origin.py --loop_num=$loop_num
# 评估