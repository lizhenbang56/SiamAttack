source activate pytorch1.4_python3.7
export PYTHONPATH=/home/etvuz/projects/adversarial_attack/video_analyst:$PYTHONPATH
cd /home/etvuz/projects/adversarial_attack/video_analyst

loop_num=8192
patch_size=64
python main/test.py --loop_num=$loop_num --patch_size $patch_size

python main/eval_origin.py --loop_num=$loop_num