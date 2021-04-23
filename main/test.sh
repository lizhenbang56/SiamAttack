source activate pytorch1.4_python3.7
export PYTHONPATH=/home/etvuz/projects/adversarial_attack/video_analyst:$PYTHONPATH
cd /home/etvuz/projects/adversarial_attack/video_analyst

loop_num=4096
patch_size=64
python main/test.py --loop_num=$loop_num --patch_size $patch_size
