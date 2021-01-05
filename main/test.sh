dataset_name="GOT-10k_Val"
loop_num=8192
do_attack=true

if [ "$dataset_name" = GOT-10k_Val ];then
cfg=experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml
fi

source activate pytorch1.4_python3.7
cd /home/etvuz/projects/adversarial_attack/video_analyst
python main/test.py --config $cfg --dataset_name=$dataset_name --do_attack=$do_attack --loop_num=$loop_num
