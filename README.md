# 环境

```bash
pytorch1.4_python3.7
```

# 训练

```
python main/train.py --config experiments/siamfcpp/train/got10k/siamfcpp_googlenet-trn.yaml -r /home/etvuz/projects/adversarial_attack/video_analyst/models/siamfcpp/siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl
```

# 训练所有数据
```bash
source activate pytorch1.4_python3.7
export CUDA_VISIBLE_DEVICES=1,2,3
cd /home/etvuz/projects/adversarial_attack/video_analyst
python main/train.py --config experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-trn-fulldata.yaml -r /home/etvuz/projects/adversarial_attack/video_analyst/models/siamfcpp/siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl
```

# 将 txt 跟踪结果可视化

```bash
main/visualize_txt_result.py
```

# 训练过程可视化

```
videoanalyst.utils.visualize_training
```

# 搜索图像可视化

```bash
videoanalyst.utils.visualize_search_img
```


# 生成相对于原始图像的 fake ground truth

```
python videoanalyst/pipeline/utils/generate_patch_anno.py
/home/etvuz/projects/adversarial_attack/patch_anno
```

# 测试 GOT-10k 数据库

```bash
# python main/test.py --config experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml
python main/test.py --config experiments/siamfcpp/train/got10k/siamfcpp_googlenet-trn.yaml
```

# 实时评估

```bash
python main/realtime_eval.py
```