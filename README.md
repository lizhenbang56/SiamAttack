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

# 可视化

## 将 txt 跟踪结果可视化

```bash
main/visualize_txt_result.py
```

## 训练过程可视化

```
videoanalyst.utils.visualize_training
```

## 搜索图像可视化

```bash
videoanalyst.utils.visualize_search_img
```


# 生成相对于原始图像的 Fake Ground Truth

```
python videoanalyst/pipeline/utils/generate_patch_anno.py
数据位于 /home/etvuz/projects/adversarial_attack/patch_anno
```

# 测试

## 测试 GOT-10k 数据库

```bash
python main/test.py --config experiments/siamfcpp/train/got10k/siamfcpp_googlenet-trn.yaml
```

## 测试 OTB 数据库

### 生成 Fake Ground Truth

```bash
[修改相应路径] python videoanalyst/pipeline/utils/generate_patch_anno.py
数据位于 /home/etvuz/projects/adversarial_attack/patch_anno/OTB_2015
```

### 使得测试代码正确读入 FGT

```bash
got_benchmark_helper.py/PipelineTracker/track():dataset_name = 'OTB_2015'
```

### 设置是否进行攻击

```bash
siamfcpp_track.py/self.do_attack=True
```

### 测试

```bash
在配置文件中指定正确的预训练模型: /home/etvuz/projects/adversarial_attack/video_analyst/models/siamfcpp/siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl
source activate pytorch1.4_python3.7
cd /home/etvuz/projects/adversarial_attack/video_analyst
python main/test.py --config experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml
```

### 评估

```bash
[修改 dataset_name] python main/eval.py
```

## 测试 LaSOT 数据库

### 生成 Fake Ground Truth

```bash
[修改相应路径] python videoanalyst/pipeline/utils/generate_patch_anno.py
数据位于 /home/etvuz/projects/adversarial_attack/patch_anno/LaSOT
```

### 使得测试代码正确读入 FGT

```bash
got_benchmark_helper.py/PipelineTracker/track():dataset_name = 'LsSOT'
```

### 设置是否进行攻击

```bash
siamfcpp_track.py/self.do_attack=True
```

### 测试

```bash
在配置文件中指定正确的预训练模型: /home/etvuz/projects/adversarial_attack/video_analyst/models/siamfcpp/siamfcpp-googlenet-got-md5_e182dc4c3823427022eccf7313d740a7.pkl
source activate pytorch1.4_python3.7
cd /home/etvuz/projects/adversarial_attack/video_analyst
python main/test.py --config experiments/siamfcpp/test/lasot/siamfcpp_googlenet-lasot.yaml
```

### 评估

```bash
[修改 dataset_name] python main/eval.py
```

# 实时评估

```bash
python main/realtime_eval.py
```