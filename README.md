# Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation 

## DL-ICG团队第7名解决方案 [比赛链接](https://www.codabench.org/competitions/4781/) [Github链接](https://github.com/maskoffs/Fetal-Ultrasound-Grand-Challenge)

### 解决方法概述
我们的方法分为两阶段。具体来说，第一阶段，我们首先采用 UniMatch 半监督学习方法，利用 10 张带标注图像作为验证集，并将 40 张已标注图像与 450 张未标注图像结合用于模型训练。随后，运用训练好的模型对未标注图像进行推理，生成伪标签，并通过手动筛选将部分高质量伪标签纳入训练集。第二阶段，我们使用全监督方法对模型进行进一步训练。此过程将不断重复，直至积累足够数量的高质量伪标签。

### 模型结果
模型采用UNet结构。我们最终预测模型，是选择PVT_v2_b1和ResNet34d平均集成方法。测试分数如下：
| model_name  |  Dice  | hd95  | time  |
|------|------|------|------|
| PVT_v2_b1 + ResNet34d | 0.8518 | 58.8085  |349.5664 |

### 环境配置
使用英伟达单卡4090，显存24G训练，python环境建议3.10。使用pip install -r requirements.txt安装第三方库。
### 项目目录
```
project-root/
├── configs/ 存放训练模型的配置文件
├── dataset/ 
├── figs/ 
├── inputs/ 存放训练数据集
├── medical_util/ 
├── model/ 模型定义代码
├── model_pth/ 预训练模型权重
├── trained_model_path/ 我们训练的最优模型权重
├── util/
├── LICENSE
├── model.py 竞赛平台的推理代码
├── requirements.txt
├── semi_supervised_unimatch.py 半监督训练代码
└── supervised_train.py 全监督训练代码
```

### 数据集和预训练模型
1. 从此下载[原训练集]()，并放到./inputs/train目录下，用于进行半监督训练。我们提供了第二阶段全监督训练时所使用到的数据集(包含伪标签)。从此处下载[全监督训练数据]()，并存放到./input/train_50_pse_374_26目录下。
2. 我们使用的预训练模型 PVT_V2_b1 ResNet34d ResNet34 (模型预训练权重已经放在./model_pth)

### 训练模型
1. **semi_supervised_unimatch.py**为半监督训练代码，**supervised_train.py**为全监督训练代码。
2. **全监督训练**:
 ```bash
   python supervised_train.py --config ./configs/pvt_fugc.yaml --data_path ./inputs/train_50_pse_374_26 --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt 
  --test_data_txt ./inputs/train_50_pse_374_26/val_images40.txt --save_path your training save path
```

4. **半监督训练**:
 ```bash
python semi_supervised_unimatch.py --config ./configs/pvt_fugc.yaml --save_path your training save path --train_unlabeled_path ./inputs/train/unlabeled_data \
   --train_labeled_path ./inputs/train/labeled_data --train_unlabeled_txt_path ./inputs/train/train_unlabeled.txt --train_labeled_txt_path ./inputs/train/train_labeled.txt --test_labeled_path \
   ./inputs/train/labeled_data --test_labeled_txt_path ./inputs/train/test_labeled.txt
```

### 复现我们的结果

#### (全监督训练)
##### 步骤 1: 下载我们提供的全监督训练数据集(50张带标注图像和我们手动筛选的高质量伪标签400张)，并将数据放在./inputs/train_50_pse_374_26目录下
##### 步骤 2: 执行训练代码
###### 训练 PVT_v2_b1_UNet 模型

确保 `pvt_fugc.yaml` 文件中的配置：
- `epochs` 设置为 `150`
- `model_name` 设置为 `pvt_v2_b1`
- `pred_model_path` 设置为 `./model_path/pvt_v2_b1_feature_only.pth`
  
使用以下命令启动训练过程：
```bash
python supervised_train.py \
  --config ./configs/pvt_fugc.yaml \
  --data_path ./inputs/train_50_pse_374_26 \
  --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt \
  --test_data_txt ./inputs/train_50_pse_374_26/val_images40.txt \
  --save_path your_training_save_path
```
###### 训练 ResNet34d_UNet 模型

确保 `resnet_fugc.yaml` 文件中的配置：
- `epochs` 设置为 `150`
- `model_name` 设置为 `resnet34d`
- `pred_model_path` 设置为 `./model_path/resnet34d_feature_only.pth`
  
使用以下命令启动训练过程：
```bash
python supervised_train.py \
  --config ./configs/resnet_fugc.yaml \
  --data_path ./inputs/train_50_pse_374_26 \
  --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt \
  --test_data_txt ./inputs/train_50_pse_374_26/val_images40.txt \
  --save_path your_training_save_path
```

##### 步骤 3: 执行竞赛平台预测代码
将全监督训练得到的权重放到./trained_model_path路径下，并更改model.py文件中的权重加载路径。也可以直接使用我们在./trained_model_path目录下提供的用于最终测试的模型权重。即pvt_b1_latest.pth 和 resnet34d_latest.pth
```bash
python model.py 
```

#### (半监督训练)
注意：由于我们在训练过程中，忘记固定随机种子。因此，执行半监督训练出来的最优模型其对应的最优epochs可能不同。在我们训练中，PVT_v2_b1_UNet最好的epoch是20，而ResNet34_UNet最好的epoch是60。因此，我们建议直接使用我们已经训练过的权重[半监督训练]()。但是，这只是半监督训练过程得到的权重，还需要将PVT_v2_b1_UNet和ResNet34_UNet进行平均集成，然后去推理伪标签。还必须保证选取伪标签与我们的一致。所以，我们建议使用我们筛选出来的伪标签进行全监督训练。[全监督训练数据]()
##### 训练 PVT_v2_b1_UNet 模型

###### 步骤 1: 准备配置文件
确保 `pvt_fugc.yaml` 文件中的配置如下：
- `model_name` 设置为 `pvt_v2_b1`
- `pred_model_path` 设置为 `./model_path/pvt_v2_b1_feature_only.pth`

###### 步骤 2: 运行训练脚本
使用以下命令启动训练过程：
```bash
python semi_supervised_unimatch.py \
  --config ./configs/pvt_fugc.yaml \
  --save_path your_training_save_path \
  --train_unlabeled_path ./inputs/train/unlabeled_data \
  --train_labeled_path ./inputs/train/labeled_data \
  --train_unlabeled_txt_path ./inputs/train/train_unlabeled.txt \
  --train_labeled_txt_path ./inputs/train/train_labeled.txt \
  --test_labeled_path ./inputs/train/labeled_data \
  --test_labeled_txt_path ./inputs/train/test_labeled.txt
```


##### 训练 ResNet34_UNet 模型

###### 步骤 1: 准备配置文件
确保 `resnet_fugc.yaml` 文件中的配置如下：
- `model_name` 设置为 `resnet34`
- `pred_model_path` 设置为 `./model_path/resnet34_feature_only.pth`

###### 步骤 2: 运行训练脚本
使用以下命令启动训练过程：
```bash
python semi_supervised_unimatch.py \
  --config ./configs/resnet34_fugc.yaml \
  --save_path your_training_save_path \
  --train_unlabeled_path ./inputs/train/unlabeled_data \
  --train_labeled_path ./inputs/train/labeled_data \
  --train_unlabeled_txt_path ./inputs/train/train_unlabeled.txt \
  --train_labeled_txt_path ./inputs/train/train_labeled.txt \
  --test_labeled_path ./inputs/train/labeled_data \
  --test_labeled_txt_path ./inputs/train/test_labeled.txt
```

