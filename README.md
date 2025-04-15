# Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation 

## The DL-ICG team's 7th place solution [Competition Link](https://www.codabench.org/competitions/4781/) [Github Link](https://github.com/maskoffs/Fetal-Ultrasound-Grand-Challenge)

### Solution Overview
Our approach is divided into two stages. Specifically, in the first stage, we initially employ the UniMatch semi-supervised learning method, utilizing 10 labeled images as a validation set, and combining 40 labeled images with 450 unlabeled images for model training. Subsequently, we use the trained model to infer the unlabeled images, generate pseudo-labels, and manually select some high-quality pseudo-labels to include in the training set. In the second stage, we further train the model using a fully supervised approach. This process will be continuously repeated until a sufficient number of high-quality pseudo-labels have been accumulated.

### Model Results
The model utilizes a UNet architecture. Our final prediction model employs an averaging ensemble method between PVT_v2_b1 and ResNet34d. The test scores are as follows:：
| model_name  |  Dice  | hd95  | time  |
|------|------|------|------|
| PVT_v2_b1 + ResNet34d | 0.8518 | 58.8085  |349.5664 |

### 环境配置
Train using a single NVIDIA 4090 card with 24GB of video memory, and it is recommended to use Python environment version 3.10. Install third-party libraries with `pip install -r requirements.txt`.
To create a virtual environment and install third-party libraries, navigate to the `apufugc2025` directory and execute the following commands:
 ```bash
conda create -n uni python=3.10
conda activate uni
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
pip install -r requirements.txt
```
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
### 训练模型
1. **semi_supervised_unimatch.py**为半监督训练代码，**supervised_train.py**为全监督训练代码。
2. **全监督训练**:
 ```bash
   python supervised_train.py --config ./configs/pvt_fugc.yaml --data_path ./inputs/train_50_pse_374_26 --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt \
  --test_data_txt ./inputs/train_50_pse_374_26/val_images40.txt --save_path your training save path
```

4. **半监督训练**:
 ```bash
python semi_supervised_unimatch.py --config ./configs/pvt_fugc.yaml --save_path your training save path --train_unlabeled_path ./inputs/train/unlabeled_data \
   --train_labeled_path ./inputs/train/labeled_data --train_unlabeled_txt_path ./inputs/train/train_unlabeled.txt --train_labeled_txt_path ./inputs/train/train_labeled.txt --test_labeled_path \
   ./inputs/train/labeled_data --test_labeled_txt_path ./inputs/train/test_labeled.txt
```

### 复现我们的结果
注意：由于我们在训练过程中，忘记固定随机种子。因此，复现结果可能会有偏差，但是应该不大。
#### (全监督训练)
##### 步骤 1: 确保使用我们提供的全监督训练数据集(50张带标注图像和我们手动筛选的400张高质量伪标签)，数据存放在./inputs/train_50_pse_374_26目录下
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
尽管我们最终方案是使用伪标签进行全监督训练，我们在这里提供半监督的训练流程。注意：由于我们在训练过程中，忘记固定随机种子。因此，执行半监督训练出来的最优模型其对应的最优epochs可能不同。在我们训练中，PVT_v2_b1_UNet最好的epoch是20，而ResNet34_UNet最好的epoch是60。因此，我们建议直接使用我们已经训练过的权重。存放路径为 ./trained_model_pth/pvt_b1_ori_imgsize_epoch_20.pth 和 ./trained_model_pth/resnet34_ori_imgsize_epoch_60.pth。但是，这只是半监督训练过程得到的权重，还需要将PVT_v2_b1_UNet和ResNet34_UNet进行平均加权融合，然后去推理伪标签。还必须保证选取伪标签与我们的一致。所以，我们建议使用我们筛选出来的伪标签进行全监督训练。
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
  --config ./configs/resnet_fugc.yaml \
  --save_path your_training_save_path \
  --train_unlabeled_path ./inputs/train/unlabeled_data \
  --train_labeled_path ./inputs/train/labeled_data \
  --train_unlabeled_txt_path ./inputs/train/train_unlabeled.txt \
  --train_labeled_txt_path ./inputs/train/train_labeled.txt \
  --test_labeled_path ./inputs/train/labeled_data \
  --test_labeled_txt_path ./inputs/train/test_labeled.txt
```

