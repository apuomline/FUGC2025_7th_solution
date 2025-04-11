# Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation 

## DL-ICG团队第7名解决方案 [比赛链接](https://www.codabench.org/competitions/4781/) [Github链接](https://github.com/maskoffs/Fetal-Ultrasound-Grand-Challenge)

### 解决方法概述
我们首先采用 UniMatch 半监督学习方法，利用 10 张带标注图像作为验证集，并将 40 张已标注图像与 450 张未标注图像结合用于模型训练。随后，运用训练好的模型对未标注图像进行推理，生成伪标签，并通过手动筛选将部分高质量伪标签纳入训练集。接着，我们使用全监督方法对模型进行进一步训练。此过程将不断重复，直至积累足够数量的高质量伪标签。

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
1. 从此下载原训练集，并放到./inputs/train目录下，用于进行半监督训练。我们提供了第二阶段全监督训练时所使用到的数据集(包含伪标签)。从此处下载数据，并存放到./input/train_50_pse_374_26目录下。22
2. 我们使用的预训练模型 PVT_V2_b1 ResNet34d ResNet34 (模型预训练权重已经放在./model_pth)

### 训练模型
1. **semi_supervised_unimatch.py**为半监督训练代码，**supervised_train.py**为全监督训练代码。
2. 全监督训练: python supervised_train.py --config ./configs/pvt_fugc.yaml --data_path ./inputs/train_50_pse_374_26 --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt \
  --test_data_txt  ./inputs/train_50_pse_374_26/val_images40.txt --save_path **your training save path**
3. 半监督训练: python semi_supervised_unimatch.py  --config ./configs/pvt_fugc.yaml  --save_path **your training save path**  --train_unlabeled_path ./inputs/train/unlabeled_data \
   --train_labeled_path ./inputs/train/labeled_data --train_unlabeled_txt_path ./inputs/train/train_unlabeled.txt --train_labeled_txt_path ./inputs/train/train_labeled.txt --test_labeled_path \
   ./inputs/train/labeled_data --test_labeled_txt_path ./inputs/train/test_labeled.txt
4. 我们最优训练权重：[半监督训练]()，[全监督训练]()
5. 
### Docker运行方法

1. 解压miccaii2d_docker.zip文件到本地，若里面没有miccaii2d.tar镜像文件，请从上面的Docker链接中下载并保存到miccaii2d_docker文件中
2. 在miccaii2d_docker文件内打开git bash
3. 执行 `mkdir outputs`，`outputs`文件用于存放最后的预测结果
4. 执行 `docker load --input miccaii2d.tar`，加载镜像
5. 执行 `docker run -it --name="miccaii2d_container" --gpus=all miccaii2d`，运行容器，会自动执行推理脚本`infer.sh`
6. 执行 `docker cp miccaii2d_container:/infers_fusai/ outputs`，将docker容器内的结果导出到`outputs`文件得到最终预测结果

### 文件目录说明

1. `fusai_image` 文件存放复赛测试集
2. `infers_fusai` 为排行榜上0.9621的预测结果
3. `save_model` 文件包含模型权重
4. `outputs` 文件初始为空，执行完上面的命令后会有跟`infers_fusai`中一样的结果

### 硬件要求

GPU显存16G以上 cuda11.3以上

### 前言

主要用的是`segmentation-models-pytorch`这一个语义分割库，这个库对新手非常友好，内置了许多主流的Backbone和SegHead。其实目前工业界主流的还是用`mmsegmentation`，这个库基于`mmcv`构建，模型更加全面，但是这个库的AIP接口太高级了，改动起来有点麻烦，对于新手不是很友好。

### 初赛模型
