# Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation 

## DL-ICG团队第7名解决方案 [比赛链接](https://www.codabench.org/competitions/4781/) [Github链接](https://github.com/maskoffs/Fetal-Ultrasound-Grand-Challenge)

### 解决方法概述
我们首先采用 UniMatch 半监督学习方法，利用 10 张带标注图像作为验证集，并将 40 张已标注图像与 450 张未标注图像结合用于模型训练。随后，运用训练好的模型对未标注图像进行推理，生成伪标签，并通过手动筛选将部分高质量伪标签纳入训练集。接着，我们使用全监督方法对模型进行进一步训练。此过程将不断重复，直至积累足够数量的高质量伪标签。

1. `cd <path/to/miccai2d_docker>`
2. `docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`
3. `docker build -t miccaii2d .`
4. `docker save -o miccaii2d.tar miccaii2d` 构建完成后会有一个miccaii2d.tar文件

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
