# Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation 

## The DL-ICG team's 7th place solution [Competition Link](https://www.codabench.org/competitions/4781/) [Github Link](https://github.com/maskoffs/Fetal-Ultrasound-Grand-Challenge)

### Solution Overview
Our approach is divided into two stages. Specifically, in the first stage, we initially employ the UniMatch semi-supervised learning method, utilizing 10 labeled images as a validation set, and combining 40 labeled images with 450 unlabeled images for model training. Subsequently, we use the trained model to infer the unlabeled images, generate pseudo-labels, and manually select some high-quality pseudo-labels to include in the training set. In the second stage, we further train the model using a fully supervised approach. This process will be continuously repeated until a sufficient number of high-quality pseudo-labels have been accumulated.

### Model Results
The model utilizes a UNet architecture. Our final prediction model employs an averaging ensemble method between PVT_v2_b1 and ResNet34d. The test scores are as follows:：
| model_name  |  Dice  | hd95  | time  |
|------|------|------|------|
| PVT_v2_b1 + ResNet34d | 0.8518 | 58.8085  |349.5664 |

### Environment Setup
Train using a single NVIDIA 4090 card with 24GB of video memory, and it is recommended to use Python environment version 3.10. Install third-party libraries with `pip install -r requirements.txt`.
To create a virtual environment and install third-party libraries, navigate to the `apufugc2025` directory and execute the following commands:
 ```bash
conda create -n uni python=3.10
conda activate uni
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
pip install -r requirements.txt
```
### Project Directory
```
project-root/
├── configs/        Store model training configuration files
├── dataset/        The dataset directory
├── figs/           Figures directory
├── inputs/         Store training dataset
├── medical_util/   Medical utility functions
├── model/          Model definition code
├── model_pth/      Pre-trained model weights
├── trained_model_path/  Our best trained model weights
├── util/           Utility functions
├── LICENSE         License file
├── model.py        Inference code for the competition platform
├── requirements.txt  List of third-party libraries to install
├── semi_supervised_unimatch.py  Semi-supervised training code
└── supervised_train.py  Fully supervised training code
```
### Model Training
1. **semi_supervised_unimatch.py** is the semi-supervised training script, and **supervised_train.py** is the fully supervised training script.
2. **Fully Supervised Training**:
 ```bash
   python supervised_train.py --config ./configs/pvt_fugc.yaml --data_path ./inputs/train_50_pse_374_26 --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt \
  --test_data_txt ./inputs/train_50_pse_374_26/val_images40.txt --save_path your training save path
```

3. **Semi-Supervised Training**:
 ```bash
python semi_supervised_unimatch.py --config ./configs/pvt_fugc.yaml --save_path your training save path --train_unlabeled_path ./inputs/train/unlabeled_data \
   --train_labeled_path ./inputs/train/labeled_data --train_unlabeled_txt_path ./inputs/train/train_unlabeled.txt --train_labeled_txt_path ./inputs/train/train_labeled.txt --test_labeled_path \
   ./inputs/train/labeled_data --test_labeled_txt_path ./inputs/train/test_labeled.txt
```

### Replicate Our Results
Note: Due to forgetting to fix the random seed during our training process, there may be some deviation when replicating the results, but it should not be significant.
#### (Fully Supervised Training)
##### Step 1: Ensure you use the fully supervised training dataset we provided (50 labeled images and the 400 high-quality pseudo labels we manually selected), which are stored in the `./inputs/train_50_pse_374_26` directory.
##### Step 2: Execute the training code
###### Train the PVT_v2_b1_UNet model

Make sure the `pvt_fugc.yaml` file is configured with:
- `epochs` set to `150`
- `model_name` set to `pvt_v2_b1`
- `pred_model_path` set to `./model_path/pvt_v2_b1_feature_only.pth`

Start the training process with the following command:

```bash
python supervised_train.py \
  --config ./configs/pvt_fugc.yaml \
  --data_path ./inputs/train_50_pse_374_26 \
  --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt \
  --test_data_txt ./inputs/train_50_pse_374_26/val_images40.txt \
  --save_path your_training_save_path
```
###### Train the ResNet34d_UNet Model

Ensure that the `resnet_fugc.yaml` file is configured with:
- `epochs` set to `150`
- `model_name` set to `resnet34d`
- `pred_model_path` set to `./model_path/resnet34d_feature_only.pth`

Start the training process using the following command:
```bash
python supervised_train.py \
  --config ./configs/resnet_fugc.yaml \
  --data_path ./inputs/train_50_pse_374_26 \
  --train_data_txt ./inputs/train_50_pse_374_26/train_images410.txt \
  --test_data_txt ./inputs/train_50_pse_374_26/val_images40.txt \
  --save_path your_training_save_path
```

##### Step 3: Run the Competition Platform Prediction Code
Place the weights obtained from fully supervised training into the `./trained_model_path` directory, and modify the weight loading path in the `model.py` file. Alternatively, you can directly use the model weights provided in the `./trained_model_path` directory for the final test, which are `pvt_b1_latest.pth` and `resnet34d_latest.pth`.
```bash
python model.py 
```

#### (Semi-Supervised Training)
Although our final solution involves using pseudo labels for fully supervised training, we provide here the semi-supervised training process. Note: Since we forgot to fix the random seed during the training process, the optimal number of epochs for the best model obtained from semi-supervised training may differ. In our training, the best epoch for PVT_v2_b1_UNet was 20, and for ResNet34_UNet it was 60. Therefore, we recommend directly using the weights we have already trained. The storage paths are `./trained_model_pth/pvt_b1_ori_imgsize_epoch_20.pth` and `./trained_model_pth/resnet34_ori_imgsize_epoch_60.pth`. However, these are only the weights obtained from the semi-supervised training process and still require averaging and fusing the weights of PVT_v2_b1_UNet and ResNet34_UNet, followed by inferring pseudo labels. It is also necessary to ensure that the selection of pseudo labels is consistent with ours. Therefore, we recommend using the pseudo labels we have filtered for fully supervised training.

##### Train the PVT_v2_b1_UNet Model

###### Step 1: Prepare the Configuration File
Ensure that the `pvt_fugc.yaml` file is configured as follows:
- `model_name` set to `pvt_v2_b1`
- `pred_model_path` set to `./model_path/pvt_v2_b1_feature_only.pth`

###### Step 2: Run the Training Script
Start the training process with the following command:
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


##### Train the ResNet34_UNet Model

###### Step 1: Prepare the Configuration File
Make sure the `resnet_fugc.yaml` file is configured as follows:
- `model_name` is set to `resnet34`
- `pred_model_path` is set to `./model_path/resnet34_feature_only.pth`

###### Step 2: Run the Training Script
Start the training process using the following command:
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

