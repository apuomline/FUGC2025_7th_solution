from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms
import SimpleITK as sitk


class ACDCDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/valtest.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        if self.mode == 'val':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)



def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

def read_img_ids_from_file(file_name):
    """
    从指定的txt文件中读取图片ID，并返回一个列表。
    
    :param file_name: 要读取的txt文件名
    :return: 包含图片ID的列表
    """
    img_ids = []  # 创建一个空列表来存储读取的图片ID

    # 打开文件进行读取
    with open(file_name, 'r') as file:
        for line in file:
            img_ids.append(line.strip())  # 使用 strip() 去除每行的换行符

    return img_ids


class FUGCDataset(Dataset):

    """
    We didn't use semi-supervised learning, this Dataset Class just load labelled datas.
    If you want to try semi-supervised learning to make full use of unlabelled datas, 
    please design your own Dataset Class!
    """
    def __init__(self, 
                  size,
                  data_dir=None, ###存放图像的地址，训练与验证图像地址有去别
                  transform=None, ###带标签的图像，使用简单的数据增强
                  labeled=None, ###标识当前状态是否为带标签
                  file_name=None, ###存放训练、验证 对应的图像名txt文件
                  singal_image_transform=None,
                  tensor_transform=None, ###对于没有标签的图像进行tensor转换
                  mode = None
                  ):
        
        self.mode = mode

        self.size = size
        self.singal_image_transform = singal_image_transform
        self.labeled_transform = transform  # using transform in torch!
        self.dir =  data_dir
        self.labeled = labeled
        self.img_ids = read_img_ids_from_file(file_name)  #读取图片ID列表
        self.transform = transform
        self.tensor_transform = tensor_transform 
        labels = [] if self.labeled else None
        images = [] 
        for img_id in self.img_ids:
            if self.labeled:
                image_path = os.path.join(data_dir, "images", img_id+'.png')
                label_path = os.path.join(data_dir, "labels", img_id+'.png')

                ###使用sitk加载图像
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image).transpose(2,0,1)
                label = sitk.ReadImage(label_path)
                label = sitk.GetArrayFromImage(label)

                images.append(image)
                labels.append(label)

            else:
                image_path = os.path.join(data_dir, "images", img_id+'.png')
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image).transpose(2,0,1)

                images.append(image)

        self.images = images 
        self.labels = labels if self.labeled else None
        self.length = len(self.images)
        print(f'self.length:{self.length}')

    def __len__(self):

        return self.length

    def __getitem__(self, idx):
        """
        对于原官方的代码中，也是用sitk库加载图像，然后将图像转换为PIL图像，
        随后对PIL图像做处理
        """
        img_id = self.img_ids[idx]
        if self.labeled :
            image = self.images[idx]
            label = self.labels[idx]

            if self.transform:
                image,label = self.transform(image,label)

            if self.mode =='val':
                return image,label,{"img_id":img_id}
            return image,label
        
        else:
            ###处理没有标签的情况
            image = self.images[idx]

            if self.singal_image_transform :
                image = self.singal_image_transform(image)

            image_cp = F.to_pil_image(image)
            img_s1, img_s2 = deepcopy(image_cp), deepcopy(image_cp)
            del image_cp

            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = blur(img_s1, p=0.5)

            cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)

            if self.tensor_transform:
                img_s1 =self.tensor_transform(img_s1)
          
            if random.random() < 0.8:
                img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
            img_s2 = blur(img_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)

            if self.tensor_transform:
              img_s2 = self.tensor_transform(img_s2)

            return image, img_s1, img_s2, cutmix_box1, cutmix_box2
        



###可以另外设置对应的数据加载类别
class Aug_FUGCDataset(Dataset):

    """
    We didn't use semi-supervised learning, this Dataset Class just load labelled datas.
    If you want to try semi-supervised learning to make full use of unlabelled datas, 
    please design your own Dataset Class!
    """
    def __init__(self, size=None,
                  data_dir=None, ###存放图像的地址，训练与验证图像地址有去别
                  transform=None, ###使用Albumentations的数据增强
                  labeled=None, ###标识当前状态是否为带标签
                  file_name=None, ###存放训练、验证 对应的图像名txt文件
                  singal_image_transform=None,
                  tensor_transform=None,
                  mode = None
                  ):
        
        self.mode = mode

        self.size = size
        self.singal_image_transform = singal_image_transform
        self.labeled_transform = transform  # using transform in torch!
        self.dir =  data_dir
        self.labeled = labeled
        self.img_ids = read_img_ids_from_file(file_name)  #读取图片ID列表
        self.transform = transform  # 这里传入的是Albumentations的Compose对象
        self.tensor_transform = tensor_transform 
        labels = [] if self.labeled else None
        images = [] 
        for img_id in self.img_ids:
            if self.labeled:
                image_path = os.path.join(data_dir, "images", img_id+'.png')
                label_path = os.path.join(data_dir, "labels", img_id+'.png')

                ###使用sitk加载图像
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image).transpose(2,0,1)  # 转换为CHW格式
                label = sitk.ReadImage(label_path)
                label = sitk.GetArrayFromImage(label)

                images.append(image)
                labels.append(label)

            else:
                ##如果没有标签则直接读取图像
                image_path = os.path.join(data_dir, "images", img_id+'.png')
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image).transpose(2,0,1)

                images.append(image)

        self.images = images 
        self.labels = labels if self.labeled else None
        self.length = len(self.images)
       

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        对于原官方的代码中，也是用sitk库加载图像，然后将图像转换为PIL图像，
        随后对PIL图像做处理
        """
        img_id = self.img_ids[idx]
        if self.labeled :
            image = self.images[idx]  # numpy数组，形状为 (C, H, W)
            label = self.labels[idx]  # numpy数组，形状为 (H, W)

            # 转换为HWC格式供Albumentations处理
            image_hwc = image.transpose(1, 2, 0)

            if self.transform:
                # 应用Albumentations增强，同时处理图像和标签
                augmented = self.transform(image=image_hwc, mask=label)
                image_aug = augmented['image']  # 形状 (H, W, C)，若包含ToTensorV2则为 (C, H, W)
                label_aug = augmented['mask']   # 形状 (H, W)
            else:
                # 若未应用transform，直接转换为张量
                image_aug = torch.from_numpy(image).float()
                label_aug = torch.from_numpy(label).long()

            # 若Albumentations的transform未包含ToTensorV2，则手动转换
            if isinstance(image_aug, np.ndarray):
                image_aug = torch.from_numpy(image_aug.transpose(2, 0, 1)).float()
                label_aug = torch.from_numpy(label_aug).long()

            # 应用额外的张量变换（如归一化）
            if self.tensor_transform:
                image_aug = self.tensor_transform(image_aug)
                # 标签通常不需要应用张量变换

            if self.mode =='val':
                return image_aug, label_aug, {"img_id": img_id}
            return image_aug, label_aug
        
        else:
            ###处理没有标签的情况（保持不变）
            image = self.images[idx]

            if self.singal_image_transform :
                image = self.singal_image_transform(image)

            image_cp = F.to_pil_image(image)
            img_s1, img_s2 = deepcopy(image_cp), deepcopy(image_cp)
            del image_cp

            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = blur(img_s1, p=0.5)

            cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)

            if self.tensor_transform:
                img_s1 =self.tensor_transform(img_s1)
          
            if random.random() < 0.8:
                img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
            img_s2 = blur(img_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)

            if self.tensor_transform:
              img_s2 = self.tensor_transform(img_s2)

            return image, img_s1, img_s2, cutmix_box1, cutmix_box2






class Supervised_FUGCDataset(Dataset):

    """
    We didn't use semi-supervised learning, this Dataset Class just load labelled datas.
    If you want to try semi-supervised learning to make full use of unlabelled datas, 
    please design your own Dataset Class!
    """
    def __init__(self, 
                  data_dir=None, ###存放图像的地址，训练与验证图像地址有去别
                  transform=None, ###带标签的图像，使用简单的数据增强
                  file_name=None, ###存放训练、验证 对应的图像名txt文
                
                  ):   
        self.dir =  data_dir
        self.img_ids = read_img_ids_from_file(file_name)  #读取图片ID列表
        # print(f'self.img_ids:{self.img_ids}')
        self.transform = transform
        labels = [] 
        images = [] 

        for img_id in self.img_ids:
           
                image_path = os.path.join(data_dir, "images", img_id+'.png')
                label_path = os.path.join(data_dir, "labels", img_id+'.png')

                ###使用sitk加载图像
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image).transpose(2,0,1)
                label = sitk.ReadImage(label_path)
                label = sitk.GetArrayFromImage(label)

                images.append(image)
                labels.append(label)

        self.images = images 
        self.labels = labels
        self.length = len(self.images)
        print(f'self.length:{self.length}')

    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        """

        """
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
                image,label = self.transform(image,label)
        
        return image,label
        