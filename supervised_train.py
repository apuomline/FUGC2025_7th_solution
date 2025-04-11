import argparse
import logging
import os
import pprint
import torch
from torch import nn
import math
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from transformers import get_cosine_schedule_with_warmup
from model.unet import UNet
from model.encoder2d_unet import encoder2d_unet
from model.resnet_unet import ResNet_UNet
from medical_util.classes import CLASSES
from medical_util.utils import AverageMeter, count_params, init_log, DiceLoss
import shutil
from dataset.dataset import Supervised_FUGCDataset
from dataset.JointTransform2d import JointTransform2D
import random
import numpy as np
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')

parser.add_argument('--config', default='/mnt/workspace/UniMatch/more-scenarios/medical/configs/resnet_fugc.yaml', type=str)

parser.add_argument('-data_path', default='/mnt/workspace/UniMatch/more-scenarios/medical/inputs/train_50_pse_374_26',type=str )
parser.add_argument('--train_data_txt', default='/mnt/workspace/UniMatch/more-scenarios/medical/inputs/train_50_pse_374_26/train_images410.txt',type=str )
parser.add_argument('--test_data_txt', default='/mnt/workspace/UniMatch/more-scenarios/medical/inputs/train_50_pse_374_26/val_images40.txt',type=str )

parser.add_argument('--save_path', default='/mnt/workspace/UniMatch/more-scenarios/medical/supervised/model_test_3',type=str )
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument( "--save_epochs", nargs="+",  type=int,  default=[],  help="A list of integers (default: [1, 2, 3])",)


"""
more-scenarios\medical\supervised\train_50_pse_374_26_19 已训练完成，用于测试自己手动标注缺失部分。
train_50_pse_374_26_arrange\resnet34d_150epochs 测试数据不随机划分对模型分割效果是否有影响？
train_50_pse_374_26\convnext_small_150epochs 测试数据convnext_small模型效果
train_50_pse_374_26_classes12_model_pred\pvt_b1_150epochs 测试新增模型推理出来的classes12对应的标注图像是有用

"""


###根据yaml文件中定义的模型名称，来实例化模型
def get_model(cfg, in_chns, class_num,):
    """
    根据模型名称动态加载模型
    :param model_name: 模型名称
    :param in_chns: 输入通道数
    :param class_num: 分类类别数
    :return: 模型实例
    """
    # 将模型名称统一转换为小写
    model_name = cfg['model_name'].lower()
    encoder_name = cfg['encoder_name'].lower()
    pred_model_path = cfg['pred_model_path']
    if "resnet" in model_name:
        model = ResNet_UNet(model_name, pred_model_path,in_chns=in_chns, class_num=class_num)

    elif "pvt" in model_name:
        model = encoder2d_unet(model_name, pred_model_path,in_chns=in_chns, class_num=class_num)

    elif "smp" in model_name:
        model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights="imagenet", 
            in_channels=in_chns, 
            classes=class_num
        )

    elif model_name == "unet":
        model = UNet(model_name,in_chns=in_chns, class_num=class_num)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported model names should contain 'resnet', 'pvt', 'smp', or be 'UNet'.")
    
    return model


def copy_file(src_file_path, dest_folder_path):
    """
    将指定路径的文件拷贝到指定的文件夹下

    :param src_file_path: 源文件的完整路径
    :param dest_folder_path: 目标文件夹的路径
    """
    # 检查源文件是否存在
    if not os.path.isfile(src_file_path):
        print(f"源文件不存在：{src_file_path}")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)
        print(f"目标文件夹不存在，已创建：{dest_folder_path}")

    # 获取源文件的文件名
    file_name = os.path.basename(src_file_path)

    # 构造目标文件的完整路径
    dest_file_path = os.path.join(dest_folder_path, file_name)

    # 拷贝文件
    try:
        shutil.copy(src_file_path, dest_file_path)
        print(f"文件已成功拷贝到：{dest_file_path}")
    except Exception as e:
        print(f"拷贝文件时发生错误：{e}")

def main():
    
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "rb"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank =0 

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': 1}  # ngpus设置为1
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    copy_file(args.config, args.save_path)


    cudnn.enabled = True
    cudnn.benchmark = True

    # seed = 42  # 可以设置为任意整数值
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    model = get_model(cfg, in_chns=3, class_num=cfg['nclass'],)
    


    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))



    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()


    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    

    train_tf = JointTransform2D(img_size=(cfg['img_h'], cfg['img_w']),crop=None, p_flip=0.5,p_rota=0.5,
                                color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                                long_mask=True)
    
    ###对于验证图像不做任何增强 直接转换为 tensor
    val_tf = JointTransform2D(img_size=(cfg['img_h'], cfg['img_w']),crop=None, p_flip=0.0,p_rota=0.0,
                              color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                                long_mask=True)

    trainset = Supervised_FUGCDataset(
                            data_dir=args.data_path,
                             file_name=args.train_data_txt,
                             transform = train_tf)

    valset = Supervised_FUGCDataset(
                        data_dir=args.data_path,
                             file_name=args.test_data_txt,
                             transform = val_tf,                            
                         ) ####val_tf--->img_size :336,544


    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True,
                            )
    
    valloader = DataLoader(valset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1,
                           drop_last=False,
                          )

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    lowest_val_loss = float('inf')
    lowest_val_loss_epoch = -1 

    num_batches = len(trainloader)
    batch_size = cfg['batch_size']
    total_images = num_batches * batch_size
    print(f"Total number of images: {total_images}")

    if cfg['optimizer'] in ['Adam', 'AdamW']:  # 兼容新旧配置
    # 计算总训练步数(按batch计算)
        num_batches_per_epoch = num_batches  # 假设num_batches是每个epoch的批次数
        max_train_steps = cfg['epochs'] * num_batches_per_epoch
        
        # 设置warmup参数（这里使用1个epoch进行预热）
        warmup_epochs = 1
        warmup_steps = math.ceil(warmup_epochs * num_batches_per_epoch)

        # 初始化AdamW优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['lr'] * 0.01,      # 保持原学习率缩放
            weight_decay=0.01,        # 权重衰减
            betas=(0.9, 0.999),       # 默认动量参数
            eps=1e-8
        )

        # 创建带warmup的余弦退火调度器
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps
        )

    
    else:
         optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = (criterion_ce(pred, mask) + criterion_dice(pred.softmax(dim=1), mask.unsqueeze(1).float())) / 2.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            if cfg['optimizer'] == 'AdamW':
                    scheduler.step()

            else:
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        model.eval()
        dice_class = [0] * 2
        val_loss_total = 1.0
        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()
                pred = model(img)

                val_loss = (criterion_ce(pred, mask.long()) + criterion_dice(pred.softmax(dim=1), mask.unsqueeze(1))) / 2.0
                val_loss_total += val_loss.item()

                pred = pred.argmax(dim=1).unsqueeze(0)

               
                for cls in range(1, cfg['nclass']):
                 
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union

        val_loss_avg = val_loss_total / len(valloader)  # 计算整个验证数据的平均损失
        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)
        
        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
            logger.info('***** Evaluation ***** >>>> ValLoss {:.2f}\n'.format(val_loss_avg))
            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            writer.add_scalar('eval/ValLoss', val_loss_avg, epoch)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], iters)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)

        
        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)

        # 检查是否是最低的验证损失
        is_lowest_val_loss = val_loss_avg < lowest_val_loss
        if is_lowest_val_loss:
            lowest_val_loss = val_loss_avg
            lowest_val_loss_epoch = epoch

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
            'lowest_val_loss': lowest_val_loss,
            'lowest_val_loss_epoch': lowest_val_loss_epoch,
        }

        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        if is_lowest_val_loss:
            torch.save(checkpoint, os.path.join(args.save_path, 'lowest_val_loss.pth'))

        logger.info(f'Lowest validation loss: {lowest_val_loss:.4f} at epoch {lowest_val_loss_epoch}')

        if epoch in args.save_epochs:
                save_path = os.path.join(args.save_path, f'epoch_{epoch}.pth')
                torch.save(checkpoint, save_path)
                print(f"Checkpoint saved for epoch {epoch} at {save_path}")


if __name__ == '__main__':
    main()
