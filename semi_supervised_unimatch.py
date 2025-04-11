import argparse
import logging
import os
import shutil
import pprint
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import math
from torch.optim import SGD
from torch.utils.data import DataLoader,RandomSampler
from torch.utils.tensorboard import SummaryWriter
import yaml
import torchvision.transforms as transforms
from dataset.dataset import ACDCDataset,FUGCDataset
from model.unet import UNet
from model.encoder2d_unet  import encoder2d_unet
from model.resnet_unet import ResNet_UNet
from functools import partial
from medical_util.classes import CLASSES ###
from medical_util.utils import AverageMeter, count_params, init_log, DiceLoss
from dataset.JointTransform2d import JointTransform2D,NomaskJointTransform2D,ValJointTransform2D
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
dist.init_process_group(backend='nccl',rank=0, world_size = 1)

import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', default='', type=str)
parser.add_argument('--save_path', default='',type=str )

###定义未训练集当中标注图像的路径
parser.add_argument('--train_unlabeled_path', default='',type=str )

###定义训练集当中标注图像的路径
parser.add_argument('--train_labeled_path', default='',type=str )

parser.add_argument('--train_unlabeled_txt_path', default='',type=str )
parser.add_argument('--train_labeled_txt_path', default='',type=str )

###从训练集当中带标注图像中选取10张标注图像，即./inputs/labeled_data
parser.add_argument('--test_labeled_path', default='',type=str )
parser.add_argument('--test_labeled_txt_path', default='',type=str )

parser.add_argument('--optimizer', default='AdamW',type=str )
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument("--save_epochs", nargs="+",help="A list of integers (default: [1, 2, 3])",) ###用于指定保存的checkpoint

###根据yaml文件中定义的模型名称，来实例化模型
def get_model(cfg, in_chns, class_num,model_path=None):
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

    cudnn.enabled = True
    cudnn.benchmark = True

    model_name = cfg.get("model_name", "UNet")  # 默认使用 UNet
    pred_model_path = cfg.get('pred_model_path',None)
    model = get_model(cfg, in_chns=3, class_num=cfg['nclass'],model_path=pred_model_path)


    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    ###将当前的训练配置文件拷贝到输出文件夹下
    copy_file(args.config, args.save_path)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    criterion_ce = nn.CrossEntropyLoss()

    criterion_dice = DiceLoss(n_classes=cfg['nclass']) 

    train_tf = JointTransform2D(img_size=(cfg['img_h'], cfg['img_w']),crop=None, p_flip=0.5,p_rota=0.5,long_mask=True)
   
    singal_tf = NomaskJointTransform2D(img_size=(cfg['img_h'], cfg['img_w']),crop=None, p_flip=0.5,p_rota=0.5)

    tensor_tf = transforms.Compose(
        [
        transforms.ToTensor(),
    ] )

    trainset_u = FUGCDataset(
                             size=(cfg['img_h'],cfg['img_w']),
                             data_dir=args.train_unlabeled_path,
                             labeled=False,
                             file_name=args.train_unlabeled_txt_path,
                             singal_image_transform = singal_tf, 
                             tensor_transform = tensor_tf                       
                             )

    trainset_l = FUGCDataset(
                            size=(cfg['img_h'],cfg['img_w']),
                             data_dir= args.train_labeled_path,
                             transform=train_tf,
                             labeled=True,
                             file_name=args.train_labeled_txt_path,                         
                             )

    valset = FUGCDataset(
                        size=(cfg['img_h'],cfg['img_w']),
                         data_dir=args.test_labeled_path,
                         transform=train_tf,
                         labeled=True,
                         file_name=args.test_labeled_txt_path,
                         ) 
    

    trainloader_l = DataLoader(
    trainset_l, 
    batch_size=cfg['batch_size'],
    sampler=RandomSampler(trainset_l),  
    pin_memory=True, 
    num_workers=1, 
    drop_last=True
)

    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],shuffle=True,
                               pin_memory=True, num_workers=1, drop_last=True,)
  

    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                    shuffle=True,
                                   pin_memory=True, num_workers=1, drop_last=True)


    valloader = DataLoader(valset, batch_size=cfg['batch_size'], 
        pin_memory=True, num_workers=1,
                           drop_last=False,)


    total_iters = len(trainloader_u) * cfg['epochs']
    print(f'totao_iters:{total_iters}')
    print(f"Length of trainloader_u: {len(trainloader_u)}")

    num_batches = len(trainloader_l)
    batch_size = cfg['batch_size']
    total_images = num_batches * batch_size
    print(f"Total number of images: {total_images}")

    if args.optimizer in ['Adam', 'AdamW']:  
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

        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=1, T_mult=2, eta_min=1e-6,),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)]

        lr_scheduler =  torch.optim.lr_scheduler.SequentialLR(optimizer,schedulers,milestones=[62])
    
    else:
         optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    
    previous_best_val_loss = 1.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_val_loss = checkpoint['previous_best_val_loss']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, previous_best_val_loss: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_val_loss))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
   
        
        print(f"Length of trainloader_u: {len(trainloader_u)}")  
        print(f"Length of trainloader_l: {len(trainloader_l)}")
        print(f"Length of trainloader_u_mix: {len(trainloader_u_mix)}")

        iter_l = iter(trainloader_l)
        iter_u = iter(trainloader_u)
        iter_u_mix = iter(trainloader_u_mix)
       
        ###对于zip，loader统一选择最小的 loader作为迭代次数
        for i in range(len(trainloader_u)):  # 使用 trainloader_u 的长度作为迭代次数
            try:
                # 从有监督数据加载器中获取数据
                (img_x, mask_x) = next(iter_l)
            except StopIteration:
                # 如果有监督数据加载器提前结束，重新初始化迭代器
                iter_l = iter(trainloader_l)
                (img_x, mask_x) = next(iter_l)
            
            ####从半监督当中加载无标签的图像
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2) = next(iter_u)
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _) = next(iter_u_mix)

            ###cutmix_box1,cutmix_box2其实是从img_u_s1,img_u_s2中选定的混合区域
 
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            
            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            ###这里img_u_s1,img_u_s2两个指定的混合区域设置为 未带标签的指定图像
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
           
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])###因为数据是在通道维度上进行拼接
            ###所以最后能够按照批次维度进行 ###这里的pred_u_w相当于伪标签

            pred_u_w_fp = preds_fp[num_lb:]###选取对应的真实标签的批次维度

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)
        
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]


            # mask_x_sque = torch.squeeze(mask_x,dim=1) 
            ###ce损失，对于模型的输出，将其使用Softmax处理
            loss_x = (criterion_ce(pred_x, mask_x.long()) + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1))) / 2.0

           
            ###对于伪标签的损失，只能计算dice 无法计算类别
            loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float())
            
            loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float())
            
            loss_u_w_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
                                         ignore=(conf_u_w < cfg['conf_thresh']).float())
            
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            # torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())
            
            iters = epoch * len(trainloader_u) + i
            print(f'iters:{iters}')

         
            if cfg['optimizer']=='AdamW':
              
                  lr_scheduler.step() ###组合学习率更新
            else:  ###SGD
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], iters)  # 记录学习率
         
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, 
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

    
        model.eval()
        dice_class = [0] * 2
        
        val_loss_total = 0.0
        with torch.no_grad():
            for i, (img, mask) in enumerate(valloader):
                img, mask = img.cuda(), mask.cuda()

                pred = model(img)
                val_loss = (criterion_ce(pred, mask.long()) + criterion_dice(pred.softmax(dim=1), mask.unsqueeze(1))) / 2.0
                val_loss_total += val_loss.item()

                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    if union != 0:
                        dice_class[cls - 1] += 2.0 * inter / union
                    else:
                        dice_class[cls - 1] += 0.0

        val_loss_avg = val_loss_total / len(valloader)  # 计算整个验证数据的平均损失
        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)

        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES['fugc'][cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
            logger.info('***** Evaluation ***** >>>> ValLoss {:.2f}\n'.format(val_loss_avg))
            writer.add_scalar('eval/ValLoss', val_loss_avg, epoch)
            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES['fugc'][i]), dice, epoch)

        # 保存模型的条件改为基于验证损失
        is_best = val_loss_avg < previous_best_val_loss
        previous_best_val_loss = min(val_loss_avg, previous_best_val_loss)

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_val_loss': previous_best_val_loss,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            if args.save_epochs is not None and epoch in args.save_epochs:
                save_path = os.path.join(args.save_path, f'epoch_{epoch}.pth')
                torch.save(checkpoint, save_path)
                print(f"Checkpoint saved for epoch {epoch} at {save_path}")



if __name__ == '__main__':
    main()








