import torch
import os
from pvt_resnet_utils.pvt_unet import  PVT_v2_UNet
import torch.nn as nn
from pvt_resnet_utils.resnet_unet import ResNet_UNet

class Final_Model(nn.Module):
    def __init__(
        self, ):
        super().__init__()


        self.pvt_unet_b1_adamw = PVT_v2_UNet(in_chns=3,class_num=3)
      
        self.resnet34d_uent_adamw = ResNet_UNet(in_chns=3,class_num=3)


    def forward(self, x):

        # unet_seg = self.unet_384(x)
        pvt_b1_admaw_seg = self.pvt_unet_b1_adamw(x)
   
        resnet34d_unet_Seg = self.resnet34d_uent_adamw(x)

        return (pvt_b1_admaw_seg + resnet34d_unet_Seg ) / 2.0
    
class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.mean = None
        self.std = None
        self.model = Final_Model().cpu()

    def load(self, path="./trained_model_path"):
     
        pvt_b1_unet_adamw_model_path = os.path.join(path, "pvt_b1_latest.pth")
       
        resnet34d_unet_adamw_model_path = os.path.join(path, "resnet34d_latest.pth")

        try:
          
            pvt_b1_unet_admaw__model_path_checkpoint = torch.load(pvt_b1_unet_adamw_model_path,
                                                         map_location="cpu")
            
            
            resnet34d_admaw__model_path_checkpoint = torch.load(resnet34d_unet_adamw_model_path,
                                                         map_location="cpu")
             

        

            self.model.resnet34d_uent_adamw.load_state_dict(resnet34d_admaw__model_path_checkpoint['model'],strict=True)
            print(f"Model weights loaded from {resnet34d_unet_adamw_model_path}")

            self.model.pvt_unet_b1_adamw.load_state_dict(pvt_b1_unet_admaw__model_path_checkpoint['model'],strict=True)
            print(f"Model weights loaded from {pvt_b1_unet_adamw_model_path}")

            return self
        except FileNotFoundError:
            print(f"Error: The file {pvt_b1_unet_adamw_model_path} or {resnet34d_unet_adamw_model_path}  does not exist.")
            return None
        except RuntimeError as e:
            print(f"Error: Failed to load model weights - {e}")
            return None

    def predict(self, X):
        """
        X: numpy array of shape (3,336,544)
        """
        self.model.eval()
        X = X / 255.0
        image = torch.tensor(X, dtype=torch.float).unsqueeze(0)

        seg = self.model(image)  # seg (1,3,336,544)

        seg = seg.squeeze(0).argmax(dim=0).detach().numpy()  # (336,544) values:{0,1,2} 1 upper 2 lower

        return seg

    def save(self, path="./"):
        '''
        Save a trained model.
        '''
        pass


if __name__=='__main__':

    x = torch.rand(3,336,544)
    net  = model()
    net.load()
    out = net.predict(x)
    print(f'out.shape:{out.shape}')