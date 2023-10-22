import monai
from resnet3d import ResNet3dClassification
from resnet3d_pretrained import *
import torch

class MedModel:
    def __init__(self):
        self.densenet_model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
        self.resnet_model = ResNet3dClassification()
        
        # resnet10 pretrained model
        self.resnet10_model = self.get_pretrained(resnet10(shortcut_type='B'), 'resnet10_23.pth')
        
        # resnet18_pretrained model
        self.resnet18_model = self.get_pretrained(resnet18(shortcut_type='A'), 'resnet18_23.pth')
        
        # resnet34_pretrained model
        self.resnet34_model = self.get_pretrained(resnet34(shortcut_type='A'), 'resnet34_23.pth')
        
        # resnet50_pretrained model
        self.resnet50_model = self.get_pretrained(resnet50(shortcut_type='B'), 'resnet50_23.pth')
        
        self.resnet101_model = self.get_pretrained(resnet101(shortcut_type='B'), 'resnet100.pth')
        self.resnet152_model = self.get_pretrained(resnet152(shortcut_type='B'), 'resnet152.pth')
        self.resnet200_model = self.get_pretrained(resnet200(shortcut_type='B'), 'resnet200.pth')
        
    def get_pretrained(model, pretrain_dir):
        # 1. pretrained 모델 불러오기
        pretrained_dict = torch.load(pretrain_dir)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. 존재하는 statedict만 overwrite
        model_dict.update(pretrained_dict)
        # 3. load statedict
        model.load_state_dict(model_dict)
        
        return model
        
        