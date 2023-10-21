import monai
from resnet3d import ResNet3dClassification


densenet_model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
resnet_model = ResNet3dClassification()




