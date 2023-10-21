import monai
from monai.networks.blocks import ResBlock
from monai.networks.layers.factories import Dropout

densenet_model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)






### segresnet for classifcation

