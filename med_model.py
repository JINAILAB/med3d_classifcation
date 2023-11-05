import torchvision
import torch


def get_one_model(model_name, weights):
    model = torchvision.models.get_model(model_name, weights=weights, num_classes=2)
    return model