import numpy as np

import torch
import torchvision

def freeze_random_layer(args, model):
    freeze_rate = args.ensemble_config["freeze_rate"]
    freeze_num = args.ensemble_config["freeze_num"]
    
    if isinstance(model, torchvision.models.vision_transformer.VisionTransformer):
        layers = model.encoder.layers
    
    elif isinstance(model, torchvision.models.resnet.ResNet):
        bottlenecks = []
        for i in range(1,5):
            if hasattr(model, f"layer{i}"):
                bottlenecks.extend(getattr(model, f"layer{i}"))
        layers = bottlenecks
    
    elif isinstance(model, torch.nn.Module):
        raise NotImplementedError
    
    n_layers = len(layers)
    freeze_n = freeze_num if freeze_num is not None else int(freeze_rate * n_layers)
    
    # Unfreeze all
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad_(True)

    # Select layers to be frozen w/o replacement
    layers = np.random.choice(layers, freeze_n, replace=False)
    
    # Freeze the selected layers
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad_(False)