import os

import torch

from models import caption
from configuration import Config

dependencies = ['torch', 'torchvision']


def v3(pretrained=False):
    config = Config()
    model, _ = caption.build_model(config)

    # relative path
    base_path = os.path.dirname(__file__)
    path = os.path.join(
        base_path, 'checkpoints', 'weight493084032.pth')

    if pretrained:
        # Absolute path
        # path = r'C:\Users\sc18co\.cache\torch\hub\checkpoints\weight493084032.pth'
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])

    return model
