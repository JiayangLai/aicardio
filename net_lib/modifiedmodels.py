from typing import Any

from torchvision.models.resnet import _resnet, ResNet, BasicBlock


def resnet18_2332(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('null', BasicBlock, [2, 3, 3, 2], pretrained, progress,
                   **kwargs)