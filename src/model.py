import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet model for CIFAR datasets."""
    
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        
        # Post-init assertions
        assert self.fc.weight.shape[0] == num_classes, f"Output layer mismatch: {self.fc.weight.shape[0]} vs {num_classes}"
        assert self.fc.bias is not None, "Linear layer must have bias"
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet20(num_classes=10, in_channels=3):
    """ResNet-20 model."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, in_channels=in_channels)

def resnet56(num_classes=10, in_channels=3):
    """ResNet-56 model."""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, in_channels=in_channels)

def build_model(model_cfg: DictConfig) -> nn.Module:
    """Build model from configuration."""
    
    architecture = model_cfg.get("architecture", "resnet20").lower()
    num_classes = model_cfg.get("num_classes", 10)
    input_channels = model_cfg.get("input_channels", 3)
    
    if architecture == "resnet20":
        model = resnet20(num_classes=num_classes, in_channels=input_channels)
    elif architecture == "resnet56":
        model = resnet56(num_classes=num_classes, in_channels=input_channels)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Post-init assertions
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model has no parameters"
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, "Model has no trainable parameters"
    
    for name, param in model.named_parameters():
        assert param.data is not None, f"Parameter {name} has None data"
    
    return model
