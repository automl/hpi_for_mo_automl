from codecarbon import EmissionsTracker
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.resnet import _resnet, BasicBlock
import lightning as l
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchmetrics import Accuracy

torch.set_float32_matmul_precision('medium')


class ResNetLayers(nn.Module):
    """
    ResNet layers creates a resnet with adjustabl layer sizes.
    """

    def __init__(self, layer0, layer1, layer2, layer3, num_classes, zero_init_residual):
        super().__init__()
        self.resnet = _resnet(
            BasicBlock,
            [layer0, layer1, layer2, layer3],
            weights=None,
            progress=False,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual
        )

    def forward(self, x):
        return self.resnet(x)


class LitResNet(l.LightningModule):
    """
    Lightning module for the ResNet.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.resnet = ResNetLayers(3, config['layer1'], config['layer2'], 3, 10, config['zero_init_residual'] == 1)
        self.acc = Accuracy(task='multiclass', num_classes=10)
        self.tracker = EmissionsTracker(tracking_mode='process', log_level='critical', pue=1.3)
        self.batch_size = 4096

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return F.cross_entropy(logits, y)

    def validation_step(self, batch, batch_idx):
        self.tracker.start_task('test')
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)
        emissions_data = self.tracker.stop_task('test')
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('energy', emissions_data.energy_consumed, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'],
                          eps=self.config['eps'])
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage=None):
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if self.config['augment'] == 1:
            train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(*stats, inplace=True)])
        else:
            train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

        self.cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=train_tfms)
        self.cifar10_test = CIFAR10(root='./data', train=False, transform=valid_tfms, download=True)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=4, pin_memory=True)
