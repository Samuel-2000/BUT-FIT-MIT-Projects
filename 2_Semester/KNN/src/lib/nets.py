import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet50_Weights, ResNet34_Weights
from .arg_parse import parse_arguments
from tqdm import tqdm


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))

        self.bridge = Bridge(512)

        self.decoderBlocks = nn.ModuleList([
            # DecoderBlock(2048, 1024),
            # DecoderBlock(1024, 512),
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 1)
        ])

    def forward(self, x):
        x = self.resnet(x)
        x = self.bridge(x)

        for block in self.decoderBlocks:
            x = block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, channels_into_upconv, channels_out):
        super(DecoderBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(channels_into_upconv, channels_out, kernel_size=2, stride=2)
        self.conv = ConvBlock(channels_out)

    def forward(self, inputs):
        x = self.up_conv(inputs)
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, channels):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            ConvBlock(channels),
            ConvBlock(channels)
        )

    def forward(self, x):
        return self.bridge(x)




def train(model, train_loader, optimizer, device, train_loss_file):
    model.train()
    train_loss = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for _, (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        
        loss = F.mse_loss(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        # Update progress bar description with current loss
        progress_bar.set_description(f'Train Loss: {loss.item()}')


    train_loss /= len(train_loader.dataset)
    print(f"average train loss: {train_loss}")
    return train_loss



def validate(model, val_loader, device):
    model.eval()
    val_loss = 0

    # Initialize tqdm to track progress
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = F.mse_loss(output, target)
            val_loss += loss.item()
            
            progress_bar.set_description(f'Validation Loss: {loss.item()}')

    val_loss /= len(val_loader.dataset)
    print(f"average validation loss: {val_loss}")
    return val_loss



def test(model, test_loader, device):
    model.eval()
    test_loss = 0

    # Initialize tqdm to track progress
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = F.mse_loss(output, target)
            test_loss += loss.item()
            
            progress_bar.set_description(f'Test Loss: {loss.item()}')

    test_loss /= len(test_loader.dataset)
    print(f"average test loss: {test_loss}")

    return test_loss