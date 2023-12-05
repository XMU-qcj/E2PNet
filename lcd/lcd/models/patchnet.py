import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/media/XH-8T/qcj/E2PNet/Dataset")
from event_utils import *
class PatchNetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channel=1):
        super(PatchNetEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.input_channel = input_channel
        self.conv1 = nn.Conv2d(self.input_channel, 32, 4, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv5 = nn.Conv2d(256, 256, 4, 2, 1)
        self.conv6 = nn.Conv2d(256, embedding_size, 4, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = x.view(batch_size, -1)
        return x


class PatchNetDecoder(nn.Module):
    def __init__(self, embedding_size, output_channel = 1):
        super(PatchNetDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.deconv1 = nn.ConvTranspose2d(embedding_size, 256, 4, 4)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, output_channel, 4, 4)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.embedding_size, 1, 1)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))
        x = torch.sigmoid(self.deconv6(x))
        x = x.permute(0, 2, 3, 1)
        return x

class PatchNetAutoencoder(nn.Module):
    def __init__(self, embedding_size, normalize=True, input_channel=1, output_channel = 1, feat_compute = True):
        super(PatchNetAutoencoder, self).__init__()
        self.normalize = normalize
        self.embedding_size = embedding_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.encoder = PatchNetEncoder(embedding_size, self.input_channel)
        self.decoder = PatchNetDecoder(embedding_size, self.output_channel)
        self.feat_compute = feat_compute
        self.output = nn.Linear(512, embedding_size)
        self.output2 = nn.Conv2d(26, 13, 1)
        self.layer_norm = nn.LayerNorm(13)
    def forward(self, x):
        z = self.encode(x)

        y = self.decode(z)
        return y, z

    def encode(self, x):
        z = self.encoder(x)
        if self.normalize:
            z = F.normalize(z)
        return z

    def decode(self, z):
        y = self.decoder(z)
        return y
