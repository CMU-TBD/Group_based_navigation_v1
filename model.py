import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1,2,2), stride=(1,2,2))   

        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        #self.pool2 = nn.MaxPool3d(2, stride=2)   
        self.pool2 = nn.MaxPool3d((1,2,2), stride=(1,2,2))   

        self.conv3_1 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv3_2 = nn.Conv3d(128, 256, 3, padding=1)
        #self.conv3 = nn.Conv3d(128, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2, stride=2)   

        self.conv4_1 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv4_2 = nn.Conv3d(256, 512, 3, padding=1)
        #self.conv4 = nn.Conv3d(256, 128, 3, padding=1)
        self.pool4 = nn.MaxPool3d(2, stride=2)   

        self.conv5_1 = nn.Conv3d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv3d(512, 512, 3, padding=1)
        #self.conv5 = nn.Conv3d(128, 64, 3, padding=1)
        self.pool5 = nn.MaxPool3d(2, stride=2)   

        #self.t_conv1 = nn.ConvTranspose2d(128, 256, 2, stride=2)
        #self.t_conv2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        #self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        #self.t_conv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        #self.t_conv5 = nn.ConvTranspose2d(64, 1, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose3d(512, 512, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose3d(128, 64, (1,2,2), stride=(1,2,2))
        self.t_conv5 = nn.ConvTranspose3d(64, 1, (1,2,2), stride=(1,2,2))

        self.conv6 = nn.Conv3d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv3d(128, 128, 3, padding=1)
        return

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        #x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        #x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        #x = F.relu(self.conv5(x))
        x = self.pool5(x)

        #x = x[:, :, -1, :, :]

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.t_conv4(x))
        x = torch.sigmoid(self.t_conv5(x))

        return x
