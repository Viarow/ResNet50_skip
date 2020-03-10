import torch
import torch.nn as nn
import torch.nn.functional


def conv_flops(c_in, c_out, k_size, h, w):
    return ((k_size * k_size * c_in) * c_out + c_out) * (h * w)


class LayerGate(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels= in_channels, out_channels=10, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        # blob shape: N*10*1*1, N refers to batch size
        self.lstm = nn.LSTM(input_size=10, hidden_size=10, num_layers=1)
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.GAP(input)
        x = self.conv_1(x)
        x = self.relu(x)

        #seq_len = 1
        x = x.view(1, -1, 10)
        self.lstm.flatten_parameters()
        x_lstm, (hn, cn) = self.lstm(x)

        # x_lstm shape = 1*N*10
        x = self.fc(x_lstm.view(-1, 10))
        x = self.relu(x)
        x = self.sigmoid(x)
        output = torch.round(x)

        return output


class ChannelGate(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=3, stride=2)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv(input)
        x = self.relu(x)
        x = self.GAP(x)
        x = self.fc(torch.squeeze(x))
        x = self.relu(x)
        x = self.sigmoid(x)
        output = torch.round(x)

        return output


class AdaptConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        self.layer_gate = LayerGate(in_channels)
        self.channel_gate = ChannelGate(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride,
                              kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, input):

        layer_idx = self.layer_gate(input)
        # shape of layer_idx: N*1
        channel_idx = self.channel_gate(input)
        # shape of channel_idx: N*C

        batch_size = input.shape[0]
        output = torch.zeros(input.shape).cuda()

        for k in range(0, batch_size):

            if layer_idx[k][0]==0 or channel_idx[k].sum()==0:
                """Skip the layer or skip all channels"""
                output[k] = input[k]
            else:
                image = torch.unsqueeze(input[k],0)
                image_conv = self.conv(image)
                #image shape: 1*C*H*W
                idx = channel_idx[k].view(1,-1,1,1)
                output[k] = idx*image_conv + (1-idx)*image

        return output


class ResidualBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            AdaptConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResidualBlock.expansion, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * ResidualBlock.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * ResidualBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResidualBlock.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResidualBlock.expansion)
            )

    def forward(self, input):
        x = self.residual_function(input) + self.shortcut(input)
        output = nn.ReLU(inplace=True)(x)

        return output


class ResNet50(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = self.make_block(64, 3, stride=1)
        self.conv3 = self.make_block(128, 4, stride=2)
        self.conv4 = self.make_block(256, 6, stride=2)
        self.conv5 = self.make_block(512, 3, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, num_classes)


    def make_block(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * ResidualBlock.expansion

        return nn.Sequential(*blocks)


    def forward(self, input):
        x = self.conv1(input)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output


























