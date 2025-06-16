import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 2)
        self.encoder2 = self.conv_block(2, 4)
        self.encoder3 = self.conv_block(4, 8)
        self.encoder4 = self.conv_block(8, 16)

        # 中间部分
        self.middle = self.conv_block(16, 32)

        # 解码器部分
        self.upconv4 = self.upconv_block(32, 16)
        self.decoder4 = self.conv_block(32, 16)
        self.upconv3 = self.upconv_block(16, 8)
        self.decoder3 = self.conv_block(16, 8)
        self.upconv2 = self.upconv_block(8, 4)
        self.decoder2 = self.conv_block(8, 4)
        self.upconv1 = self.upconv_block(4, 2)
        self.decoder1 = self.conv_block(4, 2)

        # 最后一层卷积用于输出特征图
        self.out_conv = nn.Conv2d(2, 1, kernel_size=1)

        # 全连接层用于将特征图转换为分类结果
        self.max = nn.MaxPool2d(4,4)
        self.fc = nn.Linear(3136, out_channels)  # 根据输入图像大小调整
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(100,out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # 中间部分
        middle = self.middle(F.max_pool2d(enc4, 2))

        # 解码器
        dec4 = self.upconv4(middle)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 输出特征图
        out = self.out_conv(dec1)
        out = self.max(out)
        # 展平特征图
        out = out.view(out.size(0), -1)  # 展平

        # 全连接层用于分类
        out = self.fc(out)
        # out = self.drop(out)
        # out = self.fc1(out)

        return out


if __name__ == '__main__':

    # 实例化模型
    image = torch.randn((4,1,224,224))
    model = UNet(in_channels=1, out_channels=2)  # 输入3通道（RGB图像），输出2类别（0和1）
    out = model(image)
    print(out.shape)
