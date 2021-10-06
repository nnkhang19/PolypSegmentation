from unet_parts import *
from attention import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.atten_inc = AttentionBlock(64)
        self.atten_down1 = AttentionBlock(128)
        self.atten_down2 = AttentionBlock(256)
        self.atten_down3 = AttentionBlock(512)
        self.atten_down4 = SelfAttention(1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5, map5 = self.atten_down4(x5)
        x4, map4 = self.atten_down3(x4)
        x3, map3 = self.atten_down2(x3)
        x2, map2 = self.atten_down1(x2)
        x1, map1 = self.atten_inc(x1)
  
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetV2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetV2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.atten_inc = AttentionBlock(64)
        self.atten_down1 = AttentionBlock(128)
        self.atten_down2 = AttentionBlock(256)
        self.atten_down3 = AttentionBlock(512)
        #self.atten_down4 = SelfAttention(1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        #self.atten_dec = SelfAttention(64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #x5, map5 = self.atten_down4(x5)
        x4, map4 = self.atten_down3(x4)
        x3, map3 = self.atten_down2(x3)
        x2, map2 = self.atten_down1(x2)
        x1, map1 = self.atten_inc(x1)
  
        out4 = self.up1(x5, x4)
        out3 = self.up2(out4, x3)
        out2 = self.up3(out3, x2)
        out1 = self.up4(out2, x1) 
        #out6 = self.atten_dec(out1)

        logits = self.outc(out1)
        return logits

if __name__ == '__main__':
    image_size = 256

    x = torch.randn(5, 3, image_size, image_size)

    model = UNetV2(3, 1)

    output = model(x)
    print(output.shape)
