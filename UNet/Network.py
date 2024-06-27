
import torch
import torch.nn as nn

width_base = 32

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoding (Contracting) Path
        self.enc1 = ConvBlock(in_channels, width_base)
        self.enc2 = ConvBlock(width_base, width_base*2)
        self.enc3 = ConvBlock(width_base*2, width_base*4)
        self.enc4 = ConvBlock(width_base*4, width_base*8)

        # Max Pooling
        self.pool = nn.MaxPool3d(kernel_size=2)

        # Decoding (Expansive) Path
        self.up1 = UpConv(width_base*8, width_base*4)
        self.dec1 = ConvBlock(width_base*8, width_base*4)
        self.up2 = UpConv(width_base*4, width_base*2)
        self.dec2 = ConvBlock(width_base*4, width_base*2)
        self.up3 = UpConv(width_base*2, width_base)
        self.dec3 = ConvBlock(width_base*2, width_base)

        # Output
        self.outc = nn.Conv3d(width_base, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding
        den = torch.unsqueeze(x[:,0,:,:,],1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoding
        d1 = self.dec1(torch.cat([e3, self.up1(e4)], 1))
        d2 = self.dec2(torch.cat([e2, self.up2(d1)], 1))
        d3 = self.dec3(torch.cat([e1, self.up3(d2)], 1))

        # Output
        out = self.outc(d3)*den
        return out

if __name__ == "__main__":
    # construct model
    model = UNet3D(in_channels=2, out_channels=1)

 
    print(model)

    Net_num = sum(x.numel() for x in model.parameters())
    print("Number of parameters", Net_num)

    # test forward
    x = torch.randn(1, 2, 128, 128, 128)
    output = model(x)
    print(output.shape)
