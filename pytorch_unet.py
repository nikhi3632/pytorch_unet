#https://arxiv.org/pdf/1505.04597.pdf

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os, shutil
torch_seed = 42
torch.manual_seed(torch_seed)

def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class UNet(nn.Module): #572x572
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3)
        self.encoder1_relu1 = nn.ReLU(inplace=True)
        self.encoder1_conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.encoder1_relu2 = nn.ReLU(inplace=True)
        self.encoder1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2_conv1 = nn.Conv2d(64, 128, kernel_size=3)
        self.encoder2_relu1 = nn.ReLU(inplace=True)
        self.encoder2_conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.encoder2_relu2 = nn.ReLU(inplace=True)
        self.encoder2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3_conv1 = nn.Conv2d(128, 256, kernel_size=3)
        self.encoder3_relu1 = nn.ReLU(inplace=True)
        self.encoder3_conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.encoder3_relu2 = nn.ReLU(inplace=True)
        self.encoder3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4_conv1 = nn.Conv2d(256, 512, kernel_size=3)
        self.encoder4_relu1 = nn.ReLU(inplace=True)
        self.encoder4_conv2 = nn.Conv2d(512, 512, kernel_size=3)
        self.encoder4_relu2 = nn.ReLU(inplace=True)
        self.encoder4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, kernel_size=3)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, kernel_size=3)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)
        
        # Decoder
        self.decoder1_upsample = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1_conv1 = nn.Conv2d(1024, 512, kernel_size=3)
        self.decoder1_relu1 = nn.ReLU(inplace=True)
        self.decoder1_conv2 = nn.Conv2d(512, 512, kernel_size=3)
        self.decoder1_relu2 = nn.ReLU(inplace=True)
        
        self.decoder2_upsample = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2_conv1 = nn.Conv2d(512, 256, kernel_size=3)
        self.decoder2_relu1 = nn.ReLU(inplace=True)
        self.decoder2_conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.decoder2_relu2 = nn.ReLU(inplace=True)
        
        self.decoder3_upsample = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3_conv1 = nn.Conv2d(256, 128, kernel_size=3)
        self.decoder3_relu1 = nn.ReLU(inplace=True)
        self.decoder3_conv2 = nn.Conv2d(128, 128, kernel_size=3)
        self.decoder3_relu2 = nn.ReLU(inplace=True)
        
        self.decoder4_upsample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4_conv1 = nn.Conv2d(128, 64, kernel_size=3)
        self.decoder4_relu1 = nn.ReLU(inplace=True)
        self.decoder4_conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.decoder4_relu2 = nn.ReLU(inplace=True)
        
        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        self.feature_maps = {}
        
    def copy_and_crop(self, source_tensor, target_size):
        """
        Copy the source tensor and crop it to match the size of the target tensor.
        """
        source_copied = source_tensor.clone()
        _, _, source_height, source_width = source_tensor.size()
        target_height, target_width = target_size
        diff_h = source_height - target_height
        diff_w = source_width - target_width
        if diff_h == 0 and diff_w == 0:
            return source_copied
        return source_copied[:, :, diff_h // 2:(source_height - diff_h // 2),
                            diff_w // 2:(source_width - diff_w // 2)]

    def forward(self, x):
        print(x.shape) # torch.Size([1, 1, 572, 572])
        self.feature_maps['input'] = x
        # Encoder
        enc1_conv1 = self.encoder1_conv1(x)
        self.feature_maps['enc1_conv1'] = enc1_conv1
        enc1_relu1 = self.encoder1_relu1(enc1_conv1)
        self.feature_maps['enc1_relu1'] = enc1_relu1
        enc1_conv2 = self.encoder1_conv2(enc1_relu1)
        self.feature_maps['enc1_conv2'] = enc1_conv2
        enc1_relu2 = self.encoder1_relu2(enc1_conv2)
        self.feature_maps['enc1_relu2'] = enc1_relu2

        enc1 = self.encoder1_relu1(self.encoder1_conv1(x))
        enc1 = self.encoder1_relu2(self.encoder1_conv2(enc1))
        enc1_pool = self.encoder1_pool(enc1_relu2)
        print(enc1_pool.shape) # torch.Size([1, 64, 284, 284])
        self.feature_maps['encoder1'] = enc1_pool

        enc2 = self.encoder2_relu1(self.encoder2_conv1(enc1_pool))
        enc2 = self.encoder2_relu2(self.encoder2_conv2(enc2))
        enc2_pool = self.encoder2_pool(enc2)
        print(enc2_pool.shape) # torch.Size([1, 128, 140, 140])
        self.feature_maps['encoder2'] = enc2_pool

        enc3 = self.encoder3_relu1(self.encoder3_conv1(enc2_pool))
        enc3 = self.encoder3_relu2(self.encoder3_conv2(enc3))
        enc3_pool = self.encoder3_pool(enc3)
        print(enc3_pool.shape) # torch.Size([1, 256, 68, 68])
        self.feature_maps['encoder3'] = enc3_pool

        enc4 = self.encoder4_relu1(self.encoder4_conv1(enc3_pool))
        enc4 = self.encoder4_relu2(self.encoder4_conv2(enc4))
        enc4_pool = self.encoder4_pool(enc4)
        print(enc4_pool.shape) # torch.Size([1, 512, 32, 32])
        self.feature_maps['encoder4'] = enc4_pool

        # Bottleneck
        bottleneck = self.bottleneck_relu1(self.bottleneck_conv1(enc4_pool))
        bottleneck = self.bottleneck_relu2(self.bottleneck_conv2(bottleneck))
        print(bottleneck.shape) # torch.Size([1, 1024, 28, 28])
        self.feature_maps['bottleneck'] = bottleneck

        # Decoder
        dec1_upsampled = self.decoder1_upsample(bottleneck)
        self.feature_maps['decoder1_before_skip'] = dec1_upsampled
        # Crop the corresponding feature map from encoder 4
        enc4_crop = self.copy_and_crop(enc4, dec1_upsampled.shape[2:])
        self.feature_maps['decoder1_before_skip_enc4'] = enc4_crop
        dec1 = torch.cat([dec1_upsampled, enc4_crop], dim=1)  # Skip connection
        self.feature_maps['decoder1_after_skip'] = dec1
        dec1 = self.decoder1_relu1(self.decoder1_conv1(dec1))
        dec1 = self.decoder1_relu2(self.decoder1_conv2(dec1))
        print(dec1.shape) # torch.Size([1, 512, 52, 52])
        self.feature_maps['decoder1'] = dec1

        dec2_upsampled = self.decoder2_upsample(dec1)
        self.feature_maps['decoder2_before_skip'] = dec2_upsampled
        # Crop the corresponding feature map from encoder 3
        enc3_crop = self.copy_and_crop(enc3, dec2_upsampled.shape[2:])
        self.feature_maps['decoder2_before_skip_enc3'] = enc3_crop
        dec2 = torch.cat([dec2_upsampled, enc3_crop], dim=1)  # Skip connection
        self.feature_maps['decoder2_after_skip'] = dec2
        dec2 = self.decoder2_relu1(self.decoder2_conv1(dec2))
        dec2 = self.decoder2_relu2(self.decoder2_conv2(dec2))
        print(dec2.shape) # torch.Size([1, 256, 100, 100])
        self.feature_maps['decoder2'] = dec2

        dec3_upsampled = self.decoder3_upsample(dec2)
        self.feature_maps['decoder3_before_skip'] = dec3_upsampled
        # Crop the corresponding feature map from encoder 2
        enc2_crop = self.copy_and_crop(enc2, dec3_upsampled.shape[2:])
        self.feature_maps['decoder3_before_skip_enc2'] = enc2_crop
        dec3 = torch.cat([dec3_upsampled, enc2_crop], dim=1)  # Skip connection
        self.feature_maps['decoder3_after_skip'] = dec3
        dec3 = self.decoder3_relu1(self.decoder3_conv1(dec3))
        dec3 = self.decoder3_relu2(self.decoder3_conv2(dec3))
        print(dec3.shape) # torch.Size([1, 128, 196, 196])
        self.feature_maps['decoder3'] = dec3

        dec4_upsampled = self.decoder4_upsample(dec3)
        self.feature_maps['decoder4_before_skip'] = dec4_upsampled
        # Crop the corresponding feature map from encoder 1
        enc1_crop = self.copy_and_crop(enc1, dec4_upsampled.shape[2:])
        self.feature_maps['decoder4_before_skip_enc1'] = enc1_crop
        dec4 = torch.cat([dec4_upsampled, enc1_crop], dim=1)  # Skip connection
        self.feature_maps['decoder4_after_skip'] = dec4
        dec4 = self.decoder4_relu1(self.decoder4_conv1(dec4))
        dec4 = self.decoder4_relu2(self.decoder4_conv2(dec4))
        print(dec4.shape) # torch.Size([1, 64, 388, 388])
        self.feature_maps['decoder4'] = dec4
        
        # Output
        output = self.output(dec4)
        self.feature_maps['output'] = output
        return output

if __name__ == "__main__":
    # Open the image
    image = Image.open("cat_dog.jpeg")
    image_gray = image.convert("L")
    transform = transforms.Compose([
        transforms.Resize((572, 572)),          # Resize the image to match UNet input size
        transforms.ToTensor(),                  # Convert the image to a PyTorch tensor
    ])

    sample_input = transform(image_gray).unsqueeze(0)  # Add batch dimension

    # Instantiate the U-Net model
    model = UNet(in_channels=1, out_channels=2)
    output = model(sample_input)
    print("Output shape:", output.shape) # torch.Size([1, 2, 388, 388])
    feature_maps_dir = os.getcwd() + '/feature_maps'
    create_dir(feature_maps_dir)
    for layer_name, feature_map in model.feature_maps.items():
        feature_map = feature_map.squeeze(0)
        if len(feature_map.shape) > 2:
            feature_map = feature_map.mean(dim=0)
        feature_map_np = feature_map.detach().cpu().numpy()
        plt.imshow(feature_map_np, cmap='gray')
        plt.axis('off')
        plt.savefig(f"feature_maps/{layer_name}.png", bbox_inches='tight')
