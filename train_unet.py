#https://arxiv.org/pdf/1505.04597.pdf

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os, shutil
import matplotlib.pyplot as plt

torch_seed = 0
torch.manual_seed(torch_seed)

def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

train_artifacts_dir = 'train_artifacts'
create_dir(train_artifacts_dir)

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
        # Encoder
        enc1 = self.encoder1_relu1(self.encoder1_conv1(x))
        enc1 = self.encoder1_relu2(self.encoder1_conv2(enc1))
        enc1_pool = self.encoder1_pool(enc1)

        enc2 = self.encoder2_relu1(self.encoder2_conv1(enc1_pool))
        enc2 = self.encoder2_relu2(self.encoder2_conv2(enc2))
        enc2_pool = self.encoder2_pool(enc2)

        enc3 = self.encoder3_relu1(self.encoder3_conv1(enc2_pool))
        enc3 = self.encoder3_relu2(self.encoder3_conv2(enc3))
        enc3_pool = self.encoder3_pool(enc3)

        enc4 = self.encoder4_relu1(self.encoder4_conv1(enc3_pool))
        enc4 = self.encoder4_relu2(self.encoder4_conv2(enc4))
        enc4_pool = self.encoder4_pool(enc4)

        # Bottleneck
        bottleneck = self.bottleneck_relu1(self.bottleneck_conv1(enc4_pool))
        bottleneck = self.bottleneck_relu2(self.bottleneck_conv2(bottleneck))

        # Decoder
        dec1_upsampled = self.decoder1_upsample(bottleneck)
        # Crop the corresponding feature map from encoder 4
        enc4_crop = self.copy_and_crop(enc4, dec1_upsampled.shape[2:])
        dec1 = torch.cat([dec1_upsampled, enc4_crop], dim=1)  # Skip connection
        dec1 = self.decoder1_relu1(self.decoder1_conv1(dec1))
        dec1 = self.decoder1_relu2(self.decoder1_conv2(dec1))

        dec2_upsampled = self.decoder2_upsample(dec1)
        # Crop the corresponding feature map from encoder 3
        enc3_crop = self.copy_and_crop(enc3, dec2_upsampled.shape[2:])
        dec2 = torch.cat([dec2_upsampled, enc3_crop], dim=1)  # Skip connection
        dec2 = self.decoder2_relu1(self.decoder2_conv1(dec2))
        dec2 = self.decoder2_relu2(self.decoder2_conv2(dec2))

        dec3_upsampled = self.decoder3_upsample(dec2)
        # Crop the corresponding feature map from encoder 2
        enc2_crop = self.copy_and_crop(enc2, dec3_upsampled.shape[2:])
        dec3 = torch.cat([dec3_upsampled, enc2_crop], dim=1)  # Skip connection
        dec3 = self.decoder3_relu1(self.decoder3_conv1(dec3))
        dec3 = self.decoder3_relu2(self.decoder3_conv2(dec3))
        
        dec4_upsampled = self.decoder4_upsample(dec3)
        # Crop the corresponding feature map from encoder 1
        enc1_crop = self.copy_and_crop(enc1, dec4_upsampled.shape[2:])
        dec4 = torch.cat([dec4_upsampled, enc1_crop], dim=1)  # Skip connection
        dec4 = self.decoder4_relu1(self.decoder4_conv1(dec4))
        dec4 = self.decoder4_relu2(self.decoder4_conv2(dec4))
        
        # Output
        output = self.output(dec4)
        return output

# Define the dataset class
class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_files = os.listdir(input_dir)
        self.target_files = os.listdir(target_dir)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_img_name = os.path.join(self.input_dir, self.input_files[idx])
        target_img_name = os.path.join(self.target_dir, self.target_files[idx])
        input_image = Image.open(input_img_name).convert('L')  # Convert to grayscale
        target_image = Image.open(target_img_name).convert('LA')  # Convert to grayscale

        if self.transform:
            input_image = self.transform(input_image)
        target_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        target_image_tensor = target_transform(target_image).unsqueeze(0)
        return input_image, target_image_tensor

# Define training parameters
num_epochs = 25
batch_size = 1
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the dataset and data loader
input_dir = os.getcwd() + '/input_dir'
target_dir = os.getcwd() + '/output_dir'
transform = transforms.Compose([
    transforms.Resize((572, 572)),  # Resize the image to match UNet input size
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
])
dataset = ImageDataset(input_dir, target_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = UNet(in_channels=1, out_channels=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        targets = targets.squeeze(1)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    # Save the output image after each epoch
    with torch.no_grad():
        output = model(inputs)
        output_np = output.squeeze().cpu().numpy()  # Remove the channel dimension
        # Normalize the output to [0, 1]
        output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min())
        # Convert the two-channel output to a single-channel grayscale image
        output_gray = (output_np[0] + output_np[1]) / 2
        # Apply the "gray" colormap
        plt.imshow(output_gray, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{train_artifacts_dir}/output_epoch_{epoch+1}.jpeg', bbox_inches='tight', pad_inches=0)
        plt.close()

print('Training finished!')
