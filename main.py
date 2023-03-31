import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)  # first augmented image
        z2 = self.encoder(x2)  # second augmented image

        # apply projection head to obtain representation
        p1 = self.projection_head(z1)
        p2 = self.projection_head(z2)

        return z1, z2, p1, p2


# data augmentation transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# CIFAR10 dataset
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# dataloader
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# ViT model
model = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12)

# SimCLR model
simclr_model = SimCLR(encoder=model)

# optimizer
optimizer = Adam(simclr_model.parameters(), lr=0.1, weight_decay=1e-6)

# learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=len(data_loader), eta_min=0, last_epoch=-1)

# training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simclr_model.to(device)
if __name__ == '__main__':
    for epoch in range(num_epochs):
        simclr_model.train()
        total_loss = 0
        for (x1, x2) in tqdm(data_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            # forward pass
            z1, z2, p1, p2 = simclr_model(x1, x2)
            # compute loss
            loss = 2 - 2 * (F.cosine_similarity(p1, p2, dim=-1)).mean()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update scheduler
            scheduler.step()
            # accumulate loss
            total_loss += loss.item() * batch_size

        # print average loss for epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataset):.4f}")