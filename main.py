import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# Configuration and Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 40
L1_LAMBDA = 100
IMG_SIZE = 256
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"

print(f"Running on Device: {DEVICE}")

# Dataset Class
class BrainTumorDataset(Dataset):
    def __init__(self, t1_dir, t1c_dir):
        self.t1_dir = t1_dir
        self.t1c_dir = t1c_dir
        self.t1_files = sorted(glob.glob(os.path.join(t1_dir, "*.png")))
        self.t1c_files = sorted(glob.glob(os.path.join(t1c_dir, "*.png")))

        self.length = min(len(self.t1_files), len(self.t1c_files))
        if len(self.t1_files) != len(self.t1c_files):
            print(f"Warning: Mismatch detected. T1: {len(self.t1_files)}, T1c: {len(self.t1c_files)}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        t1_path = self.t1_files[index]
        t1c_path = self.t1c_files[index]
        
        img_t1 = Image.open(t1_path).convert("L")
        img_t1c = Image.open(t1c_path).convert("L")

        # Resize guarantees consistent shapes for the DataLoader
        img_t1 = img_t1.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        img_t1c = img_t1c.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        
        input_img = np.array(img_t1)
        target_img = np.array(img_t1c)
        
        # Normalize to [-1, 1]
        input_img = (input_img / 127.5) - 1.0
        target_img = (target_img / 127.5) - 1.0
        
        input_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target_img, dtype=torch.float32).unsqueeze(0)
        
        return input_tensor, target_tensor

# Generator Model (U-Net)
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        ) 
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features*4, features*8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"), nn.ReLU()
        )
        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features*2*2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        bottleneck = self.bottleneck(d6)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        return self.final_up(torch.cat([u6, d1], 1))

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        # Initial Layer: Takes 2 channels (x+y) -> converts to 64
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        layers = []
        in_c = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Conv2d(in_c, feature, 4, 1 if feature==features[-1] else 2, 1, bias=False, padding_mode="reflect")
            )
            layers.append(nn.BatchNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = feature
        
        layers.append(nn.Conv2d(in_c, 1, 4, 1, 1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1) 
        x = self.initial(x)          
        return self.model(x)

# Utils
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def calculate_metrics(real, fake):
    # Denormalize to [0, 1] range for metric calculation
    real = (real.detach().cpu().numpy() + 1) / 2.0
    fake = (fake.detach().cpu().numpy() + 1) / 2.0
    ssim_val = 0
    psnr_val = 0
    batch_size = real.shape[0]
    for i in range(batch_size):
        ssim_val += ssim_metric(real[i,0], fake[i,0], data_range=1.0)
        psnr_val += psnr_metric(real[i,0], fake[i,0], data_range=1.0)
    return ssim_val/batch_size, psnr_val/batch_size

# Training Function
def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    avg_ssim = 0
    avg_psnr = 0
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        # Calculate Metrics
        s, p = calculate_metrics(y, y_fake)
        avg_ssim += s
        avg_psnr += p

        if idx % 10 == 0:
            print(f"Batch {idx} | G Loss: {G_loss.item():.4f} | SSIM: {s:.4f} | PSNR: {p:.4f}")

    print(f"Epoch Avg -> SSIM: {avg_ssim/len(loader):.4f} | PSNR: {avg_psnr/len(loader):.4f}")

# Main execution
def main():
    # Update your paths here
    T1_PATH = "/content/T1 Cropped 3D-T/T1_imgs_middle_only" 
    T1C_PATH = "/content/T1c Cropped 3D-T/T1c_imgs_middle_only"
    
    if not os.path.exists(T1_PATH) or not os.path.exists(T1C_PATH):
        print("Error: Paths not found.")
        return

    dataset = BrainTumorDataset(T1_PATH, T1C_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    gen = Generator(in_channels=1).to(DEVICE)
    disc = Discriminator(in_channels=1).to(DEVICE)
    
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    g_scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE=="cuda"))
    d_scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE=="cuda"))

    if os.path.isfile(CHECKPOINT_GEN):
        load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
        load_checkpoint(CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        train_fn(disc, gen, loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

if __name__ == "__main__":
    main()