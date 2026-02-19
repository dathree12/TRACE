import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- 1. U-Net 모델 정의 (Input: 256x256x1) ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))

# --- 2. Dice 계수 및 손실 함수 ---
def dice_coef(pred, target):
    smooth = 1e-6
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

# --- 3. 데이터셋 정의 (256x256 리사이즈 포함) ---
class LungDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # CXR 로드 (Grayscale)
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # 정규화
        
        # Mask 로드
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 127).astype(np.float32) # 이진화 및 정규화
        
        # Tensor 변환 [C, H, W]
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
        
        return img_tensor, mask_tensor, os.path.basename(self.img_paths[idx])

# --- 4. 메인 학습 루틴 ---
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 경로 설정
    img_files = sorted(glob.glob('/mnt/hdd/daseul/cxr_tuber/lung_seg_mask/CXR_png/*'))
    mask_files = sorted(glob.glob('/mnt/hdd/daseul/cxr_tuber/lung_seg_mask/ManualMask/MergedMask/*'))

    # 데이터 분할 (6:2:2)
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(img_files, mask_files, test_size=0.4, random_state=42)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs, temp_masks, test_size=0.5, random_state=42)

    train_ds = LungDataset(train_imgs, train_masks)
    val_ds = LungDataset(val_imgs, val_masks)
    test_ds = LungDataset(test_imgs, test_masks)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = UNet().to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Early Stopping 설정
    patience = 7
    best_val_dice = 0.0
    counter = 0

    print("Start Training...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_dice_total = 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_dice_total += dice_coef(preds, masks).item()
        
        avg_val_dice = val_dice_total / len(val_loader)
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {avg_val_dice:.4f}")

        # Early Stopping 체크
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), 'best_unet_lung.pth')
            counter = 0
            print("Best Model Saved!")
        else:
            counter += 1
            if counter >= patience:
                print("Early Stopping Triggered!")
                break

    # --- 5. 테스트 및 결과 저장 ---
    print("Testing and Saving Results...")
    model.load_state_dict(torch.load('best_unet_lung.pth'))
    model.eval()
    os.makedirs('./results', exist_ok=True)

    with torch.no_grad():
        for imgs, masks, filenames in test_loader:
            imgs_dev = imgs.to(device)
            preds = model(imgs_dev)
            dice_score = dice_coef(preds, masks.to(device)).item()
            
            # 이미지 저장용 변환
            orig = (imgs.squeeze().numpy() * 255).astype(np.uint8)
            gt = (masks.squeeze().numpy() * 255).astype(np.uint8)
            pred = (preds.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            
            # 가로로 합치기 (Input | GT | Prediction)
            result_img = np.hstack([orig, gt, pred])
            save_path = f"./results/dice_{dice_score:.4f}_{filenames[0]}"
            cv2.imwrite(save_path, result_img)

    print("All tasks completed.")

if __name__ == "__main__":
    train_model()