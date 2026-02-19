import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from tqdm import tqdm

# ==========================================
# ⚙️ 설정
# ==========================================
DATA_ROOT = "/mnt/hdd/medsam/dataset" 
OUTPUT_DIR = "tb_results_efficientnet_b3"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_efficientnet_b3.pth")
EXCEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "efficientnet_b3_test_result.xlsx")
CAM_SAVE_DIR = os.path.join(OUTPUT_DIR, "grad_cam_images")

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
PATIENCE = 5
IMG_SIZE = 512  # EfficientNet-B3의 권장 해상도는 300x300 입니다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CAM_SAVE_DIR, exist_ok=True)

# ==========================================
# 📂 데이터셋 & 밸런스 분할
# ==========================================
class CXRDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.filenames = []
        
        self.class_map = {'normal': 0, 'tuberculosis': 1}
        target_dir = os.path.join(root_dir, mode) if os.path.exists(os.path.join(root_dir, mode)) else root_dir
        
        for class_name, label in self.class_map.items():
            class_path = os.path.join(target_dir, class_name)
            if not os.path.exists(class_path): continue
            
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                self.image_paths.append(os.path.join(class_path, f))
                self.labels.append(label)
                self.filenames.append(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]
        try:
            image = Image.open(path).convert("RGB")
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), filename, path

def create_balanced_split(dataset, val_count_per_class=250):
    indices = list(range(len(dataset)))
    labels = dataset.labels
    
    idx_normal = [i for i, label in enumerate(labels) if label == 0]
    idx_tuber = [i for i, label in enumerate(labels) if label == 1]
    
    random.shuffle(idx_normal)
    random.shuffle(idx_tuber)
    
    if len(idx_normal) < val_count_per_class or len(idx_tuber) < val_count_per_class:
        print("⚠️ 데이터 부족으로 비율 분할(8:2)로 대체합니다.")
        return torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    val_indices = idx_normal[:val_count_per_class] + idx_tuber[:val_count_per_class]
    train_indices = idx_normal[val_count_per_class:] + idx_tuber[val_count_per_class:]
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# ==========================================
# 🛑 Early Stopping 클래스
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   ⚠️ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

# ==========================================
# 🧠 Grad-CAM
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        output = self.model(x)
        target_category = output.argmax(dim=1)
        score = output[:, target_category]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, output

# ==========================================
# 🚀 학습 함수
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5):
    print(f"🔥 학습 시작 (Device: {DEVICE}, Model: EfficientNet-B3)")
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", leave=False)
        for images, labels, _, _ in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item())
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        v_correct, v_total = 0, 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val  ", leave=False)
        with torch.no_grad():
            for images, labels, _, _ in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                v_total += labels.size(0)
                v_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * v_correct / v_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    --> 🎉 Best Model Saved (Acc: {best_acc:.2f}%)")
            
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("🛑 Early Stopping triggered!")
            break

def evaluate(model, :qtest_loader):
    model.eval()
    
    # [EfficientNet-B3 Target Layer] features의 마지막 레이어
    grad_cam = GradCAM(model, model.features[-1])
    
    results = []
    labels_list, preds_list, probs_list = [], [], []
    
    print("\n🔎 추론 및 시각화 진행 중...")
    test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference")
    
    for i, (images, labels, fname, fpath) in test_bar:
        images = images.to(DEVICE)
        
        heatmap, logits = grad_cam(images)
        probs = torch.softmax(logits, dim=1)
        prob_tb = probs[0][1].item()
        pred = logits.argmax(dim=1).item()
        
        labels_list.append(labels.item())
        probs_list.append(prob_tb)
        preds_list.append(pred)
        
        # Grad-CAM 저장
        res_img = np.vstack([np.zeros((30, IMG_SIZE, 3), np.uint8), 
                             cv2.addWeighted(cv2.resize(cv2.imread(fpath[0]), (IMG_SIZE, IMG_SIZE)), 0.6, 
                                             cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET), 0.4, 0)])
        cv2.putText(res_img, f"GT:{labels.item()} P:{pred} ({prob_tb:.1%})", (5, 20), 4, 0.5, (255,255,255), 1)
        cv2.imwrite(os.path.join(CAM_SAVE_DIR, f"{fname[0]}_cam.png"), res_img)
        
        results.append({"File": fname[0], "GT": labels.item(), "Pred": pred, "Prob": prob_tb})

    acc = accuracy_score(labels_list, preds_list)
    auc = roc_auc_score(labels_list, probs_list) if len(set(labels_list)) > 1 else 0
    tn, fp, fn, tp = confusion_matrix(labels_list, preds_list).ravel()
    sen, spe = tp/(tp+fn), tn/(tn+fp)
    
    print(f"\n📊 Final Result - ACC: {acc:.3f}, AUC: {auc:.3f}, SEN: {sen:.3f}, SPE: {spe:.3f}")
    pd.DataFrame(results).to_excel(EXCEL_SAVE_PATH, index=False)

def main():
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    full_ds = CXRDataset(DATA_ROOT, transform, 'train')
    train_ds, val_ds = create_balanced_split(full_ds, val_count_per_class=250)
    test_ds = CXRDataset(DATA_ROOT, transform, 'test')
    
    print(f"🔍 데이터셋: Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # [EfficientNet-B3 정의]
    try:
        # 최신 버전 torchvision
        model = models.efficientnet_b3(weights='DEFAULT')
    except:
        # 구버전 torchvision 호환용
        model = models.efficientnet_b3(pretrained=True)

    # EfficientNet Classifier 수정 (classifier[1]이 마지막 Linear Layer)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)
    
    train_model(model, train_dl, val_dl, nn.CrossEntropyLoss(), 
                optim.Adam(model.parameters(), lr=LEARNING_RATE), 
                NUM_EPOCHS, patience=PATIENCE)
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("✅ Best Model 로드 완료")
    
    evaluate(model, test_dl)

if __name__ == "__main__":
    main()