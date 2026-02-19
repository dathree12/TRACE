import sys
import os
import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QGroupBox, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# ==========================================
# ⚙️ 설정 및 모델 경로
# ==========================================
SEG_MODEL_PATH = "lung_segmentation_model_path" 
CLS_MODEL_PATH = "tuberculosis_model_path" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🏗️ 모델 구조 정의 
# ==========================================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(1, 64); self.enc2 = CBR(64, 128); self.enc3 = CBR(128, 256); self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec1 = CBR(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.gradients = None; self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def __call__(self, x):
        output = self.model(x); target_category = output.argmax(dim=1)
        score = output[:, target_category]; self.model.zero_grad(); score.backward(retain_graph=True)
        gradients = self.gradients.cpu().data.numpy()[0]; activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights): cam += w * activations[i]
        cam = np.maximum(cam, 0); cam = cv2.resize(cam, (256, 256))
        cam = cam - np.min(cam); cam = cam / (np.max(cam) + 1e-8)
        return cam, output

class TB_FullAnalyzer:
    def __init__(self):
        self.seg_model = UNet().to(DEVICE)
        if os.path.exists(SEG_MODEL_PATH): self.seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
        self.seg_model.eval()

        self.cls_model = models.resnet50(weights=None); num_ftrs = self.cls_model.fc.in_features
        self.cls_model.fc = nn.Linear(num_ftrs, 2)
        if os.path.exists(CLS_MODEL_PATH):
            state_dict = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
            if list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items(): new_state_dict[k[7:]] = v
                state_dict = new_state_dict
            self.cls_model.load_state_dict(state_dict)
        self.cls_model.to(DEVICE).eval()
        self.grad_cam = GradCAM(self.cls_model, self.cls_model.layer4[-1])
        
        self.transform_seg = transforms.Compose([transforms.Grayscale(), transforms.Resize((256, 256)), transforms.ToTensor()])
        self.transform_cls = transforms.Compose([transforms.Resize((512, 512
                                                                    )), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(self, img_pil):
        # 1. Lung Segmentation 예측
        img_seg_in = self.transform_seg(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask = self.seg_model(img_seg_in)
            # Thresholding (0 or 1)
            mask_np = (mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # --- Morphological Closing 보정 ---
        # 작은 구멍을 메우고 경계선을 부드럽게 연결합니다.
        kernel = np.ones((10, 10), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        # ---------------------------------------------

        # 2. 원본 이미지 전처리 (Segmentation Mask 적용을 위해)
        orig_np = np.array(img_pil.convert("L"))
        orig_resized = cv2.resize(orig_np, (256, 256))
        
        # 3. 배경 제거 (Lung Only)
        masked_lung = cv2.bitwise_and(orig_resized, orig_resized, mask=mask_np)
        # 4. TB Classification & Grad-CAM
        masked_pil = Image.fromarray(cv2.cvtColor(masked_lung, cv2.COLOR_GRAY2RGB))
        img_cls_in = self.transform_cls(masked_pil).unsqueeze(0).to(DEVICE)
        heatmap, logits = self.grad_cam(img_cls_in)
        
        prob_tb = torch.softmax(logits, dim=1)[0][1].item()
        result_text = "Tuberculosis" if logits.argmax(dim=1).item() == 1 else "Normal"
        
        return result_text, prob_tb, heatmap, masked_lung, mask_np

# ==========================================
# 🖥️ PyQt5 GUI 메인 (기존과 동일)
# ==========================================
class TBDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.analyzer = TB_FullAnalyzer()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('AI TB Detector (DICOM Info + Grad-CAM)')
        self.resize(1100, 800)
        
        main_layout = QVBoxLayout()

        # 1. 상단: 파일 로드 버튼 & 기본 정보
        top_layout = QHBoxLayout()
        self.btn_load = QPushButton('📂 이미지 불러오기 (DICOM/PNG)')
        self.btn_load.setFixedSize(250, 50)
        self.btn_load.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_load.clicked.connect(self.load_image)
        
        self.lbl_status = QLabel("이미지를 선택해주세요."); 
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 14px; color: #555;")
        
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.lbl_status)
        main_layout.addLayout(top_layout)

        # 2. 환자 정보 패널 (DICOM Header Info)
        self.info_group = QGroupBox("📋 환자 및 검사 정보 (DICOM Header)")
        self.info_layout = QGridLayout()
        
        def make_kv_label(key):
            lbl_k = QLabel(key); lbl_k.setStyleSheet("font-weight: bold; color: #333;")
            lbl_v = QLabel("-"); lbl_v.setStyleSheet("color: #000; background-color: #f0f0f0; padding: 2px;")
            return lbl_k, lbl_v

        self.lbl_pid_k, self.lbl_pid_v = make_kv_label("Patient ID:")
        self.lbl_dob_k, self.lbl_dob_v = make_kv_label("Birth Date:")
        self.lbl_sex_k, self.lbl_sex_v = make_kv_label("Sex:")
        self.lbl_age_k, self.lbl_age_v = make_kv_label("Age:")
        self.lbl_sdate_k, self.lbl_sdate_v = make_kv_label("Study Date:")
        self.lbl_series_date_k, self.lbl_series_date_v = make_kv_label("Series Date:")
        self.lbl_desc_k, self.lbl_desc_v = make_kv_label("Description:")

        self.info_layout.addWidget(self.lbl_pid_k, 0, 0); self.info_layout.addWidget(self.lbl_pid_v, 0, 1)
        self.info_layout.addWidget(self.lbl_dob_k, 0, 2); self.info_layout.addWidget(self.lbl_dob_v, 0, 3)
        self.info_layout.addWidget(self.lbl_sex_k, 1, 0); self.info_layout.addWidget(self.lbl_sex_v, 1, 1)
        self.info_layout.addWidget(self.lbl_age_k, 1, 2); self.info_layout.addWidget(self.lbl_age_v, 1, 3)
        self.info_layout.addWidget(self.lbl_sdate_k, 2, 0); self.info_layout.addWidget(self.lbl_sdate_v, 2, 1)
        self.info_layout.addWidget(self.lbl_series_date_k, 2, 2); self.info_layout.addWidget(self.lbl_series_date_v, 2, 3)
        self.info_layout.addWidget(self.lbl_desc_k, 3, 0); self.info_layout.addWidget(self.lbl_desc_v, 3, 1, 1, 3)

        self.info_group.setLayout(self.info_layout)
        main_layout.addWidget(self.info_group)

        # 3. AI 결과 표시줄
        self.text_result = QLabel("AI 판독 대기 중..."); 
        self.text_result.setAlignment(Qt.AlignCenter)
        self.text_result.setStyleSheet("border: 2px solid #aaa; font-size: 18px; padding: 10px; background-color: #fff;")
        main_layout.addWidget(self.text_result)

        # 4. 이미지 뷰어 (원본 vs 결과)
        img_layout = QHBoxLayout()
        def create_img_label(title):
            container = QVBoxLayout(); lbl_title = QLabel(title); lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; font-size: 14px;")
            lbl_img = QLabel(); lbl_img.setFixedSize(450, 450); lbl_img.setStyleSheet("background-color: black; border: 1px solid #777;")
            container.addWidget(lbl_title); container.addWidget(lbl_img)
            return container, lbl_img

        self.box_orig, self.lbl_orig = create_img_label("원본 영상 (Original)")
        self.box_cam, self.lbl_cam = create_img_label("AI 분석 결과 (Heatmap)")

        img_layout.addLayout(self.box_orig); img_layout.addLayout(self.box_cam)
        main_layout.addLayout(img_layout)
        self.setLayout(main_layout)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '이미지 선택', './', "Medical Images (*.dcm *.png *.jpg *.jpeg)")
        if fname: self.process_image(fname)

    def extract_dicom_meta(self, fpath):
        meta = {
            "PatientID": "N/A", "PatientBirthDate": "N/A", "PatientSex": "N/A",
            "PatientAge": "N/A", "StudyDate": "N/A", "SeriesDate": "N/A", "SeriesDescription": "N/A"
        }
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=False)
            meta["PatientID"] = str(ds.get("PatientID", "N/A"))
            meta["PatientBirthDate"] = str(ds.get("PatientBirthDate", "N/A"))
            meta["PatientSex"] = str(ds.get("PatientSex", "N/A"))
            meta["PatientAge"] = str(ds.get("PatientAge", "N/A"))
            meta["StudyDate"] = str(ds.get("StudyDate", "N/A"))
            meta["SeriesDate"] = str(ds.get("SeriesDate", "N/A"))
            meta["SeriesDescription"] = str(ds.get("SeriesDescription", "N/A"))
            return ds, meta
        except Exception as e:
            print(f"DICOM Read Error: {e}")
            return None, meta

    def process_image(self, fpath):
        try:
            img_pil = None
            is_dicom = fpath.lower().endswith('.dcm')
            
            if is_dicom:
                ds, meta = self.extract_dicom_meta(fpath)
                self.lbl_pid_v.setText(meta["PatientID"])
                self.lbl_dob_v.setText(meta["PatientBirthDate"])
                self.lbl_sex_v.setText(meta["PatientSex"])
                self.lbl_age_v.setText(meta["PatientAge"])
                self.lbl_sdate_v.setText(meta["StudyDate"])
                self.lbl_series_date_v.setText(meta["SeriesDate"])
                self.lbl_desc_v.setText(meta["SeriesDescription"])
                
                if hasattr(ds, 'pixel_array'):
                    px = ds.pixel_array.astype(float)
                    px = (px - px.min()) / (px.max() - px.min()) * 255.0
                    img_array = np.uint8(px)
                    img_pil = Image.fromarray(img_array).convert("RGB")
            else:
                for lbl in [self.lbl_pid_v, self.lbl_dob_v, self.lbl_sex_v, self.lbl_age_v, 
                            self.lbl_sdate_v, self.lbl_series_date_v, self.lbl_desc_v]:
                    lbl.setText("파일 정보 없음 (Non-DICOM)")
                img_pil = Image.open(fpath).convert("RGB")

            self.lbl_status.setText(f"로드 완료: {os.path.basename(fpath)}")

            result_text, prob_tb, heatmap, masked_lung, mask_np = self.analyzer.predict(img_pil)

            color = "red" if prob_tb > 0.5 else "blue"
            self.text_result.setStyleSheet(f"border: 2px solid {color}; font-size: 18px; padding: 10px; background-color: #fff; color: {color}; font-weight: bold;")
            self.text_result.setText(f"AI 판독: {result_text} (TB 확률: {prob_tb*100:.1f}%)")

            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_disp = cv2.resize(img_bgr, (450, 450))
            self.lbl_orig.setPixmap(self.convert_cv_qt(img_disp))

            masked_lung_disp = cv2.resize(masked_lung, (450, 450))
            masked_lung_bgr = cv2.cvtColor(masked_lung_disp, cv2.COLOR_GRAY2BGR)
            
            weighted_heatmap = heatmap * prob_tb 
            heatmap_uint8 = np.uint8(255 * weighted_heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_disp = cv2.resize(heatmap_color, (450, 450))
            
            cam_overlay = cv2.addWeighted(masked_lung_bgr, 0.6, heatmap_disp, 0.4, 0)
            mask_c = cv2.resize(mask_np, (450, 450))
            cam_overlay[mask_c == 0] = 0 
            
            self.lbl_cam.setPixmap(self.convert_cv_qt(cam_overlay))

        except Exception as e: 
            self.lbl_status.setText(f"오류 발생: {str(e)}")
            print(e)
            
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(450, 450, Qt.KeepAspectRatio)

if __name__ == '__main__':
    app = QApplication(sys.argv); ex = TBDetectorApp(); ex.show(); sys.exit(app.exec_())
