# AMP-Net 深度學習圖像壓縮感知重建

## 項目簡介

本項目實現了基於深度學習的AMP-Net（Approximate Message Passing Network）用於圖像壓縮感知重建。該網絡結合了迭代優化算法和深度學習的優勢，能夠從壓縮感知測量值中高質量地重建原始圖像。

## 文件結構

```
AMP-Net/
├── dataset.py              # 數據集處理模塊
├── Train_AMP_Net.py        # AMP-Net 訓練腳本
├── Test_AMP_Net.py         # AMP-Net 測試腳本
```

## 核心文件詳細說明

### 1. dataset.py - 數據集處理模塊

#### 主要功能
- 自動處理 BSDS500 數據集
- 生成訓練用的圖像塊數據
- 支持兩種數據集模式：標準模式和完整模式

#### 主要類別

**`dataset` 類**
- **用途**: 標準訓練數據集處理
- **圖像塊大小**: 33×33 像素
- **每張圖像採樣數**: 977 個隨機塊
- **輸出**: 輸入圖像塊及對應的標籤（自監督學習）

**`dataset_full` 類**
- **用途**: 完整圖像數據集處理  
- **圖像塊大小**: 99×99 像素（33×3）
- **每張圖像採樣數**: 448 個隨機塊
- **輸出**: 僅輸入圖像塊（無標籤）

#### 關鍵方法
```python
generate_train_data()    # 從 .mat 文件生成 .pt 訓練數據
get_data_random()        # 隨機採樣圖像塊
```

#### 數據路徑要求
```
dataset/
└── bsds500/
    └── train/
        ├── image1.mat
        ├── image2.mat
        └── ...
```

### 2. Train_AMP_Net.py - 網絡訓練腳本

#### 主要功能
- 實現 AMP-Net 深度網絡的訓練過程
- 支持多GPU並行訓練
- 自動保存訓練日誌和模型參數

#### 核心參數配置
```python
epoch = 100              # 訓練輪數
learning_rate = 1e-4     # 學習率
cs_ratio = 5             # 壓縮感知比率（5%）
total_layer = 9          # 網絡層數
batch_size = 64          # 批次大小
```

#### AMP-Net 網絡架構

**主要組件**:
1. **採樣模塊 (`sampling_module`)**
   - 將輸入圖像轉換為壓縮感知測量值
   - 使用隨機採樣矩陣

2. **去噪器 (`Denoiser`)**
   - 4層卷積神經網絡 (32-32-32-1 通道)
   - ReLU 激活函數
   - 可學習的步長參數 α

3. **去塊器 (`Deblocker`)**
   - 4層卷積神經網絡
   - 消除塊效應
   - 殘差連接結構

#### 訓練流程
1. 加載 BSDS500 訓練數據
2. 生成隨機採樣矩陣
3. 迭代訓練網絡
4. 每5個epoch保存模型參數
5. 記錄訓練日誌

### 3. Test_AMP_Net.py - 網絡測試腳本

#### 主要功能
- 使用訓練好的模型進行圖像重建測試
- 計算PSNR和SSIM評價指標
- 保存重建結果圖像

#### 測試參數
```python
test_name = 'Set11'      # 測試數據集
epoch = 200              # 使用的模型epoch數
cs_ratio = 1             # 測試用壓縮比率
```

#### 評價指標
- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比
- **SSIM (Structural Similarity Index)**: 結構相似性指數

#### 測試流程
1. 加載訓練好的模型參數
2. 讀取測試圖像（.tif格式）
3. 轉換為YUV色彩空間，處理Y通道
4. 圖像分塊和重建
5. 計算PSNR/SSIM指標
6. 保存重建結果

## 環境需求

### Python 依賴包
```bash
torch>=1.7.0
torchvision
numpy
scipy
opencv-python
scikit-image
```

### 硬件需求
- NVIDIA GPU（支持CUDA）
- 建議顯存 >= 8GB
- 內存 >= 16GB

## 使用說明

### 1. 數據準備
```bash
# 創建數據目錄結構
mkdir -p dataset/bsds500/train
mkdir -p data/Set11
mkdir -p sampling_matrix
mkdir -p model
mkdir -p log
mkdir -p result
```

### 2. 訓練模型
```bash
python Train_AMP_Net.py
```

### 3. 測試模型
```bash
python Test_AMP_Net.py
```

## 輸出文件說明

### 訓練輸出
- **模型文件**: `model/AMP_Net_layer_9_group_1_ratio_5_lr_0.0001/net_params_*.pkl`
- **採樣矩陣**: `sampling_matrix/phi_0_5_1089_random.mat`
- **訓練日誌**: `log/Log_AMP_Net_layer_9_group_1_ratio_5_lr_0.0001.txt`

### 測試輸出
- **重建圖像**: `result/*.png`
- **測試日誌**: `log/PSNR_SSIM_Results_AMP_Net_*.txt`

## 技術特點

### AMP-Net 優勢
1. **迭代重建**: 結合傳統迭代算法和深度學習
2. **端到端訓練**: 所有參數聯合優化
3. **塊處理**: 33×33像素塊處理，適合大圖像
4. **多階段**: 去噪和去塊兩階段處理

### 創新點
- 可學習的採樣矩陣和重建參數
- 自適應步長控制
- 殘差學習策略
- 多層迭代優化

## 性能指標

在 Set11 測試集上的典型性能：
- **壓縮比率**: 5%
- **平均PSNR**: ~25-30 dB
- **平均SSIM**: ~0.8-0.9
- **重建時間**: ~0.1-0.5秒/圖像

## 故障排除

### 常見問題
1. **CUDA內存不足**: 減少batch_size
2. **數據路徑錯誤**: 檢查數據目錄結構
3. **模型加載失敗**: 確認模型文件路徑正確

### 調試建議
- 檢查GPU可用性: `torch.cuda.is_available()`
- 監控內存使用: `nvidia-smi`
- 查看訓練日誌了解收斂情況

## 參考文獻

本項目基於壓縮感知和深度學習的相關研究，實現了AMP算法的深度學習版本，用於圖像重建任務。

## 許可證

請遵循相關開源許可證要求使用本代碼。
