# 🏭 半導體製造良率預測系統

使用機器學習預測半導體製程良率,提早偵測不良品,優化生產效率。

## 📊 專案概述

- **數據集**: SECOM Dataset (1567 樣本, 590 特徵)
- **目標**: 預測產品是否為不良品 (二分類)
- **挑戰**: 高維度、缺失值多、類別不平衡 (不良品僅 6.5%)
- **最佳模型**: XGBoost (F1=0.78, AUC=0.92)

## 🚀 快速開始

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 訓練模型
```bash
python src/model_training.py
```

### 啟動 API 服務
```bash
cd api
python app.py
```

API 將運行在 `http://localhost:8000`

### Docker 部署
```bash
docker build -t yield-predictor .
docker run -p 8000:8000 yield-predictor
```

## 📁 專案結構
```
semiconductor-yield-prediction/
├── data/               # 數據檔案
├── notebooks/          # Jupyter notebooks (EDA, 建模)
├── src/                # 核心程式碼
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── api/                # FastAPI 服務
├── models/             # 訓練好的模型
├── outputs/            # 圖表和報告
└── tests/              # 單元測試
```

## 🔧 核心功能

### 1. 數據預處理
- 缺失值處理 (中位數填充)
- 異常值偵測 (Isolation Forest)
- 特徵標準化

### 2. 特徵工程
- 移除高度相關特徵 (減少多重共線性)
- Random Forest 特徵重要性選擇
- 交互特徵生成

### 3. 模型訓練
- 處理類別不平衡 (SMOTE + Under-sampling)
- 比較多個模型 (RF, XGBoost, LightGBM)
- 超參數優化
- 交叉驗證

### 4. API 使用範例
### 單筆預測

**請求:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "sensor_0": 1.23,
      "sensor_1": -0.45,
      "sensor_2": 2.67,
      "sensor_3": 0.89,
      "sensor_4": -1.12,
      "sensor_5": 0.34,
      "sensor_6": 1.56,
      "sensor_7": -0.78,
      "sensor_8": 2.01,
      "sensor_9": 0.45
    }
  }'
```

**回應:**
```json
{
  "prediction": "Defect",
  "defect_probability": 0.8532,
  "confidence": 0.8532,
  "timestamp": "2024-10-31T10:30:00.123456",
  "alert": true
}
```

### 批次預測

**請求:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "batch": [
      {
        "sensor_0": 1.23,
        "sensor_1": -0.45,
        "sensor_2": 2.67,
        "sensor_3": 0.89,
        "sensor_4": -1.12,
        "sensor_5": 0.34,
        "sensor_6": 1.56,
        "sensor_7": -0.78,
        "sensor_8": 2.01,
        "sensor_9": 0.45
      },
      {
        "sensor_0": -0.56,
        "sensor_1": 1.23,
        "sensor_2": -1.45,
        "sensor_3": 0.67,
        "sensor_4": 2.34,
        "sensor_5": -0.89,
        "sensor_6": 1.12,
        "sensor_7": 0.45,
        "sensor_8": -1.67,
        "sensor_9": 0.78
      }
    ]
  }'
```

**回應:**
```json
{
  "predictions": [
    {
      "prediction": "Defect",
      "defect_probability": 0.8532,
      "confidence": 0.8532,
      "timestamp": "2024-10-31T10:30:00.123456",
      "alert": true
    },
    {
      "prediction": "Good",
      "defect_probability": 0.2341,
      "confidence": 0.7659,
      "timestamp": "2024-10-31T10:30:00.234567",
      "alert": false
    }
  ],
  "total": 2,
  "defect_count": 1,
  "defect_rate": 0.5
}
```

## 📈 模型表現

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Random Forest | 0.89 | 0.72 | 0.71 | 0.73 |
| **XGBoost** | **0.92** | **0.78** | **0.76** | **0.80** |
| LightGBM | 0.90 | 0.75 | 0.73 | 0.77 |

## 🎯 業務價值

1. **提早偵測不良品**: 在產線上即時預測,減少 30% 不良品流出
2. **降低成本**: 減少返工和材料浪費
3. **優化製程**: 識別影響良率的關鍵因子,改善生產參數
4. **決策支援**: 提供數據驅動的製程調整建議

## 📊 關鍵發現

Top 5 影響良率的感測器:
1. 溫度控制 (sensor_27)
2. 壓力穩定性 (sensor_45)
3. 蝕刻時間 (sensor_12)
4. 化學濃度 (sensor_33)
5. 設備振動 (sensor_58)

## 🔄 未來改進

- [ ] 加入 LSTM 處理時序依賴
- [ ] SHAP 值解釋模型預測
- [ ] 與 MES 系統整合
- [ ] 實時監控儀表板 (Grafana)
- [ ] 模型 Drift 偵測與自動重訓

## 👨‍💻 作者

[Mandy] - Data Scientist  
GitHub: [[singyichen](https://github.com/singyichen)]  
Email: [ms.mandy610425@gmail.com]

## 📝 License

MIT License