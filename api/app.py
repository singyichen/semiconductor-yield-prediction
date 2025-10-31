"""
FastAPI 即時預測服務
提供 RESTful API 進行良率預測
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import uvicorn

# 初始化 FastAPI
app = FastAPI(
    title="半導體良率預測 API",
    description="即時預測產品良率,檢測不良品",
    version="1.0.0"
)

# 載入模型
try:
    model_data = joblib.load('../models/best_yield_model.pkl')
    model = model_data['model']
    threshold = model_data['threshold']
    
    # preprocessor_data = joblib.load('../models/preprocessor.pkl')
    # scaler = preprocessor_data['scaler']
    # feature_names = preprocessor_data['feature_names']
    
    print("模型載入成功!")
except Exception as e:
    print(f"模型載入失敗: {e}")
    model = None

# 請求模型
class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., 
        description="感測器數據字典,key 為特徵名稱,value 為數值"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "sensor_0": 1.23,
                    "sensor_1": -0.45,
                    "sensor_2": 2.67,
                    # ... 更多特徵
                }
            }
        }

class BatchPredictionRequest(BaseModel):
    batch: List[Dict[str, float]] = Field(
        ...,
        description="批次預測,包含多筆感測器數據"
    )

# 回應模型
class PredictionResponse(BaseModel):
    prediction: str  # "Good" or "Defect"
    defect_probability: float
    confidence: float
    timestamp: str
    alert: bool

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    defect_count: int
    defect_rate: float

# API 路由
@app.get("/")
async def root():
    """API 根路徑"""
    return {
        "message": "半導體良率預測 API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """單筆預測
    
    輸入感測器數據,返回預測結果
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未載入")
    
    try:
        # 準備特徵
        features_df = pd.DataFrame([request.features])
        
        # 確保特徵順序正確
        if not all(feat in features_df.columns for feat in feature_names):
            missing = [f for f in feature_names if f not in features_df.columns]
            raise HTTPException(
                status_code=400,
                detail=f"缺少特徵: {missing}"
            )
        
        features_df = features_df[feature_names]
        
        # 標準化
        features_scaled = scaler.transform(features_df)
        
        # 預測
        pred_proba = model.predict_proba(features_scaled)[0, 1]
        prediction = "Defect" if pred_proba >= threshold else "Good"
        confidence = pred_proba if prediction == "Defect" else (1 - pred_proba)
        
        # 告警邏輯 (不良品機率 > 80%)
        alert = pred_proba >= 0.8
        
        return PredictionResponse(
            prediction=prediction,
            defect_probability=round(pred_proba, 4),
            confidence=round(confidence, 4),
            timestamp=datetime.now().isoformat(),
            alert=alert
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測錯誤: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """批次預測
    
    一次預測多筆數據
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未載入")
    
    try:
        predictions = []
        defect_count = 0
        
        for features_dict in request.batch:
            # 準備特徵
            features_df = pd.DataFrame([features_dict])[feature_names]
            features_scaled = scaler.transform(features_df)
            
            # 預測
            pred_proba = model.predict_proba(features_scaled)[0, 1]
            prediction = "Defect" if pred_proba >= threshold else "Good"
            confidence = pred_proba if prediction == "Defect" else (1 - pred_proba)
            alert = pred_proba >= 0.8
            
            if prediction == "Defect":
                defect_count += 1
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                defect_probability=round(pred_proba, 4),
                confidence=round(confidence, 4),
                timestamp=datetime.now().isoformat(),
                alert=alert
            ))
        
        total = len(predictions)
        defect_rate = defect_count / total if total > 0 else 0
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=total,
            defect_count=defect_count,
            defect_rate=round(defect_rate, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批次預測錯誤: {str(e)}")

@app.get("/model/info")
async def model_info():
    """模型資訊"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未載入")
    
    return {
        "model_type": type(model).__name__,
        "threshold": threshold,
        "n_features": len(feature_names),
        "feature_names": feature_names[:10]  # 只顯示前 10 個
    }

# 運行服務
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )