"""
數據預處理模組
處理缺失值、異常值和特徵縮放
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def load_data(self, data_path, label_path=None):
        """載入 SECOM 數據集"""
        # SECOM 數據集: 1567 samples, 590 features
        df = pd.read_csv(data_path, sep=' ', header=None)
        
        if label_path:
            labels = pd.read_csv(label_path, sep=' ', header=None)
            df['target'] = labels[0].map({-1: 0, 1: 1})  # 不良品為 1
        
        print(f"數據形狀: {df.shape}")
        print(f"缺失值比例: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")
        
        return df
    
    def handle_missing_values(self, df):
        """處理缺失值
        
        策略:
        1. 刪除缺失超過 50% 的特徵
        2. 其餘用中位數填充
        """
        # 計算缺失比例
        missing_pct = df.isnull().sum() / len(df)
        
        # 刪除缺失過多的欄位
        cols_to_drop = missing_pct[missing_pct > 0.5].index
        print(f"刪除 {len(cols_to_drop)} 個缺失率 > 50% 的特徵")
        
        df_clean = df.drop(columns=cols_to_drop)
        
        # 記錄特徵名稱(排除 target)
        self.feature_names = [col for col in df_clean.columns if col != 'target']
        
        # 分離特徵和標籤
        if 'target' in df_clean.columns:
            X = df_clean.drop(columns=['target'])
            y = df_clean['target']
        else:
            X = df_clean
            y = None
        
        # 中位數填充
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_imputed, y
    
    def remove_outliers(self, X, y=None, contamination=0.05):
        """使用 Isolation Forest 移除異常值"""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = iso_forest.fit_predict(X)
        
        # -1 表示異常值
        mask = outlier_pred != -1
        print(f"移除 {(~mask).sum()} 個異常樣本 ({(~mask).sum()/len(X):.2%})")
        
        X_clean = X[mask]
        y_clean = y[mask] if y is not None else None
        
        return X_clean, y_clean
    
    def feature_scaling(self, X):
        """標準化特徵"""
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled
    
    def get_low_variance_features(self, X, threshold=0.01):
        """識別低變異特徵"""
        variances = X.var()
        low_var_features = variances[variances < threshold].index.tolist()
        print(f"發現 {len(low_var_features)} 個低變異特徵")
        return low_var_features
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """儲存預處理器"""
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }, path)
        print(f"預處理器已儲存至 {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """載入預處理器"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        self.feature_names = data['feature_names']
        print(f"預處理器已載入自 {path}")

# 使用範例
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # 載入數據 (使用 Kaggle SECOM dataset)
    # df = preprocessor.load_data('data/secom.data', 'data/secom_labels.data')
    
    # 模擬數據測試
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(1000, 100),
        columns=[f'sensor_{i}' for i in range(100)]
    )
    df['target'] = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])
    
    # 加入缺失值
    mask = np.random.random(df.shape) < 0.3
    df = df.mask(mask)
    
    # 預處理
    X, y = preprocessor.handle_missing_values(df)
    X_clean, y_clean = preprocessor.remove_outliers(X, y)
    X_scaled = preprocessor.feature_scaling(X_clean)
    
    print(f"\n處理後數據形狀: {X_scaled.shape}")
    print(f"目標分佈:\n{y_clean.value_counts(normalize=True)}")