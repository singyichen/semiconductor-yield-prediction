"""
模型訓練模組
處理不平衡數據、超參數調優、模型訓練
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, f1_score,
                             roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class YieldPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.threshold = 0.5
        
    def handle_imbalance(self, X, y, method='smote'):
        """處理不平衡數據
        
        Args:
            method: 'smote', 'undersample', 'hybrid'
        """
        print(f"原始數據分佈:\n{pd.Series(y).value_counts(normalize=True)}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'hybrid':
            # 先過採樣,再欠採樣
            sampler = ImbPipeline([
                ('over', SMOTE(sampling_strategy=0.5, random_state=42)),
                ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=42))
            ])
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        print(f"重採樣後分佈:\n{pd.Series(y_resampled).value_counts(normalize=True)}")
        
        return X_resampled, y_resampled
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """訓練多個模型並比較"""
        
        # 模型配置
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                random_state=42,
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
        }
        
        results = []
        
        print("=" * 60)
        print("開始訓練模型...")
        print("=" * 60)
        
        for name, model in models_config.items():
            print(f"\n訓練 {name}...")
            
            # 訓練
            model.fit(X_train, y_train)
            
            # 預測
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 評估
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            
            # 找最佳閾值 (最大化 F1)
            precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_threshold = thresholds[np.argmax(f1_scores)]
            
            y_pred_best = (y_pred_proba >= best_threshold).astype(int)
            f1_best = f1_score(y_val, y_pred_best)
            
            results.append({
                'Model': name,
                'ROC-AUC': roc_auc,
                'F1 (0.5)': f1,
                'F1 (best)': f1_best,
                'Best Threshold': best_threshold
            })
            
            self.models[name] = model
            
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  F1 Score (0.5): {f1:.4f}")
            print(f"  F1 Score (best): {f1_best:.4f}")
            print(f"  Best Threshold: {best_threshold:.4f}")
        
        # 結果比較
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("模型比較:")
        print("=" * 60)
        print(results_df.to_string(index=False))
        
        # 選擇最佳模型 (根據 F1)
        best_idx = results_df['F1 (best)'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        self.threshold = results_df.loc[best_idx, 'Best Threshold']
        
        print(f"\n最佳模型: {self.best_model_name}")
        print(f"最佳閾值: {self.threshold:.4f}")
        
        return results_df
    
    def plot_roc_curves(self, X_val, y_val, save_path='outputs/roc_curves.png'):
        """繪製所有模型的 ROC 曲線"""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC 曲線已儲存至 {save_path}")
    
    def plot_confusion_matrix(self, X_val, y_val, save_path='outputs/confusion_matrix.png'):
        """繪製混淆矩陣"""
        y_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        cm = confusion_matrix(y_val, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Good', 'Defect'],
                   yticklabels=['Good', 'Defect'])
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩陣已儲存至 {save_path}")
    
    def cross_validate(self, X, y, cv=5):
        """交叉驗證"""
        print(f"\n執行 {cv}-Fold 交叉驗證...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            self.best_model, X, y, 
            cv=skf, 
            scoring='f1',
            n_jobs=-1
        )
        
        print(f"CV F1 Scores: {cv_scores}")
        print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_scores
    
    def feature_importance_analysis(self, feature_names, top_n=20, 
                                   save_path='outputs/feature_importance_final.png'):
        """特徵重要性分析"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        else:
            print("模型不支援特徵重要性")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {self.best_model_name}', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTop 10 重要特徵:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def save_model(self, path='../models/best_yield_model.pkl'):
        """儲存最佳模型"""
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'threshold': self.threshold
        }, path)
        print(f"\n模型已儲存至 {path}")
    
    def load_model(self, path='../models/best_yield_model.pkl'):
        """載入模型"""
        data = joblib.load(path)
        self.best_model = data['model']
        self.best_model_name = data['model_name']
        self.threshold = data['threshold']
        print(f"模型已載入: {self.best_model_name}")

# 完整訓練流程
if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # 模擬數據
    np.random.seed(42)
    n_samples = 2000
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 不平衡目標 (10% 不良品)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # 訓練
    predictor = YieldPredictor()
    
    # 處理不平衡
    X_train_balanced, y_train_balanced = predictor.handle_imbalance(
        X_train, y_train, method='hybrid'
    )
    
    # 訓練模型
    results = predictor.train_models(
        X_train_balanced, y_train_balanced,
        X_val, y_val
    )
    
    # 視覺化
    predictor.plot_roc_curves(X_val, y_val)
    predictor.plot_confusion_matrix(X_val, y_val)
    predictor.feature_importance_analysis(X.columns)
    
    # 交叉驗證
    predictor.cross_validate(X_train_balanced, y_train_balanced)
    
    # 最終測試集評估
    y_test_pred_proba = predictor.best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_pred_proba >= predictor.threshold).astype(int)
    
    print("\n" + "=" * 60)
    print("測試集最終結果:")
    print("=" * 60)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Good', 'Defect']))
    
    # 儲存模型
    predictor.save_model('../models/best_yield_model.pkl')