"""
特徵工程模組
包含特徵選擇、交互特徵、時序特徵等
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    def __init__(self):
        self.selected_features = None
        self.pca = None
        
    def correlation_analysis(self, X, y, threshold=0.8):
        """相關性分析
        
        1. 移除高度相關的特徵 (multicollinearity)
        2. 識別與目標相關的特徵
        """
        # 特徵間相關性
        corr_matrix = X.corr().abs()
        
        # 找出高度相關的特徵對
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper_tri.columns 
                   if any(upper_tri[col] > threshold)]
        
        print(f"移除 {len(to_drop)} 個高度相關特徵")
        
        # 與目標的相關性
        df_temp = X.copy()
        df_temp['target'] = y
        target_corr = df_temp.corr()['target'].drop('target').abs()
        
        # 視覺化 Top 20 特徵
        plt.figure(figsize=(12, 6))
        target_corr.nlargest(20).plot(kind='barh')
        plt.title('Top 20 Features Correlated with Target')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.savefig('outputs/feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return to_drop
    
    def statistical_feature_selection(self, X, y, k=50):
        """統計方法特徵選擇
        
        使用 F-test 和互資訊選擇最重要的 k 個特徵
        """
        # F-test
        selector_f = SelectKBest(f_classif, k=k)
        selector_f.fit(X, y)
        
        # Mutual Information
        selector_mi = SelectKBest(mutual_info_classif, k=k)
        selector_mi.fit(X, y)
        
        # 取交集
        f_features = X.columns[selector_f.get_support()].tolist()
        mi_features = X.columns[selector_mi.get_support()].tolist()
        
        selected = list(set(f_features) & set(mi_features))
        print(f"F-test 和 MI 共同選擇 {len(selected)} 個特徵")
        
        return selected
    
    def model_based_feature_selection(self, X, y, top_k=50):
        """基於模型的特徵選擇
        
        使用 Random Forest 特徵重要性
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # 特徵重要性
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 視覺化 Top 30
        plt.figure(figsize=(12, 8))
        plt.barh(
            importance_df.head(30)['feature'],
            importance_df.head(30)['importance']
        )
        plt.xlabel('Importance')
        plt.title('Top 30 Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        selected_features = importance_df.head(top_k)['feature'].tolist()
        
        return selected_features, importance_df
    
    def create_interaction_features(self, X, top_features, n_interactions=10):
        """創建交互特徵
        
        針對最重要的特徵創建交互項
        """
        X_new = X.copy()
        
        for i in range(min(n_interactions, len(top_features))):
            for j in range(i+1, min(n_interactions, len(top_features))):
                feat_i = top_features[i]
                feat_j = top_features[j]
                
                # 乘法交互
                X_new[f'{feat_i}_x_{feat_j}'] = X[feat_i] * X[feat_j]
                
                # 除法交互 (避免除以零)
                X_new[f'{feat_i}_div_{feat_j}'] = X[feat_i] / (X[feat_j] + 1e-5)
        
        print(f"創建 {X_new.shape[1] - X.shape[1]} 個交互特徵")
        
        return X_new
    
    def apply_pca(self, X, n_components=0.95):
        """主成分分析降維
        
        保留 95% 變異
        """
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} 維")
        print(f"解釋變異: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        # 視覺化
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/pca_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return pd.DataFrame(
            X_pca,
            columns=[f'PC_{i+1}' for i in range(X_pca.shape[1])],
            index=X.index
        )
    
    def engineer_features(self, X, y, method='rf', top_k=50):
        """完整特徵工程流程"""
        
        print("=" * 50)
        print("開始特徵工程...")
        print("=" * 50)
        
        # 1. 移除高度相關特徵
        to_drop = self.correlation_analysis(X, y)
        X_decorr = X.drop(columns=to_drop)
        
        # 2. 特徵選擇
        if method == 'statistical':
            selected = self.statistical_feature_selection(X_decorr, y, k=top_k)
        elif method == 'rf':
            selected, _ = self.model_based_feature_selection(X_decorr, y, top_k=top_k)
        else:
            selected = X_decorr.columns.tolist()
        
        self.selected_features = selected
        X_selected = X_decorr[selected]
        
        # 3. 創建交互特徵
        X_with_interactions = self.create_interaction_features(
            X_selected, 
            selected[:10]  # 只用前 10 個特徵創建交互
        )
        
        print(f"\n最終特徵數: {X_with_interactions.shape[1]}")
        
        return X_with_interactions

# 使用範例
if __name__ == "__main__":
    # 模擬數據
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'sensor_{i}' for i in range(n_features)]
    )
    
    # 創建有意義的目標
    important_features = [0, 1, 2, 5, 10]
    y = (X[f'sensor_{important_features[0]}'] + 
         X[f'sensor_{important_features[1]}'] * 2 + 
         np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    # 特徵工程
    import os
    os.makedirs('outputs', exist_ok=True)
    
    engineer = FeatureEngineer()
    X_engineered = engineer.engineer_features(X, y, method='rf', top_k=30)
    
    print(f"\n工程後特徵: {X_engineered.columns.tolist()[:10]}")