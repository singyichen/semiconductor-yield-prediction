"""
API 測試腳本
用於測試 API 的各種功能
"""

import requests
import json
import numpy as np
from typing import Dict, List

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_health(self):
        """測試健康檢查"""
        print("=" * 60)
        print("測試健康檢查...")
        print("=" * 60)
        
        response = requests.get(f"{self.base_url}/health")
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        return response.status_code == 200
    
    def test_model_info(self):
        """測試模型資訊"""
        print("\n" + "=" * 60)
        print("測試模型資訊...")
        print("=" * 60)
        
        response = requests.get(f"{self.base_url}/model/info")
        print(f"狀態碼: {response.status_code}")
        print(f"回應: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        return response.status_code == 200
    
    def generate_random_features(self, n_features: int = 50) -> Dict[str, float]:
        """生成隨機特徵數據"""
        return {
            f"feature_{i}": float(np.random.randn())
            for i in range(n_features)
        }
    
    def test_single_prediction(self):
        """測試單筆預測"""
        print("\n" + "=" * 60)
        print("測試單筆預測...")
        print("=" * 60)
        
        # 生成測試數據
        features = self.generate_random_features()
        
        response = requests.post(
            f"{self.base_url}/predict",
            json={"features": features}
        )
        
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"預測結果: {result['prediction']}")
            print(f"不良品機率: {result['defect_probability']:.4f}")
            print(f"信心度: {result['confidence']:.4f}")
            print(f"告警: {result['alert']}")
        else:
            print(f"錯誤: {response.text}")
        
        return response.status_code == 200
    
    def test_batch_prediction(self, batch_size: int = 10):
        """測試批次預測"""
        print("\n" + "=" * 60)
        print(f"測試批次預測 (批次大小: {batch_size})...")
        print("=" * 60)
        
        # 生成批次測試數據
        batch = [
            self.generate_random_features()
            for _ in range(batch_size)
        ]
        
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"batch": batch}
        )
        
        print(f"狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"總數: {result['total']}")
            print(f"不良品數: {result['defect_count']}")
            print(f"不良率: {result['defect_rate']:.2%}")
            
            # 顯示前 3 筆預測
            print("\n前 3 筆預測:")
            for i, pred in enumerate(result['predictions'][:3], 1):
                print(f"  {i}. {pred['prediction']} (機率: {pred['defect_probability']:.4f})")
        else:
            print(f"錯誤: {response.text}")
        
        return response.status_code == 200
    
    def test_error_handling(self):
        """測試錯誤處理"""
        print("\n" + "=" * 60)
        print("測試錯誤處理...")
        print("=" * 60)
        
        # 測試缺少特徵
        print("\n1. 測試缺少特徵:")
        response = requests.post(
            f"{self.base_url}/predict",
            json={"features": {"sensor_0": 1.0}}  # 只有一個特徵
        )
        print(f"   狀態碼: {response.status_code} (預期 400)")
        
        # 測試無效格式
        print("\n2. 測試無效格式:")
        response = requests.post(
            f"{self.base_url}/predict",
            json={"invalid_key": {}}
        )
        print(f"   狀態碼: {response.status_code} (預期 422)")
        
        return True
    
    def run_all_tests(self):
        """運行所有測試"""
        print("\n" + "🧪" * 30)
        print("開始 API 測試")
        print("🧪" * 30 + "\n")
        
        results = {
            "健康檢查": self.test_health(),
            "模型資訊": self.test_model_info(),
            "單筆預測": self.test_single_prediction(),
            "批次預測": self.test_batch_prediction(),
            "錯誤處理": self.test_error_handling()
        }
        
        # 總結
        print("\n" + "=" * 60)
        print("測試總結")
        print("=" * 60)
        
        for test_name, passed in results.items():
            status = "✅ 通過" if passed else "❌ 失敗"
            print(f"{test_name}: {status}")
        
        total = len(results)
        passed = sum(results.values())
        print(f"\n總計: {passed}/{total} 通過")

if __name__ == "__main__":
    # 確保 API 服務正在運行
    tester = APITester(base_url="http://localhost:8000")
    tester.run_all_tests()