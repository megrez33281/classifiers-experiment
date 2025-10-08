
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits

def load_dataset(name: str):
    """
    根據名稱載入指定的數據集，採用本地優先策略。

    Args:
        name (str): 數據集名稱。可選值: 
                      'breast_cancer', 'digits', 'banknote', 'dry_bean'

    Returns:
        tuple: (X, y) or (None, None) if loading fails.
               X: 特徵數據 (np.ndarray)
               y: 標籤 (np.ndarray)
    """
    if name == 'breast_cancer':
        data = load_breast_cancer()
        return data.data, data.target
    
    elif name == 'digits':
        data = load_digits()
        return data.data, data.target
        
    elif name == 'banknote':
        local_path = 'data/data_banknote_authentication.txt'
        if os.path.exists(local_path):
            print(f"從本地路徑加載 '{name}' 數據集: {local_path}")
            df = pd.read_csv(local_path, header=None)
        else:
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
            print(f"本地檔案未找到，嘗試從網路後備連結加載: {url}")
            try:
                df = pd.read_csv(url, header=None)
            except Exception as e:
                print(f"無法從網路讀取 banknote 數據，請手動下載至 '{local_path}'。錯誤: {e}")
                return None, None
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    elif name == 'dry_bean':
        # 根據建議，我們預期用戶已將解壓縮後的檔案放在 data/DryBeanDataset/ 目錄下
        local_path = 'data/DryBeanDataset/Dry_Bean_Dataset.xlsx'
        if not os.path.exists(local_path):
            print(f"錯誤: 找不到 '{local_path}'。")
            print("請確認您已手動從Kaggle下載數據集，並將其解壓縮後的 .xlsx 檔案放置在正確的路徑中。")
            return None, None
        
        print(f"從本地路徑加載 '{name}' 數據集: {local_path}")
        df = pd.read_excel(local_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
        
    else:
        raise ValueError(f"未知的數據集名稱: {name}。請從 'breast_cancer', 'digits', 'banknote', 'dry_bean' 中選擇。")

