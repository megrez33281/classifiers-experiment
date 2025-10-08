from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def visualize_pca(X, y, dataset_name):
    """將資料降維至2D並以PCA可視化，圖片儲存在plots資料夾中"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # === 新增：將字串標籤轉成整數 ===
    if y.dtype == object or isinstance(y[0], str):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
    else:
        y_encoded = y
        class_names = np.unique(y)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_encoded, cmap='Spectral', alpha=0.7, s=30
    )
    plt.title(f"PCA Visualization - {dataset_name}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # 新增：顯示圖例（每個顏色對應的類別）
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 儲存圖
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"PCA_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA 圖已儲存至: {save_path}")


def visualize_lda(X, y, dataset_name: str):
    """
    使用 LDA（Linear Discriminant Analysis）進行有監督降維，
    並將結果以散點圖可視化，輸出至 plots 資料夾。

    Parameters
    ----------
    X : np.ndarray
        特徵矩陣
    y : np.ndarray
        標籤向量（可為數字或字串）
    dataset_name : str
        資料集名稱（將用於檔名與標題）
    """
    # === 確保標籤是數字 ===
    if y.dtype == object or isinstance(y[0], str):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
    else:
        y_encoded = y
        class_names = np.unique(y)

    n_classes = len(np.unique(y_encoded))

    # === LDA 降維 ===
    n_components = 2 if n_classes > 2 else 1
    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X, y_encoded)

    # === 畫圖 ===
    plt.figure(figsize=(8, 6))
    if n_components == 2:
        scatter = plt.scatter(
            X_lda[:, 0], X_lda[:, 1],
            c=y_encoded, cmap="Spectral", alpha=0.7, s=30
        )
        plt.xlabel("LDA Component 1")
        plt.ylabel("LDA Component 2")
    else:
        scatter = plt.scatter(
            X_lda[:, 0], np.zeros_like(X_lda),
            c=y_encoded, cmap="Spectral", alpha=0.7, s=30
        )
        plt.xlabel("LDA Component 1")
        plt.yticks([])

    plt.title(f"LDA Visualization - {dataset_name}")

    # 圖例
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes",
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # === 儲存圖像 ===
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"LDA_{dataset_name}.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"LDA 圖已儲存至: {save_path}")