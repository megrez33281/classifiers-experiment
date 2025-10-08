import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ================================================================
#  混淆矩陣繪圖
# ================================================================
def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """
    計算並繪製混淆矩陣，並將其儲存為圖片檔案。
    支援任意類別數量與格式（含字串標籤）。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues", cbar=True, ax=ax,
                xticklabels=labels, yticklabels=labels)

    # === 手動標註每格 ===
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(j + 0.5, i + 0.5, f"{value:d}",
                    ha="center", va="center", color=color, fontsize=12)

    # === 額外資訊 ===
    acc = np.trace(cm) / np.sum(cm)
    total = np.sum(cm)

    ax.set_title(f"{title}\nAccuracy = {acc:.3f}, Total = {total}")
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩陣已儲存至: {save_path}")


# ================================================================
#  ROC 曲線繪圖
# ================================================================
def plot_roc_curves(results, title: str, save_path: str):
    """
    在同一張圖上繪製多個分類器的 ROC 曲線，並儲存為檔案。
    results 格式為 [(y_true, y_score, name), ...]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    for y_true, y_scores, name in results:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" ROC 曲線圖已儲存至: {save_path}")


# ================================================================
#  單一指標比較長條圖
# ================================================================
def plot_metric_comparison(results, metric_name: str, title: str, save_path: str):
    """
    繪製不同分類器在同一指標下的比較圖。
    results = {'SVM': 0.98, 'KNN': 0.95, ...}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.xlabel('Classifier')
    plt.ylabel(metric_name)
    plt.title(title)

    # 動態調整 Y 軸範圍
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        plt.ylim([min_score * 0.95 - 0.05, max_score * 1.05 + 0.05])
    else:
        plt.ylim([min_score - (max_score - min_score) * 0.1,
                  max_score + (max_score - min_score) * 0.1])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" {metric_name} 比較圖已儲存至: {save_path}")


# ================================================================
#  Cross-Validation 穩定性圖（新功能）
# ================================================================
def plot_cv_stability_bar(results_dict, metric_name: str, title: str, save_path: str):
    """
    繪製不同分類器的 Cross-Validation 平均分數 ± 標準差。
    results_dict = {
        'SVM': (mean, std),
        'KNN': (mean, std),
        ...
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names = list(results_dict.keys())
    means = [v[0] for v in results_dict.values()]
    stds = [v[1] for v in results_dict.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=6,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.85)
    plt.xlabel("Classifier")
    plt.ylabel(metric_name)
    plt.title(title)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i],
                 f"{means[i]:.4f}±{stds[i]:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" {metric_name} 穩定性比較圖已儲存至: {save_path}")
