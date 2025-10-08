import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """
    計算並繪製混淆矩陣，並將其儲存為圖片檔案。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩陣已儲存至: {save_path}")

def plot_roc_curves(results, title: str, save_path: str):
    """
    在同一張圖上繪製多個分類器的ROC曲線，並儲存為檔案。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    for y_true, y_scores, name in results:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    plt.savefig(save_path)
    plt.close()
    print(f"ROC曲線圖已儲存至: {save_path}")

def plot_metric_comparison(results, metric_name: str, title: str, save_path: str):
    """
    繪製一個長條圖來比較不同分類器在某個指標上的表現。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.xlabel('Classifier')
    plt.ylabel(metric_name)
    plt.title(title)

    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        plt.ylim([min_score * 0.95 - 0.05, max_score * 1.05 + 0.05])
    else:
        plt.ylim([min_score - (max_score-min_score)*0.1, max_score + (max_score-min_score)*0.1])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', ha='center', va='bottom')

    plt.savefig(save_path)
    plt.close()
    print(f"{metric_name} 比較圖已儲存至: {save_path}")