import numpy as np
import pandas as pd
import warnings
import os
from utils import load_dataset
from classifiers import BaseClassifierWrapper, KNeighborsClassifierWrapper, RandomForestClassifierWrapper, SupportVectorMachineWrapper
from evaluation import plot_confusion_matrix, plot_roc_curves, plot_metric_comparison
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from visualize import visualize_pca, visualize_lda

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """主函式：使用 5-Fold Cross Validation 執行所有數據集的模型搜尋與評估"""
    
    os.makedirs('plots', exist_ok=True)
    dataset_names = ['breast_cancer', 'banknote', 'digits', 'dry_bean']
    final_results = []

    for dataset_name in dataset_names:
        print(f"\n{'='*50}")
        print(f"處理數據集: {dataset_name}")
        print(f"{'='*50}")

        # === 載入資料 ===
        X, y = load_dataset(dataset_name)
        visualize_pca(X, y, dataset_name)
        visualize_lda(X, y, dataset_name)
        if X is None:
            continue
        
        is_binary = len(np.unique(y)) == 2

        # === 參數設定 ===
        param_grids = {
            "KNN": {'knn__n_neighbors': [3, 5, 7]},
            "Random Forest": {'rf__n_estimators': [50, 100, 200]},
            "SVM": {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}
        }

        classifiers = {
            "KNN": KNeighborsClassifierWrapper(),
            "Random Forest": RandomForestClassifierWrapper(random_state=42),
            "SVM": SupportVectorMachineWrapper(random_state=42)
        }

        roc_results_for_dataset = []
        metrics_for_dataset = {'Accuracy': {}, 'AUC': {}}

        # === Cross-validation 分割設定 ===
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, clf_wrapper in classifiers.items():
            print(f"\n--- 為 {name} 執行 5-Fold Cross Validation on {dataset_name} ---")

            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision_macro',
                'recall': 'recall_macro',
                'f1_score': 'f1_macro'
            }

            # GridSearchCV for parameter tuning
            grid_search = GridSearchCV(
                clf_wrapper.model,
                param_grids[name],
                cv=cv,
                scoring=scoring,
                refit='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            print(f"找到的最佳參數: {grid_search.best_params_}")

            # Cross-validation 預測（用於 Confusion Matrix & ROC）
            y_pred = cross_val_predict(best_model, X, y, cv=cv)
            y_proba, _ = BaseClassifierWrapper._get_scores(best_model, X)

            # === 平均CV結果 ===
            mean_accuracy = grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]
            mean_precision = grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]
            mean_recall = grid_search.cv_results_['mean_test_recall'][grid_search.best_index_]
            mean_f1 = grid_search.cv_results_['mean_test_f1_score'][grid_search.best_index_]

            auc_score = None
            if y_proba is not None:
                if is_binary:
                    auc_score = roc_auc_score(y, y_proba[:, 1])
                    roc_results_for_dataset.append((y, y_proba[:, 1], name))
                else:
                    auc_score = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')

            print(f"平均CV準確率: {mean_accuracy:.4f}, 精確率: {mean_precision:.4f}, 召回率: {mean_recall:.4f}, F1: {mean_f1:.4f}")
            if auc_score is not None:
                print(f"整體資料AUC: {auc_score:.4f}")

            final_results.append({
                'dataset': dataset_name, 'classifier': name,
                'best_params': str(grid_search.best_params_),
                'mean_accuracy': mean_accuracy,
                'mean_precision': mean_precision,
                'mean_recall': mean_recall,
                'mean_f1_score': mean_f1,
                'auc': auc_score
            })
            metrics_for_dataset['Accuracy'][name] = mean_accuracy
            if auc_score is not None:
                metrics_for_dataset['AUC'][name] = auc_score

            # === 繪製混淆矩陣 ===
            cm = confusion_matrix(y, y_pred)
            cm_title = f'CM (5-Fold CV) for {name} on {dataset_name}\nBest Params: {grid_search.best_params_}'
            cm_save_path = f'plots/CM_{name}_{dataset_name}.png'
            plot_confusion_matrix(y, y_pred, title=cm_title, save_path=cm_save_path)

        # === 繪製 ROC 與比較圖 ===
        if is_binary:
            plot_roc_curves(roc_results_for_dataset, title=f'ROC Curves on {dataset_name} Dataset (5-Fold CV)',
                            save_path=f'plots/ROC_{dataset_name}.png')

        plot_metric_comparison(metrics_for_dataset['Accuracy'], metric_name='Mean Accuracy (5-Fold CV)',
                               title=f'Mean Accuracy Comparison on {dataset_name}', save_path=f'plots/ACC_BAR_{dataset_name}.png')

        if metrics_for_dataset['AUC']:
            plot_metric_comparison(metrics_for_dataset['AUC'], metric_name='AUC Score',
                                   title=f'AUC Score Comparison on {dataset_name}', save_path=f'plots/AUC_BAR_{dataset_name}.png')

    # === 匯出結果 ===
    print(f"\n\n{'='*50}")
    print("所有實驗已完成 - 匯出結果摘要")
    print(f"{'='*50}")

    results_df = pd.DataFrame(final_results)
    results_df.to_csv('results_summary.csv', index=False)
    print("結果摘要已儲存至 results_summary.csv")

if __name__ == "__main__":
    main()
