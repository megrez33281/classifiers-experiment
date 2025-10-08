
import numpy as np
import pandas as pd
import warnings
import os
from utils import load_dataset
from classifiers import BaseClassifierWrapper, KNeighborsClassifierWrapper, RandomForestClassifierWrapper, SupportVectorMachineWrapper
from evaluation import plot_confusion_matrix, plot_roc_curves, plot_metric_comparison
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """主函式，執行所有數據集的超參數搜索和最終模型評估。"""
    
    os.makedirs('plots', exist_ok=True)
    dataset_names = ['breast_cancer', 'banknote', 'digits', 'dry_bean']
    final_results = []

    for dataset_name in dataset_names:
        print(f"\n{'='*50}")
        print(f"處理數據集: {dataset_name}")
        print(f"{'='*50}")

        X, y = load_dataset(dataset_name)
        if X is None: continue
        
        is_binary = len(np.unique(y)) == 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

        for name, clf_wrapper in classifiers.items():
            print(f"\n--- 為 {name} 執行 GridSearchCV on {dataset_name} ---")
            
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision_macro',
                'recall': 'recall_macro',
                'f1_score': 'f1_macro'
            }
            grid_search = GridSearchCV(clf_wrapper.model, param_grids[name], cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            print(f"找到的最佳參數: {grid_search.best_params_}")
            
            best_index = grid_search.best_index_
            mean_accuracy = grid_search.cv_results_['mean_test_accuracy'][best_index]
            mean_precision = grid_search.cv_results_['mean_test_precision'][best_index]
            mean_recall = grid_search.cv_results_['mean_test_recall'][best_index]
            mean_f1_score = grid_search.cv_results_['mean_test_f1_score'][best_index]

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_proba, y_dec = BaseClassifierWrapper._get_scores(best_model, X_test)

            # 計算 AUC
            auc_score = None
            if y_proba is not None:
                if is_binary:
                    auc_score = roc_auc_score(y_test, y_proba[:, 1])
                    roc_results_for_dataset.append((y_test, y_proba[:, 1], name))
                else:
                    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            
            print(f"平均CV準確率: {mean_accuracy:.4f}, 平均CV精確率: {mean_precision:.4f}, 平均CV召回率: {mean_recall:.4f}, 平均CV F1分數: {mean_f1_score:.4f}")
            if auc_score is not None: print(f"測試集上的 AUC: {auc_score:.4f}")

            final_results.append({
                'dataset': dataset_name, 'classifier': name, 'best_params': str(grid_search.best_params_),
                'mean_accuracy': mean_accuracy, 'mean_precision': mean_precision,
                'mean_recall': mean_recall, 'mean_f1_score': mean_f1_score, 'auc': auc_score
            })
            metrics_for_dataset['Accuracy'][name] = mean_accuracy
            if auc_score is not None: metrics_for_dataset['AUC'][name] = auc_score

            cm_title = f'CM for {name} on {dataset_name}\nBest Params: {grid_search.best_params_}'
            cm_save_path = f'plots/CM_{name}_{dataset_name}.png'
            plot_confusion_matrix(y_test, y_pred, title=cm_title, save_path=cm_save_path)

        if is_binary:
            plot_roc_curves(roc_results_for_dataset, title=f'ROC Curves on {dataset_name} Dataset', save_path=f'plots/ROC_{dataset_name}.png')
        
        plot_metric_comparison(metrics_for_dataset['Accuracy'], metric_name='Mean Accuracy (CV)', title=f'Mean Accuracy Comparison on {dataset_name}', save_path=f'plots/ACC_BAR_{dataset_name}.png')
        if metrics_for_dataset['AUC']:
            plot_metric_comparison(metrics_for_dataset['AUC'], metric_name='AUC Score', title=f'AUC Score Comparison on {dataset_name}', save_path=f'plots/AUC_BAR_{dataset_name}.png')

    # 5. 匯出最終結果
    print(f"\n\n{'='*50}")
    print("所有實驗已完成 - 匯出結果摘要")
    print(f"{'='*50}")
    
    results_df = pd.DataFrame(final_results)
    results_df.to_csv('results_summary.csv', index=False)
    print("結果摘要已儲存至 results_summary.csv")

    # 更新 Readme.md
    readme_table = results_df.to_markdown(index=False)
    print("\n" + readme_table)
    try:
        with open('Readme.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        final_readme_content = readme_content.split("## 最終實驗結果")[0] + "## 最終實驗結果\n\n" + readme_table
        with open('Readme.md', 'w', encoding='utf-8') as f:
            f.write(final_readme_content)
        print("\nReadme.md 已更新完畢。\n")
    except FileNotFoundError:
        print("\n錯誤: Readme.md 未找到，無法自動更新。\n")

if __name__ == "__main__":
    main()
