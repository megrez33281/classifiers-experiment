# PR_HW1: 分類器比較分析實驗

本專案旨在根據課程要求，對三種不同的機器學習分類器（KNN, Random Forest, SVM）在四個不同的數據集上進行系統性的性能比較與分析。

## 程式碼架構

- `main.py`: 主執行腳本，負責協調整個實驗流程，包括數據加載、模型訓練、超參數搜索和評估。
- `utils.py`: 工具模組，提供加載所有數據集的統一接口。
- `classifiers.py`: 分類器模組，封裝了所有分類器，並內建了數據標準化 Pipeline。
- `evaluation.py`: 評估模組，提供繪製混淆矩陣、ROC 曲線和性能比較長條圖的功能。
- `data/`: 存放本地數據集的資料夾。
- `plots/`: 存放所有生成圖表的資料夾。
- `Readme.md`: 本說明檔案，記錄專案細節與成果。

## 實驗設計

- **分類器:**
  1. K-Nearest Neighbors (KNN)
  2. Random Forest
  3. Support Vector Machine (SVM)

- **數據集:**
  1. **二元分類:**
     - Breast Cancer Wisconsin
     - Banknote Authentication
  2. **多類別分類:**
     - Digits Dataset
     - Dry Bean Dataset

## 最終實驗結果

| dataset       | classifier    | best_params                              |   mean_accuracy |   mean_precision |   mean_recall |   mean_f1_score |      auc |
|:--------------|:--------------|:-----------------------------------------|----------------:|-----------------:|--------------:|----------------:|---------:|
| breast_cancer | KNN           | {'knn__n_neighbors': 7}                  |        0.971429 |         0.972631 |      0.966512 |        0.969193 | 0.988426 |
| breast_cancer | Random Forest | {'rf__n_estimators': 200}                |        0.958242 |         0.955545 |      0.955986 |        0.955356 | 0.993221 |
| breast_cancer | SVM           | {'svm__C': 0.1, 'svm__kernel': 'linear'} |        0.978022 |         0.980659 |      0.972962 |        0.976215 | 0.993717 |
| banknote      | KNN           | {'knn__n_neighbors': 3}                  |        0.998178 |         0.997969 |      0.998361 |        0.998157 | 1        |
| banknote      | Random Forest | {'rf__n_estimators': 100}                |        0.994529 |         0.994112 |      0.994874 |        0.994468 | 1        |
| banknote      | SVM           | {'svm__C': 10, 'svm__kernel': 'rbf'}     |        1        |         1        |      1        |        1        | 1        |
| digits        | KNN           | {'knn__n_neighbors': 5}                  |        0.97493  |         0.975774 |      0.974852 |        0.974785 | 0.995028 |
| digits        | Random Forest | {'rf__n_estimators': 50}                 |        0.976326 |         0.976895 |      0.976226 |        0.976044 | 0.99866  |
| digits        | SVM           | {'svm__C': 1, 'svm__kernel': 'rbf'}      |        0.983297 |         0.983805 |      0.983221 |        0.983219 | 0.999454 |
| dry_bean      | KNN           | {'knn__n_neighbors': 7}                  |        0.923861 |         0.937511 |      0.934165 |        0.9356   | 0.985772 |
| dry_bean      | Random Forest | {'rf__n_estimators': 200}                |        0.925331 |         0.937882 |      0.934889 |        0.936256 | 0.993243 |
| dry_bean      | SVM           | {'svm__C': 10, 'svm__kernel': 'rbf'}     |        0.933229 |         0.945989 |      0.942785 |        0.944291 | 0.99459  |