
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class BaseClassifierWrapper:
    """提供一個通用的方法來獲取模型的預測分數"""
    @staticmethod
    def _get_scores(model, X):
        """優先嘗試 decision_function，其次是 predict_proba（回傳每個類別的機率）。"""
        # 先嘗試取得原生的分數，若不行再嘗試獲取每個類別的機率
        decision_values = None
        proba_values = None
        if hasattr(model, 'decision_function'):
            try:
                decision_values = model.decision_function(X)
            except Exception:
                pass # 忽略錯誤
        if hasattr(model, 'predict_proba'):
            try:
                proba_values = model.predict_proba(X)
            except Exception:
                pass # 忽略錯誤
        return proba_values, decision_values

class KNeighborsClassifierWrapper(BaseClassifierWrapper):
    """KNN 分類器的封裝。"""
    def __init__(self, n_neighbors=5):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test):
        y_pred = self.model.predict(X_test)
        proba, dec = self._get_scores(self.model, X_test)
        return y_pred, proba, dec

class RandomForestClassifierWrapper(BaseClassifierWrapper):
    """Random Forest 分類器的封裝。"""
    def __init__(self, n_estimators=100, random_state=42):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
        ])
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test):
        y_pred = self.model.predict(X_test)
        proba, dec = self._get_scores(self.model, X_test)
        return y_pred, proba, dec

class SupportVectorMachineWrapper(BaseClassifierWrapper):
    """SVM 分類器的封裝。"""
    def __init__(self, C=1.0, kernel='rbf', probability=True, random_state=42):
        # 註：probability=True 讓 SVC 能夠使用 predict_proba，但會增加訓練時間。
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=C, kernel=kernel, probability=probability, random_state=random_state))
        ])
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test):
        y_pred = self.model.predict(X_test)
        proba, dec = self._get_scores(self.model, X_test)
        return y_pred, proba, dec
