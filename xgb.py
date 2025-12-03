from xgboost import XGBClassifier
from algoritmo import Algoritmo


class XGB(Algoritmo):
    def __str__(self):
        return "XGBOOST CLASSIFIER"

    def definirMinimo(self):
        return [100, 0.01, 3, 0.5, 0.5, 0, 0, 0, 1]

    def definirMaximo(self):
        return [1000, 0.3, 12, 1, 1, 5, 3, 3, 10]

    def gerarModelo(self, genes):
        return XGBClassifier(
            n_estimators=genes[0],
            learning_rate=genes[1],
            max_depth=genes[2],
            subsample=genes[3],
            colsample_bytree=genes[4],
            gamma=genes[5],
            reg_lambda=genes[6],
            reg_alpha=genes[7],
            min_child_weight=genes[8],
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )
