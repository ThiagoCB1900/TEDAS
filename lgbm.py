from lightgbm import LGBMClassifier
from algoritmo import Algoritmo


class LGBM(Algoritmo):
    def __str__(self):
        return "LIGHT GBM CLASSIFIER"

    def definirMinimo(self):
        return [100, 0.01, 16, 3, 0.5, 0.5, 5, 0.0, 0.0]

    def definirMaximo(self):
        return [1500, 0.2, 64, 10, 1, 1, 30, 3, 3]

    def gerarModelo(self, genes):
        return LGBMClassifier(
            n_estimators=genes[0],
            learning_rate=genes[1],
            num_leaves=genes[2],
            max_depth=genes[3],
            subsample=genes[4],
            colsample_bytree=genes[5],
            min_child_samples=genes[6],
            reg_lambda=genes[7],
            reg_alpha=genes[8],
            n_jobs=-1,
            device="cpu",
            boosting_type="gbdt",
            class_weight="balanced",
            importance_type="gain"
        )
