from lightgbm import LGBMClassifier
from algoritmo import Algoritmo


class LGBM(Algoritmo):
    def __str__(self):
        return "LIGHT GBM CLASSIFIER"

    def definirMinimo(self):
        return [100, 0.01, 16, 3, 0.5, 0.5, 5, 0.0, 0.0]

    def definirMaximo(self):
        return [1500, 0.2, 64, 10, 1, 1, 30, 3, 3]

    def gerarModelo(self, pos):
        return LGBMClassifier(
            n_estimators=pos[0],
            learning_rate=pos[1],
            num_leaves=pos[2],
            max_depth=pos[3],
            subsample=pos[4],
            colsample_bytree=pos[5],
            min_child_samples=pos[6],
            reg_lambda=pos[7],
            reg_alpha=pos[8],
            n_jobs=-1,
            device="cpu",
            boosting_type="gbdt",
            class_weight="balanced",
            importance_type="gain"
        )
