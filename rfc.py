from sklearn.ensemble import RandomForestClassifier
from algoritmo import Algoritmo


class RFC(Algoritmo):
    def __str__(self):
        return 'RANDOM FOREST CLASSIFIER'

    def definirMinimo(self):
        return [100, 5, 2, 1, 0]

    def definirMaximo(self):
        return [400, 25, 10, 5, 2]

    def gerarModelo(self, genes):
        max_features_list = ["sqrt", "log2", 0.8]

        return RandomForestClassifier(
            n_estimators=genes[0],
            max_depth=genes[1],
            min_samples_split=genes[2],
            min_samples_leaf=genes[3],
            max_features=max_features_list[genes[4]],
            n_jobs=-1,
            class_weight="balanced",
            bootstrap=True
        )
