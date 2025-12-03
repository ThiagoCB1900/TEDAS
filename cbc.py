from catboost import CatBoostClassifier
from algoritmo import Algoritmo


class CBC(Algoritmo):
    def __str__(self):
        return "CAT BOOST CLASSIFIER"

    def definirMinimo(self):
        return [50, 0.0005, 3, 0.1, 0.0, 0.0]

    def definirMaximo(self):
        return [1500, 0.2, 10, 10, 5, 1]

    def gerarModelo(self, genes):
        return CatBoostClassifier(
            iterations=genes[0],
            learning_rate=genes[1],
            depth=genes[2],
            l2_leaf_reg=genes[3],
            random_strength=genes[4],
            bagging_temperature=genes[5],
            task_type="CPU",
            thread_count=-1,
            verbose=False,
        )
