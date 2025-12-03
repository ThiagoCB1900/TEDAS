from algoritmo import Algoritmo
from sklearn.neighbors import KNeighborsClassifier


class KNN(Algoritmo):
    def __str__(self):
        return "K NEIGHBORS CLASSIFIER"

    def definirMinimo(self):
        return [1, 0, 0, 10]

    def definirMaximo(self):
        return [30, 1, 2, 100]

    def gerarModelo(self, genes):
        lista_weights = ['uniform', 'distance']
        lista_metric = ['euclidean', 'manhattan', 'minkowski']

        return KNeighborsClassifier(n_neighbors=genes[0], weights=lista_weights[genes[1]],
                                    metric=lista_metric[genes[2]], leaf_size=genes[3], n_jobs=-1)
