from algoritmo import Algoritmo
from sklearn.linear_model import LogisticRegression


class LR(Algoritmo):
    def __str__(self):
        return 'LOGISTIC REGRESSION'

    def definirMinimo(self):
        return [0.01, 100, 0, 0]

    def definirMaximo(self):
        return [10, 500, 1, 1]

    def gerarModelo(self, genes):
        solvers = ['liblinear', 'saga']
        penalties = ['l1', 'l2']

        return LogisticRegression(C=genes[0], max_iter=genes[1], solver=solvers[genes[2]], penalty=penalties[genes[3]])
