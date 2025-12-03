from abc import ABC, abstractmethod
from random import randint, uniform


class Algoritmo(ABC):
    def __init__(self):
        self.min = self.definirMinimo()
        self.max = self.definirMaximo()
        self.qtd_genes = len(self.max)
        self.indices_float = self.definirIndices()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def definirMinimo(self):
        pass

    @abstractmethod
    def definirMaximo(self):
        pass

    @abstractmethod
    def gerarModelo(self, pos):
        pass

    def definirIndices(self):
        indices = []

        for i in range(self.qtd_genes):
            if not isinstance(self.min[i], int):
                indices.append(i)

        return indices

    def gerarCromossomo(self):
        cromossomo = []

        for i in range(self.qtd_genes):
            if i in self.indices_float:
                cromossomo.append(round(uniform(self.min[i], self.max[i]), 4))
            else:
                cromossomo.append(randint(self.min[i], self.max[i]))

        return cromossomo

    def gerarGene(self, i):
        if i in self.indices_float:
            return round(uniform(self.min[i], self.max[i]), 4)

        return randint(self.min[i], self.max[i])
