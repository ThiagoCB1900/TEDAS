class Individuo:
    def __init__(self, cromossomo):
        self.cromossomo = cromossomo
        self.performance = 0
        self.acuracia = None
        self.precisao = None
        self.recall = None

    def retornarPerformance(self):
        return round((self.acuracia + 2 * self.precisao + 3 * self.recall) / 6, 1)

    def __str__(self):
        return ('Cromossomo: {} -> Acuracia: {}%, Precisao: {}%, Recall: {}%, Performance: {}%'.
                format(self.cromossomo, self.acuracia, self.precisao, self.recall, self.performance))
