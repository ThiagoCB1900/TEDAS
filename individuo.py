class Individuo:
    def __init__(self, cromossomos):
        self.cromossomos = cromossomos
        self.acuracia = None
        self.precisao = None
        self.recall = None

    def retornarPerformance(self):
        return round((self.acuracia + 2 * self.precisao + 4 * self.recall) / 7, 1)

    def __str__(self):
        performance = self.retornarPerformance()

        return ('Cromossomos: {} -> Acuracia: {}%, Precisao: {}%, Recall: {}%, Performance: {}%\n'.
                format(self.cromossomos, self.acuracia, self.precisao, self.recall, performance))
