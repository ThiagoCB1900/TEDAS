from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from individuo import Individuo
from pandas import read_parquet
from random import randint
from copy import deepcopy

from logistic_regression import LR
from lgbm import LGBM
from knn import KNN
from rfc import RFC
from xgb import XGB
from cbc import CBC


def definirAlgoritmo(tipo_algoritmo):
    if tipo_algoritmo == "KNeighborsClassifier":
        return KNN()
    if tipo_algoritmo == "RandomForestClassifier":
        return RFC()
    if tipo_algoritmo == "LogisticRegression":
        return LR()
    if tipo_algoritmo == "XGBClassifier":
        return XGB()
    if tipo_algoritmo == "LGBMClassifier":
        return LGBM()
    if tipo_algoritmo == "CatBoostClassifier":
        return CBC()

    return None


class GA:
    x_treinamento = None
    y_treinamento = None
    x_validacao = None
    y_validacao = None
    x_teste = None
    y_teste = None

    def __init__(self, num_individuos, num_populacoes, chance_de_mutar, tipo_algoritmo):
        self.num_individuos = num_individuos
        self.num_populacoes = num_populacoes
        self.chance_de_mutar = chance_de_mutar
        self.algoritmo = definirAlgoritmo(tipo_algoritmo)
        self.qtd_cromossomos = self.algoritmo.qtd_cromossomos
        self.melhor_individuo = None
        self.melhor_performance = 0

    @staticmethod
    def definirXY(endereco_parquet):
        dataframe = read_parquet(endereco_parquet)
        x = dataframe.iloc[:, :-1].values
        y = dataframe.iloc[:, -1].values
        scaler = MinMaxScaler()

        x_train, x_rest, GA.y_treinamento, y_rest = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y)

        x_val, x_test, GA.y_validacao, GA.y_teste = train_test_split(
            x_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest)

        GA.x_treinamento = scaler.fit_transform(x_train)
        GA.x_validacao = scaler.transform(x_val)
        GA.x_teste = scaler.transform(x_test)

    def gerarPopulacao(self):
        populacao = []

        for _ in range(self.num_individuos):
            populacao.append(Individuo(self.algoritmo.gerarCromossomos()))

        return populacao

    def ajustarMetricas(self, individuo):
        modelo = self.algoritmo.gerarModelo(individuo.cromossomos)
        modelo.fit(GA.x_treinamento, GA.y_treinamento)
        previsoes = modelo.predict(GA.x_validacao)

        individuo.acuracia = round(accuracy_score(GA.y_validacao, previsoes) * 100, 1)
        individuo.precisao = round(precision_score(GA.y_validacao, previsoes, zero_division=0) * 100, 1)
        individuo.recall = round(recall_score(GA.y_validacao, previsoes, zero_division=0) * 100, 1)

    def definirIndividuosTorneio(self, populacao):
        individuos = []

        for _ in range(4):
            individuos.append(populacao[randint(0, self.num_individuos - 1)])

        return individuos[0], individuos[1], individuos[2], individuos[3]

    def selecionarPorTorneio(self, populacao):
        pai1, pai2, mae1, mae2 = self.definirIndividuosTorneio(populacao)

        if pai1.retornarPerformance() > pai2.retornarPerformance():
            pai = pai1
        else:
            pai = pai2

        if mae1.retornarPerformance() > mae2.retornarPerformance():
            mae = mae1
        else:
            mae = mae2

        return deepcopy(pai), deepcopy(mae)

    def fazerCrossover(self, pai, mae):
        for i in range(randint(1, self.qtd_cromossomos)):
            pai.cromossomos[i], mae.cromossomos[i] = mae.cromossomos[i], pai.cromossomos[i]

        return pai, mae

    def mutar(self, filho, filha):
        for i in range(self.qtd_cromossomos):
            if randint(1, 100) <= self.chance_de_mutar:
                filho.cromossomos[i] = self.algoritmo.gerarCromossomo(i)
            if randint(1, 100) <= self.chance_de_mutar:
                filha.cromossomos[i] = self.algoritmo.gerarCromossomo(i)

        self.ajustarMetricas(filho)
        self.ajustarMetricas(filha)

        return [filho, filha]

    def executar(self):
        populacao = self.gerarPopulacao()

        for individuo in populacao:
            self.ajustarMetricas(individuo)

        with open(str(self.algoritmo) + ".txt", "w") as file:
            for num_populacao in range(self.num_populacoes):
                file.write("{}* Populacao:\n".format(num_populacao + 1))
                nova_populacao = []

                for _ in range(self.num_individuos // 2):
                    pai, mae = self.selecionarPorTorneio(populacao)
                    filho, filha = self.fazerCrossover(pai, mae)
                    irmaos = self.mutar(filho, filha)

                    for irmao in irmaos:
                        file.write(str(irmao))

                        if irmao.retornarPerformance() > self.melhor_performance:
                            self.melhor_performance = irmao.retornarPerformance()
                            self.melhor_individuo = deepcopy(irmao)

                    nova_populacao.extend(irmaos)

                file.write('\nMelhores ' + str(self.melhor_individuo) + "\n")
                populacao = nova_populacao
