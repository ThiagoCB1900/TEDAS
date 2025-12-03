from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from random import randint, uniform
from individuo import Individuo
from pandas import read_parquet
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

    def __init__(self, num_individuos, num_populacoes, tipo_algoritmo):
        self.num_individuos = num_individuos
        self.num_populacoes = num_populacoes
        self.algoritmo = definirAlgoritmo(tipo_algoritmo)
        self.qtd_genes = self.algoritmo.qtd_genes
        self.chance_de_mutar = 1 / self.qtd_genes
        self.melhor_individuo = None

        if num_individuos % 2 == 1:
            self.num_individuos += 1

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
            populacao.append(Individuo(self.algoritmo.gerarCromossomo()))

        return populacao

    def ajustarMetricas(self, individuo):
        modelo = self.algoritmo.gerarModelo(individuo.cromossomo)
        modelo.fit(GA.x_treinamento, GA.y_treinamento)
        previsoes = modelo.predict(GA.x_validacao)

        individuo.acuracia = round(accuracy_score(GA.y_validacao, previsoes) * 100, 1)
        individuo.precisao = round(precision_score(GA.y_validacao, previsoes, zero_division=0) * 100, 1)
        individuo.recall = round(recall_score(GA.y_validacao, previsoes, zero_division=0) * 100, 1)

        individuo.performance = individuo.retornarPerformance()

    def definirIndividuosTorneio(self, populacao):
        individuos = []

        for _ in range(4):
            individuos.append(populacao[randint(0, self.num_individuos - 1)])

        return individuos[0], individuos[1], individuos[2], individuos[3]

    def selecionarPorTorneio(self, populacao):
        pai1, pai2, mae1, mae2 = self.definirIndividuosTorneio(populacao)

        if pai1.performance > pai2.performance:
            pai = pai1
        else:
            pai = pai2

        if mae1.performance > mae2.performance:
            mae = mae1
        else:
            mae = mae2

        return deepcopy(pai), deepcopy(mae)

    def fazerCrossover(self, pai, mae):
        indices = list(range(self.qtd_genes))

        for _ in range(randint(1, self.qtd_genes - 1)):
            del indices[randint(0, len(indices) - 1)]

        for i in indices:
            pai.cromossomo[i], mae.cromossomo[i] = mae.cromossomo[i], pai.cromossomo[i]

        return pai, mae

    def mutar(self, filho, filha):
        for i in range(self.qtd_genes):
            if uniform(0, 1) <= self.chance_de_mutar:
                filho.cromossomo[i] = self.algoritmo.gerarGene(i)
            if uniform(0, 1) <= self.chance_de_mutar:
                filha.cromossomo[i] = self.algoritmo.gerarGene(i)

        self.ajustarMetricas(filho)
        self.ajustarMetricas(filha)

        return [filho, filha]

    def avaliarMelhorNoTeste(self, file):
        modelo = self.algoritmo.gerarModelo(self.melhor_individuo.cromossomo)
        modelo.fit(GA.x_treinamento, GA.y_treinamento)
        previsoes = modelo.predict(GA.x_teste)

        acuracia = round(accuracy_score(GA.y_teste, previsoes) * 100, 1)
        precisao = round(precision_score(GA.y_teste, previsoes, zero_division=0) * 100, 1)
        recall = round(recall_score(GA.y_teste, previsoes, zero_division=0) * 100, 1)
        media = round((acuracia + 2 * precisao + 3 * recall) / 6, 1)

        file.write("\nPerformance no Teste -> Acuracia: {}%, Precisao: {}%, Recall: {}%, Media: {}%".
                   format(acuracia, precisao, recall, media))

    def executar(self):
        self.melhor_individuo = Individuo(None)
        populacao = self.gerarPopulacao()

        for individuo in populacao:
            self.ajustarMetricas(individuo)

        with open("resultados/" + str(self.algoritmo) + ".txt", "w") as file:
            for num_populacao in range(self.num_populacoes):
                file.write("{}* Populacao:\n".format(num_populacao + 1))
                nova_populacao = []

                for _ in range(self.num_individuos // 2):
                    pai, mae = self.selecionarPorTorneio(populacao)
                    filho, filha = self.fazerCrossover(pai, mae)
                    irmaos = self.mutar(filho, filha)

                    for irmao in irmaos:
                        file.write(str(irmao) + '\n')

                        if irmao.performance > self.melhor_individuo.performance:
                            self.melhor_individuo = deepcopy(irmao)

                    nova_populacao.extend(irmaos)

                file.write('\nMelhor ' + str(self.melhor_individuo) + "\n\n")
                populacao = nova_populacao

            self.avaliarMelhorNoTeste(file)
