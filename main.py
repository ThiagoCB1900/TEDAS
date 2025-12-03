from genetic_algorithm import GA

lista = ["RandomForestClassifier", "LogisticRegression", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

GA.definirXY('datasets/mimic IV.parquet')

for item in lista:
    print(item)
    ga = GA(10, 10, item)
    ga.executar()
