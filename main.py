from genetic_algorithm import GA

lista = ["KNeighborsClassifier", "RandomForestClassifier", "LogisticRegression",
         "XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

GA.definirXY('Heart.parquet')

ga = GA(10, 10, 10, lista[0])
ga.executar()
