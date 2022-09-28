import pandas as pd
import seaborn as sns

from sklearn.svm import LinearSVC 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

import numpy as np


#Fazendo a leitura dos dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)

#mudando o nome das colunas
nomes = {
    'unfinished':'nao_finalizado',
    'expected_hours':'horas_esperadas',
    'price':'preco'    
    }
dados = dados.rename(columns = nomes)

#Realizar mapeamento na coluna 'nao_finalizado' para trocar por 'finalizado'
troca = {
        0 : 1,
        1 : 0
    }
dados['finalizado'] = dados.nao_finalizado.map(troca)

#Eliminando a coluna 'nao_finalizado'
dados.pop('nao_finalizado')

#Realizando a plotagem dos dados das duas colunas de classificação
sns.scatterplot(x = "horas_esperadas", y = "preco", data = dados)

#Realizando a mesma plotagem dos dados separando por classe (finalizado ou não)
#usando o parâmetro >> hue <<
sns.scatterplot(x = "horas_esperadas", 
                y = "preco",
                hue = "finalizado",
                data = dados)

#Realizando plots relaativos em figuras diferentes usando
# o método HELPLOT com o parametro >> col <<
sns.relplot(x = "horas_esperadas", 
                y = "preco",
                hue = "finalizado",
                col = "finalizado",
                data = dados)

#_______________Escrevendo o modelo de aprendizado
x = dados[['horas_esperadas','preco']]
y = dados['finalizado']

SEED = 20

#Separando os dados de treino e teste
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, 
                                    random_state = SEED,
                                    test_size = 0.25,
                                    stratify = y)

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100 
print('A acurácia foi %.2f%%' % acuracia)


#Definindo as previsões de base
previsoes_base = np.ones(540)
acuracia_base = accuracy_score(teste_y, previsoes_base) * 100
print('A acurácia de base foi %.2f%%' % acuracia_base)


####______ MELHORANDO OS NOSSO ERROS _______________________


