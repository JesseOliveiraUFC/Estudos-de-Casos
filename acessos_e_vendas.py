import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

dados = pd.read_csv(uri)

print(dados.head())

#Renomeando as colunas de dados
mapa = {
        "home":"principal",
        "how_it_works":"como_funciona",
        "contact":"contato",
        "bought":"comprou"
        }
dados = dados.rename(columns = mapa)

#Fazendo a separação dos dados de aprendizado das classes
x = dados[['principal','como_funciona','contato']]
y = dados['comprou']

#Realizando a separação dos dados entre treino e teste
#Treinando com 75 dados e testando com 24 dados || Método direto
treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]

#Realizando a separação dos dados com a função tratest_split
#Criando uma SEED para manter o código repetitivo e
#Estratificando em função de y para manter as devidas proporções de treino e teste
SEED = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, 
                                                        test_size = 0.25, 
                                                        random_state=SEED, 
                                                        stratify=y)

#Realizando a criação do modelo de predição LinearSVC
modelo = LinearSVC()

#Realizando o treino do algoritmo
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A acurácia foi %.2f %%" % acuracia)