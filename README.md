# Estudos-de-Casos
Estudo de casos diversos durante o processo de aprendizado de ferramentas de Machine Learning

________________________________________________________

# 1 - Acessos e Vendas

### O Código está no arquivo `acessos_e_vendas.py`

Este é um estudo de caso utilizando o método LinearSVC do SkLearn.
Os dados são separados em dados de treino e teste de duas maneiras:

* Separação manual
* Utilizando o método train_test_split do sklearn

## Resultados

O algoritmo consegue prever com uma média de 96% se um usuário vai comprar um produto baseado nas páginas acessadas por ele no site.

A boa precisão do aprendizado se dá, na maior parte, devido à simplicidade dos dados de treinamento que só possuem 3 colunas.
Para modelos mais elaborados, com muitos dados de entrada, será necessário utilizar novos modelos de aprendizagem.

> OBS.: Estarei estudando a parte teórica do método LinearSVC e as atualizações estarei postando aqui.
__________________________________________________________

# 2 - Projetos de Sites

### O Código está no arquivo `projetos_de_sites.py`

Este é um estudo de caso de dados de projetos de sites e sua conclusão baseada no preço a ser pago pelas quantidades de horas de dedicação.

Inicialmente, realizamos o tratamento da base dos dados. Mudamos os nomes das colunas, realizamos mapeamento para mudar a coluna de não-finalizados para finalizados (para garantir que 0 é False e 1 é True).

Após o tratamento, realizamos o plot para visualização dos dados de duas formas:
* Usando o scatterplot:

![](plot1.png)

* Usando o relplot:

![](plot2.png)

Em seguida, escrevemos o modelo para realizar o treinamento e previsão. Usamos o método LinearSVC para fit dos dados.

Neste caso, obtivemos uma margem de acerto de ***55.37%***.

Definimos a baseline com o comando `baseline = np.ones()` e obtivemos uma precisão de ***52.59%***.

Isso significa que nossa previsão ainda está longe de ser boa.

> Em breve darei continuidade a métodos de melhoria desse algoritmo.



