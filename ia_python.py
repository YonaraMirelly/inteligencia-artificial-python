# https://www.kaggle.com/ -> site para encontrar base de dados
# Utilize o arquivo .ipynb -> estou usando o .py por conta do github, mas use o formato .ipynb!
# Instalar as bibliotecas:
# pip install pandas
# pip install pandas numpy scikit-learn
# PASSOS:
# 1 - Entenda o desafio da empresa!
# 2 - Importar a base de dados
import pandas as pd
tabela = pd.read_csv("clientes.csv")
print(tabela.info())

# 3 - Fazer o tratamento na base de dados -> transformar tudo que for string em int!
# Para toda coluna na tabela, se o dtype for "object", ou seja, string, ele vai tranformar em número.
# exemplo: mecanico, medico, advogado vai virar 0,1,2.
# E tem que ser diferente do score, pois é ele que queremos que a IA preveja
from sklearn.preprocessing import LabelEncoder
codificador = LabelEncoder()
for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score-crédito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])

# lembre-se: no .ipynb, utilize o display(tabela.info())
# verificando se realmente todas as colunas foram modificadas
print(tabela.info())

# 4 - Escolher quais colunas você vai usar para o treino da IA
# y é a coluna que queremos que a IA calcule
# x são as colunas que vamos usar para treinar a IA, tirando a coluna 
# id_cliente porque ela é um numero qualquer que nao ajuda a previsao
y = tabela["score_credito"]
x = tabela.drop(columns = ["score_credito", "id_cliente"])

# 5 - Treinar a IA com 2 modelos
from sklearn.model_selection import train_test_split
# os dados foram separados em treino e teste!
# treino é para o modelo aprender e teste é pra saber se o modelo aprendeu corretamente
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y)
# importando os 2 modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# treinando os modelos
modelo_arvore = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()
modelo_arvore.fit(x_treino,y_treino)
modelo_knn.fit(x_treino,y_treino)

# 6 - Verificar o melhor modelo
from sklearn.metrics import accuracy_score
# calculando as previsões
previsao_arvore = modelo_arvore.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)
# comparando as previsões com o y_teste
# queremos o de maior acurácia!
print(accuracy_score(y_teste, previsao_arvore))
print(accuracy_score(y_teste, previsao_knn))
# Demora um pouquinho mesmo ;) 
#0.82832 -> previsao_arvore
#0.74752 -> previsao_knn
# Essa foi a acurácia! ou seja, o modelo da árvore é melhor para essa situação


# E............... se teu patrão chega p tu e te manda novos cliente?
# é a mesma lógica amore!
# importar novos cliente
novos_clientes = pd.read_csv("novos_clientes.csv")
print(novos_clientes)
for coluna in novos_clientes.columns:
    if novos_clientes[coluna].dtype == "object" and coluna != "score_credito":
        novos_clientes[coluna] = codificador.fit_transform(novos_clientes[coluna])

previsoes = modelo_arvore.predict(novos_clientes)
print(previsoes)
#0.82848
#0.74396

# quais as caracteristicas mais importantes para definir o score de credito?

#colunas = list(x_teste.columns)
#importancia = pd.DataFrame(index=colunas, data=modelo_arvore.feature_importances_)
#importancia = importancia * 100
#print(importancia)

# e é isso!! obrigada! :)