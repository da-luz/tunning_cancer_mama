# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 13:17:30 2022

@author: da-luz
"""

#Para criação das redes neurais
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, ELU
from scikeras.wrappers import KerasClassifier

#Para tratamento dos dados
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, minmax_scale

import seaborn as sns
import matplotlib.pyplot as plt

#Para validação e avaliação dos modelos
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

previsores = pd.read_csv('atributos_cm.csv')
classes = pd.read_csv('classes_cm.csv')

desc = previsores.describe().T

plt.figure(figsize=(15, 12)).patch.set_facecolor('white')
local = 1
for coluna in previsores.columns:
    plt.subplot(6, 5, local).set_title(coluna)
    plt.hist(x=previsores[coluna], color = 'green')
    local += 1
plt.tight_layout()

colunas = desc.loc[desc['std'] < desc['mean']].index.to_list()

normal = pd.DataFrame(minmax_scale(previsores.values, axis = 0) , columns=[previsores.columns])
escala = pd.DataFrame(scale(previsores.values, axis = 0), columns=[previsores.columns])
opcoes = [previsores, normal, escala]

corte_previsores = previsores[colunas]
corte_normal = normal[colunas]
corte_escalado = escala[colunas]
cortes = [corte_previsores, corte_normal, corte_escalado]

parametros_testados = {
    'otimizador' : ['adam', 'SGD', 'RMSprop'],
    'ativacao' : ['relu', 'elu'],
    'neuronios' : [8, 16, 32],
    'camadas': [1, 2, 3],
    'dropout': [0, 0.2, 0.25]
}

def rede_teste (otimizador, ativacao, neuronios, camadas, dropout, dimensao, **kwargs):
    '''
    Função que retorna o modelo para teste do dicionário de parametros especificado.
    '''
    modelo = Sequential()
    modelo.add(Dense(units = neuronios, activation = ativacao, kernel_initializer = 'random_uniform', input_dim = dimensao))
    modelo.add(Dropout(dropout))
    for camadas in range(camadas):
        modelo.add(Dense(units = neuronios, activation = ativacao, kernel_initializer = 'random_uniform'))
        modelo.add(Dropout(dropout))
    modelo.add(Dense(units = 1, activation = 'sigmoid'))
    modelo.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return modelo

s_filtro = len(previsores.columns)
filtrado = len(corte_previsores.columns)

modelo = KerasClassifier(model = rede_teste, epochs = 100, batch_size = 10, verbose = False, dimensao = s_filtro, ativacao = None, otimizador = None, neuronios = None, camadas = None, dropout = None)
testes = GridSearchCV(estimator = modelo, param_grid = parametros_testados, scoring = 'accuracy', cv = 4, verbose = 1)

modelo_cortes = KerasClassifier(model = rede_teste, epochs = 100, batch_size = 10, verbose = False, dimensao = filtrado, ativacao = None, otimizador = None, neuronios = None, camadas = None, dropout = None)
testes_cortes = GridSearchCV(estimator = modelo_cortes, param_grid = parametros_testados, scoring = 'accuracy', cv = 4, verbose = 1)

# contador = 0
# resultados = []
# for item in range(len(opcoes)):
#     print(f'Começando teste {contador}')
#     testes.fit(opcoes[item], classes)
#     resultados.append([contador, testes.best_params_, testes.best_score_])
#     contador += 1

# contador = 0
# resultados_cortes = []
# for item in range(len(cortes)):
#     print(f'Começando teste {contador}')
#     testes_cortes.fit(cortes[item], classes)
#     resultados_cortes.append([contador, testes.best_params_, testes.best_score_])
#     contador += 1

n_preprocessado = {
    'ativacao': 'elu',
    'camadas': 3,
    'dropout': 0.2,
    'neuronios': 32
}
normalizado = {
    'ativacao': 'relu',
    'camadas': 2,
    'dropout': 0,
    'neuronios': 32
}
escalado = {
    'ativacao': 'elu',
    'camadas': 1,
    'dropout': 0.2,
    'neuronios': 16
}

parametros_corte = {
    'ativacao': 'elu',
    'camadas': 1,
    'dropout': 0.2,
    'neuronios': 16
}

def rede (dimensao, ativacao = 'elu', camadas = 1, dropout = 0.2, neuronios = 16, **kwargs):
    '''
    Função que retorna o modelo conforme os argumentos ou o dicionário de kwargs especificado.
        Os kwargs são:
            dimensao : int
                diz respeito ao número de variáveis previsoras na base de dados
            ativacao : \'elu\' or \'relu\', default : \'elu\'
                função de ativação das camadas ocultas
            camadas : int, default : 1
                quantas camadas ocultas, além da camada de ativação, o modelo terá
                    obs: o número total de camadas ocultas será "camadas" + 1
            dropout : float, default : 0.2
                cria camadas de dropout com a porcentagem de neuronios a serem zerados
            neuronio : int, default : 16
                determina quantas unidades haverão em cada camada
    '''
    modelo = Sequential()
    modelo.add(Dense(units = neuronios, activation = ativacao, kernel_initializer = 'random_uniform', input_dim = dimensao))
    modelo.add(Dropout(dropout))
    for camadas in range(camadas):
        modelo.add(Dense(units = neuronios, activation = ativacao, kernel_initializer = 'random_uniform'))
        modelo.add(Dropout(dropout))
    modelo.add(Dense(units = 1, activation = 'sigmoid'))
    modelo.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return modelo

modelo_n_preprocessado = KerasClassifier(model = rede(dimensao = s_filtro, **n_preprocessado), epochs = 100, batch_size = 10, verbose = False)

modelo_normalizado = KerasClassifier(model = rede(dimensao = s_filtro, **normalizado), epochs = 100, batch_size = 10, verbose = False)

modelo_escalado = KerasClassifier(model = rede(dimensao = s_filtro, **escalado), epochs = 100, batch_size = 10, verbose = False)

modelo_filtrado = KerasClassifier(model = rede(dimensao = filtrado, **parametros_corte), epochs = 100, batch_size = 10, verbose = False)

modelo_fn = KerasClassifier(model = rede(dimensao = filtrado, **parametros_corte), epochs = 100, batch_size = 10, verbose = False)

modelo_fe = KerasClassifier(model = rede(dimensao = filtrado, **parametros_corte), epochs = 100, batch_size = 10, verbose = False)

cv_n_preprocessado = cross_val_score(estimator = modelo_n_preprocessado, X = previsores, y = classes, cv = 10, scoring = 'accuracy')

cv_normalizado = cross_val_score(estimator = modelo_normalizado, X = normal, y = classes, cv = 10, scoring = 'accuracy')

cv_escalado = cross_val_score(estimator = modelo_escalado, X = escala, y = classes, cv = 10, scoring = 'accuracy')

cv_n_preprocessado.mean(), cv_normalizado.mean(), cv_escalado.mean()

cv_cnp = cross_val_score(estimator = modelo_filtrado, X = corte_previsores, y = classes, cv = 10, scoring = 'accuracy')

cv_cn = cross_val_score(estimator = modelo_fn, X = corte_normal, y = classes, cv = 10, scoring = 'accuracy')

cv_ce = cross_val_score(estimator = modelo_fe, X = corte_escalado, y = classes, cv = 10, scoring = 'accuracy')

cv_cnp.mean(), cv_cn.mean(), cv_ce.mean()

p_treino, p_teste, c_treino, c_teste = train_test_split(previsores, classes, test_size = 0.25, random_state = 0)
n_treino, n_teste, _, _ = train_test_split(normal, classes, test_size = 0.25, random_state = 0)
e_treino, e_teste, _, _ = train_test_split(escala, classes, test_size = 0.25, random_state = 0)
cp_treino, cp_teste, _, _ = train_test_split(corte_previsores, classes, test_size = 0.25, random_state = 0)
cn_treino, cn_teste, _, _ = train_test_split(corte_normal, classes, test_size = 0.25, random_state = 0)
ce_treino, ce_teste, _, _ = train_test_split(corte_escalado, classes, test_size = 0.25, random_state = 0)

modelo_n_preprocessado.fit(p_treino, c_treino)
modelo_normalizado.fit(n_treino, c_treino)
modelo_escalado.fit(e_treino, c_treino)

modelo_filtrado.fit(cp_treino, c_treino)
modelo_fn.fit(cn_treino, c_treino)
modelo_fe.fit(ce_treino, c_treino)

p_previsores = modelo_n_preprocessado.predict(p_teste)
p_normal = modelo_normalizado.predict(n_teste)
p_escala = modelo_escalado.predict(e_teste)
previsoes_integras = [p_previsores, p_normal, p_escala]

p_cp = modelo_filtrado.predict(cp_teste)
p_cn = modelo_fn.predict(cn_teste)
p_ce = modelo_fe.predict(ce_teste)
previsores_filtradas = [p_cp, p_cn, p_ce]

for item in previsoes_integras:
    print(accuracy_score(item, c_teste))
    
for item in previsores_filtradas:
    print(accuracy_score(item, c_teste))

