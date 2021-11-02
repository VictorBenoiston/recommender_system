import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Carregando os dados:
data = pd.read_csv('consulta_geral.csv', header=None, names=["disciplina", "assuntos", "conteudo", "aula",
                                                             "atividadeEspecifica", "serie"])
data = data.drop_duplicates()
data = data.reset_index()

a = data.head()
# print(a)
# print(data)


# Dividindo em listas, para melhor uso
data['assuntos'] = data['assuntos'].apply(lambda x: x.split("|"))
b = data.head()
# print(b)


from collections import Counter

# Contando quantos asssutos existem:
assuntos_counts = Counter(a for assuntos in data['assuntos'] for a in assuntos)
# print(f"Existem {len(assuntos_counts)} assuntos diferentes")

# Printando os 5 mais comuns:
# print("Os 5 assuntos mais comuns foram: \n", assuntos_counts.most_common(5))

# Visualizando a popularidade dos assuntos graficamente

assuntos_counts_df = pd.DataFrame([assuntos_counts]).T.reset_index()
assuntos_counts_df.columns = ['assuntos', 'aparicoes']
assuntos_counts_df = assuntos_counts_df.sort_values(by='aparicoes', ascending=False)

plt.figure(figsize=(11, 5))
sns.barplot(x='assuntos', y='aparicoes', data=assuntos_counts_df, palette='viridis')
plt.xticks(rotation=90)
# plt.show()


import re

# Quantidade de séries únicas:

series = data['serie'].nunique()
# print(series)

# Quantidade de atividades específicas únicas:

atividades_esopecificas = data['atividadeEspecifica'].nunique()
# print(atividades_esopecificas)


# Transformando os dados:

assuntos = list(assuntos_counts.keys())


for a in assuntos:
    data[a] = data['assuntos'].transform(lambda x: int(a in x))


# Testando a primeira transformação (dos asssuntos em binário):

i = data[assuntos].head()
# print(i)

# Transformando as séries:

data_series = pd.get_dummies(data['serie'])

# Combinando os assuntos e séries em uma matriz:

data_features = pd.concat([data[assuntos], data_series], axis=1)
k = data_features.head()
# print(k)


# construindo o recomendador usando similaridade dos cossenos:
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(data_features, data_features)
# print(f"Dimensões das features da matriz de similaridade dos cossenos: {cosine_sim.shape}")


# Criando a função localizadora de aulas
from fuzzywuzzy import process


def localizador_de_aula(nome_da_aula):
    todas_as_aulas = data['aula'].tolist()
    achado_mais_proximo = process.extractOne(nome_da_aula, todas_as_aulas)
    return achado_mais_proximo[0]


# Testando a função:
nome_da_aula = 'AulaDeSemelhançaDeTriângulos'

id_aula = dict(zip(data['aula'], list(data.index)))
idx = id_aula[nome_da_aula]
# print(f'A aula {nome_da_aula}, se encontra no índice: {idx}')


# Pegando as 10 aulas mais próximas à 'AulaDeSemelhançaDeTriângulos'
n_recomendacao = 10
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:(n_recomendacao+1)]
aulas_similares = [i[0] for i in sim_scores]

print(f'Porque você assistiu {nome_da_aula}')
print(f"Você provavelmente irá gostar de: \n{data['aula'].iloc[aulas_similares]}")

print('-------------')


# Testando outras aulas:
def get_recomendacao_conteudo(string_aula, n_recomendacoses=10):
    aula = localizador_de_aula(string_aula)
    idx = id_aula[string_aula]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recomendacoses+1)]
    aulas_similares = [i[0] for i in sim_scores]
    print(f"Recomendações para: {string_aula}")
    print(data['aula'].iloc[aulas_similares])


get_recomendacao_conteudo('AulaDeVolumeDePrismasECilindros', 5)

