import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Carregando os dados

ratings = pd.read_csv("")  # Inserir o path do arquivo csv contendo os dados com as notas das aulas.
a = ratings.head()
# print(a)

aulas = pd.read_csv("")  # Inserir o path do arquivo csv contendo os dados com as aulas.
b = aulas.head()
# print(b)  # Apenas printando os dados para ver se está tudo bem

# Análise de dados

n_ratings = len(ratings)
n_aulas = ratings['aulaId'].nunique()
n_users = ratings['userId'].nunique()

# print(f"Número de avaliações: {n_ratings}")
# print(f"Número de id de aulas únicos: {n_aulas}")
# print(f"Número de usuários únicos: {n_users}")
# print(f"Média de avaliações por usuário: {round(n_aulas/n_users, 2)}")
# print(f"Média de avaliações por aula: {round(n_ratings/n_aulas, 2)}")


# Dividindo os usuários e suas quantidades de ratings

user_freq = ratings[['userId', 'aulasId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
c = user_freq.head()
# print(c)  # Apenas printando os 5 primeiros, para checar

# Quantidade de aulas que cadaa usuário, em média, avaliou
# print(f"Média de aulas avaliadas por usuário: {user_freq['n_ratings'].mean():.2f}")


# Visualizando os dados

sns.set_style("whitegrid")
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
ax = sns.countplot(x="rating", data=ratings, palette="viridis")
plt.title("Distribution of movie ratings")

plt.subplot(1, 2, 2)
ax = sns.kdeplot(user_freq['n_ratings'], shade=True, legend=False)
plt.axvline(user_freq['n_ratings'].mean(), color="k", linestyle="--")
plt.xlabel("# ratings por usuário")
plt.ylabel("density")
plt.title("Número de aulas avaliadas por usuário")
# plt.show()


# Qual aula tem a maior nota??

mean_rating = ratings.groupby('aulaId')[['rating']].mean()

lowest_rated = mean_rating['rating'].idxmin()
# print(aulas.loc[aulas['aulasId'] == lowest_rated])

highest_rated = mean_rating['rating'].idxmax()
# print(aulas.loc[aulas['aulasId'] == highest_rated])


# Avaliando a quantidade de votos

# print(ratings[ratings['aulasId'] == highest_rated])


# Para anular a hipótese de uma aular ter a maior nota com poucas avaliações, utilizaremos
# a média Bayesiana


aula_stats = ratings.groupby('aulasId')[['rating']].agg(['count', 'mean'])
aula_stats.columns = aula_stats.columns.droplevel()

C = aula_stats['count'].mean()
m = aula_stats['mean'].mean()


def bayesian_avg(ratings):
    bayesian_avg = (C * m + ratings.sum()) / (C + ratings.count())
    return bayesian_avg


bayesian_avg_ratings = ratings.groupby('aulasId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['aulasId', 'bayesian_avg']
aula_stats = aula_stats.merge(bayesian_avg_ratings, on='aulasId')

# Utilizando a média bayesiana para ver o pior e o melhor avaliado

aula_stats = aula_stats.merge(aulas[['aulasId', 'title']])
d = aula_stats.sort_values('bayesian_avg', ascending=False).head()
# print(d)  # Apenas printando para verificar

e = aula_stats.sort_values('bayesian_avg', ascending=True).head()
# print(e)  # Apenas printando para verificar


# Transformando os dados

from scipy.sparse import csr_matrix


def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.

    Args:
        df: pandas dataframe
    Returns:
        X: sparse matrix
        user_mapper: dict that maps id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        aula_mapper: dict that maps classes id's to classes indices
        aula_inv_mapper: dict that maps class indices to classes id's
    """
    N = df['userId'].nunique()
    M = df['aulasId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    aula_mapper = dict(zip(np.unique(df["aulasId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    aula_inv_mapper = dict(zip(list(range(M)), np.unique(df["aulasId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    aula_index = [aula_mapper[i] for i in df['aulasId']]

    X = csr_matrix((df["rating"], (aula_index, user_index)), shape=(M, N))

    return X, user_mapper, aula_mapper, user_inv_mapper, aula_inv_mapper


X, user_mapper, aula_mapper, user_inv_mapper, aula_inv_mapper = create_X(ratings)

sparsity = X.count_nonzero() / (X.shape[0] * X.shape[1])
# print(f"Matrix sparsity: {round(sparsity*100, 2)}%")  # Printando a matriz


# Salvando a matriz em um arquivo

from scipy.sparse import save_npz
# save_npz('Users\Dalvani\Desktop\TCC', X)  # Escolhe o diretório onde deseja salvar


# Utilizando k-Nearest Neighbours para achar aulas parecidos

from sklearn.neighbors import NearestNeighbors


def find_similar_aulas(aula_id, X, k, metric='cosine', show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    Returns:
        list of k similar movie ID's
    """
    neighbours_ids = []

    aula_ind = aula_mapper[aula_id]
    aula_vec = X[aula_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    if isinstance(aula_vec, (np.ndarray)):
        movie_vec = aula_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(aula_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbours_ids.append(aula_inv_mapper[n])
    neighbours_ids.pop(0)
    return neighbours_ids


aulas_titles = dict(zip(aulas['movieId'], aulas['title']))

aula_id = 2

similar_ids = find_similar_aulas(aula_id, X, k=10)
aula_title = aulas_titles[aula_id]

print(f"Porque você assistiu: {aula_title}")
print('Você deveria assistir também: ')
print('-' * 30)
for i in similar_ids:
    print(aulas_titles[i])
