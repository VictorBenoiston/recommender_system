import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import implicit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ratings = pd.read_csv("")  # Path dos ratings
aulas = pd.read_csv("")  # Path das aulas

a = ratings.head()
# print(a)  # Testando a leitura de dados


def create_X(df):
    """
    Args:
        df: pandas dataframe
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        aula_mapper: dict that maps class id's to movie indices
        aula_inv_mapper: dict that maps classes indices to movie id's
    """
    N = df['userId'].nunique()
    M = df['aulasId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    aula_mapper = dict(zip(np.unique(df["aulasId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    aula_inv_mapper = dict(zip(list(range(M)), np.unique(df["aulasId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    aula_index = [aula_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (aula_index, user_index)), shape=(M, N))

    return X, user_mapper, aula_mapper, user_inv_mapper, aula_inv_mapper


X = create_X(ratings)[0]
user_mapper = create_X(ratings)[1]
aula_mapper = create_X(ratings)[2]
user_inv_mapper = create_X(ratings)[3]
aula_inv_mapper = create_X(ratings)[4]


from fuzzywuzzy import process


def aula_finder(title):
    all_titles = aulas['title'].tolist()  # Os nomes das aulas
    closets_match = process.extractOne(title, all_titles)
    return closets_match[0]


aula_title_mapper = dict(zip(aulas['title'], aulas['aulasId']))
aulas_title_inv_mapper = dict(zip(aulas['aulasId'], aulas['title']))


def get_aula_index(title):
    """
    :return: The idx of the class, based on the title.
    """
    fuzzy_title = aula_finder(title)
    aula_id = aula_title_mapper[fuzzy_title]
    aula_idx = aula_mapper[aula_id]
    return aula_idx


def get_aula_title(aula_idx):
    """
    :return: The title of the class based on the idx.
    """
    movie_id = aula_inv_mapper[aula_idx]
    title = aulas_title_inv_mapper[movie_id]
    return title


b = get_aula_index('AulaDeDízimasPeriódicasFraçãoGeratriz')
# print(b)  # Testando

c = get_aula_title(31)
# print(c)  # Testando

model = implicit.als.AlternatingLeastSquares(factors=50)
model.fit(X)

aula_of_intereset = 'AulaDePropriedadeDaIguadade'

aula_index = get_aula_index(aula_of_intereset)
related = model.similar_items(aula_index)
# print(related)  # Testando

print(f"Porque você assistiu a aula: {aula_finder(aula_of_intereset)}...")
# for r in related:
#     recommended_title = get_aula_title(r[0])
#     if recommended_title != aula_finder(aula_of_intereset):
#         print(recommended_title)


user_id = 95

user_ratings = ratings[ratings['userId']==user_id].merge(aulas[['movieId', 'title']])
user_ratings = user_ratings.sort_values('rating', ascending=False)
print(f"Número de aulas avaliadas por usuário: {user_id}: {user_ratings['movieId'].nunique()}")

user_ratings = ratings[ratings['userId']==user_id].merge(aulas[['aulasId', 'title']])
user_ratings = user_ratings.sort_values('rating', ascending=False)
top_5 = user_ratings.head()  # Os melhores avaliados
print(top_5)

bottom_5 = user_ratings[user_ratings['rating']<3].tail()  # Os piores avaliados
print(bottom_5)

X_t = X.T.tocsr()

user_idx = user_mapper[user_id]
recommendations = model.recommend(user_idx, X_t)
print(recommendations)

for r in recommendations:
    recommended_title = get_aula_title(r[0])
    print(recommended_title)

