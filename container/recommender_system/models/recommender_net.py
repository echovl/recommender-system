import logging
import sys
import os
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

prefix = '/opt/program/'
model_path = os.path.join(prefix, 'models/data')

products_filename = "products.csv"
ratings_filename = "ratings.csv"

def fuzzy_matching(mapper, fav_movie, verbose=True):
    """
    return the closest match via fuzzy ratio. If no match found, return None
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie
    
    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]



def make_recommendation(model_knn, data, mapper, product, n_recommendations):
    """
    return top n similar movie recommendations based on user's input movie

    Parameters
    ----------
    model_knn: sklearn model, knn model

    data: movie-user matrix

    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar movie recommendations
    """
    # get input movie index
    print('You have input movie:', product)
    idx = fuzzy_matching(mapper, product, verbose=True)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1], reverse= True)[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(product))
    candidates = []
    for i, (idx, dist) in enumerate(raw_recommends):
        candidates.append({ "name": reverse_mapper[idx], "dist": dist})
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))
    
    # Remove the product itself from the recommendation
    candidates.pop(0)

    return candidates

class RecommenderNet:
    model = None
    data = None
    mapper = None

    @classmethod
    def get_model(cls):
        if cls.model == None:

            print("Building RecommenderNet...")

            df_products = pd.read_csv(
                os.path.join(model_path, products_filename),
                usecols = ["productId", "name"],
                dtype = {"productId": "int32", "name": "str"}
            )

            df_ratings = pd.read_csv(
                os.path.join(model_path, ratings_filename),
                usecols = ["userId", "productId", "rating"],
                dtype = {"userId": "int32", "productId": "int32", "rating": "int32"}
            )

            # pivot and create movie-user matrix
            product_user_mat = df_ratings.pivot(index='productId', columns='userId', values='rating').fillna(0)
            
            # create mapper from movie title to index
            product_to_idx = {
                product: i for i, product in 
                enumerate(list(df_products.set_index('productId').loc[product_user_mat.index].name))
            }
            
            # transform matrix to scipy sparse matrix
            product_mat_sparse = csr_matrix(product_user_mat.values)

            # define model
            model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
            
            # fit
            model_knn.fit(product_mat_sparse)

            cls.model = model_knn
            cls.mapper = product_to_idx   
            cls.data = product_mat_sparse    

        return cls.model

    @classmethod
    def predict(cls, input):
        clf = cls.get_model()

        return make_recommendation(
            model_knn = clf,
            mapper = cls.mapper,
            data = cls.data,
            product = input,
            n_recommendations = 4
        )