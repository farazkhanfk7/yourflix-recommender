from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

def fuzzy_matching(mapper, fav_movie, verbose=True):    
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

def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    # print('Recommendations for {}:'.format(fav_movie))
    movie_list = []
    for i, (idx, dist) in enumerate(raw_recommends):
        movie_list.append(reverse_mapper[idx])
    return movie_list

def get_html_content(movie):
    import requests
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
    LANGUAGE = "en-US,en;q=0.5"
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    session.headers['Accept-Language'] = LANGUAGE
    session.headers['Content-Language'] = LANGUAGE
    movie = movie.split(' (')
    movie = movie[0]
    movie = movie.replace(' ','+')
    url = f'https://www.rottentomatoes.com/search?search={movie}'
    html_content = requests.get(url).text
    return html_content

def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 

# Create your views here.
def home(request):
    if request.method == 'POST':
        movie_name = request.POST.get('movie_name')
        # Unpickle model
        model_knn = pd.read_pickle("model/recommender_model.pickle")
        movies = pd.read_csv("model/movies.csv")
        ratings = pd.read_csv('model/ratings.csv')
        df = pd.merge(movies,ratings)
        movie_user_mat = df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        movie_to_idx = {movie: i for i, movie in enumerate(list(movies.set_index('movieId').loc[movie_user_mat.index].title))}
        movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

        movie_list = make_recommendation(
            model_knn = model_knn,
            data=movie_user_mat_sparse,
            fav_movie=movie_name,
            mapper=movie_to_idx,
            n_recommendations=10)
        key_list = ["rec1","rec2","rec3","rec4","rec5","rec6","rec7","rec8","rec9","rec10"]
        result = dict(zip(key_list,movie_list))

        from bs4 import BeautifulSoup
        img_url_list = []
        for movie in movie_list:
            html_content = get_html_content(movie)
            soup = BeautifulSoup(html_content, 'html.parser')
            c = str(soup.find(id='movies-json'))
            c = c.split('"')
            img_url_list.append(c[9])
        
        key_url_list = ["img1","img2","img3","img4","img5","img6","img7","img8","img9","img10"]
        images = dict(zip(key_url_list,img_url_list))

        final_dict = Merge(result,images)
        print(final_dict)
        return render(request,'predict.html',final_dict)
    else:
        return render(request,'movie.html')
