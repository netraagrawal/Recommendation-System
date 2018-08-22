import numpy as np
import math
from numpy import loadtxt

def save_data_from_train():

    train_matrix = loadtxt("train.txt", comments="#", delimiter="\t", unpack=False)
    return train_matrix

train_matrix_data = save_data_from_train()
k=100
class users_detail:
    def __init__(self, users, movies, ratings, average):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.average= average

    def get_users(self):
        return self.users

    def get_movies(self):
        return self.movies

    def get_ratings(self):
        return self.ratings

    def get_average(self):
        return self.average


class neighbor:
    def __init__(self, user_id, similarity):
        self.user_id = user_id
        self.similarity = similarity

    def get_similarity(self):
        return self.similarity

    def get_user_id(self):
        return self.user_id

list_of_similarity=[]
final_k_neighbor=[]

def open_test():
    test_matrix = np.loadtxt("test5.txt", comments="#", delimiter=' ', dtype='int')
    # print(test_matrix)
    return test_matrix

open_test()
def get_user_data(user_id, test_matrix):

    movies=[]
    ratings=[]
    users=[]
    average_rating=0
    for i in range(0, len(test_matrix)):
        uid=test_matrix[i][0]
        urating = test_matrix[i][2]
        movieid = test_matrix[i][1]
        if (user_id == uid) & (urating != 0):
            movies.append(movieid)
            ratings.append(urating)
            users.append(uid)
            average_rating+=urating
    average_rating= average_rating/len(ratings)
    #print(average_rating)
    user= users_detail(users, movies, ratings,average_rating)
    return user
k=100
def user_movie_prediction(test_matrix):

    movies = []
    ratings = []
    users = []

    for i in range(0,len(test_matrix)):
        uid=test_matrix[i][0]
        movieid=test_matrix[i][1]
        urating=test_matrix[i][2]
        if urating==0:
            users.append(uid)
            ratings.append(urating)
            movies.append(movieid)
    movies_for_prediction=users_detail(users,movies,ratings,0.0)
    return movies_for_prediction

def calculate_average():

    avg_train_users={}
    for neighbor in range(0, len(train_matrix_data)):
        n_avg=0;
        count=0;
        for i in range(0,len(train_matrix_data[neighbor])):
            if(train_matrix_data[neighbor][i]) != 0:
                n_avg+=train_matrix_data[neighbor][i]*1.0
                count+=1
        n_avg=n_avg/count
        avg_train_users[neighbor+1]=n_avg
    #print(n_avg)
    return avg_train_users

kmost=[]
def calculate_kmost(list):

    c=0
    for i in range(0, len(list)):
        if i < k:
            kmost.append(list[i])
            c+=1

    return kmost


#variation of cosine_similarity
own_neighbour=[]
def calculate_similarity(userid):

    user=get_user_data(userid);
    user_avg=user.get_average()
    train_avg=calculate_average()
    for user in range(0, 200):
        n_val = 0.0
        d_val = 0.0
        dd_val = 0.0
        total=0.0
        u_avg= train_avg[user+1]
        for movie in range(0, len(user.get_movies)):
            m_id = user.get_movies()[movie]
            if (user.get_ratings()[movie] != 0) & (train_matrix_data[user][m_id-1] != 0):
                n_val += (user_avg-user.get_ratings[movie]) * (train_avg-train_matrix_data[user][m_id-1])

                d_val += math.pow(user_avg- user.get_ratings()[movie], 2)
                dd_val += math.pow(train_avg- train_matrix_data[user][m_id-1], 2)

        total = math.sqrt(d_val) * math.sqrt(dd_val)

        if total != 0.0:
            similarity = n_val/total
            obj = neighbor(user+1, similarity)
            own_neighbour.append(obj)

    # sort need top K nearest neighbor, para is K_neighbor
    own_neighbour.sort(key=lambda x: x.similarity, reverse=True)
    calculate_kmost(own_neighbour)

def cal_own_prediction(userid,mid):

    calculate_similarity(userid)

    for i in range(0, len(list)):
        neighbor = list[i]
        training_user_id = neighbor.get_user_id()

        if training_matrix[training_user_id-1][movie_id-1] > 0:
            rating = training_matrix[training_user_id-1][movie_id-1]

            numerator += neighbor.get_similarity() * rating
            denominator += neighbor.get_similarity()

    # if denominator is zero, then the prediction rating is average
    # of total movies in training matrix
    # else result is equal to average in test file
    if denominator != 0:
        result = numerator/denominator
    elif the_avg_movie_rating != 0:
        result = the_avg_movie_rating
    else:
        result = avg_user_rating

    result = int(round(result))
    return result





def calling_method():
    output= open("itemOutputtest5.txt",'w')
    test_matrix=open_test()
    k=100
    movies_to_predit = user_movie_prediction(test_matrix)
    #val=calculate_prediction(201,1,test_matrix)
    uid= movies_to_predit.get_users()
    mid=movies_to_predit.get_movies()
    #print("len",mid[0])
    for val in range(0,len(uid)):
        user=uid[val]
        movie=mid[val]
        #cosine_similarity(user)
        calculate_own_prediction(user,movie)


calling_method()
