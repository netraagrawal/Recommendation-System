import numpy as np
import math
from numpy import loadtxt


# this is to store training data set
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
average_rating=0
pearson_k_neighbor_final=[]
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


def open_test():
    test_matrix_data = np.loadtxt("test5.txt", delimiter=' ', dtype='int')
    return test_matrix_data

def calculate_similarity(user,specific_user):

    product_ratings=0.0
    denominator_test=0.0
    denominator_train=0.0
    c=0
    for m in range(0,len(specific_user.movies)):
       # print(specific_user.movies[m])
        if (train_matrix_data[user][specific_user.movies[m]-1] > 0) & (specific_user.ratings[m]>0):
            product_ratings = product_ratings + (train_matrix_data[user][specific_user.movies[m]-1] * specific_user.ratings[m])
            denominator_test = denominator_test + math.pow(specific_user.ratings[m],2)
            denominator_train =denominator_train + math.pow(train_matrix_data[user][specific_user.movies[m]-1],2)
    #print("product", product_ratings)
    #print("denomination", math.sqrt(denominator_train) * math.sqrt(denominator_test))
    if (math.sqrt(denominator_train) * math.sqrt(denominator_test)) !=0:
        total_similarity = product_ratings / (math.sqrt(denominator_train) * math.sqrt(denominator_test))
        # case modification
        #total_similarity*=math.pow(math.fabs(total_similarity),2.5-1)
        neighbor_similarity = neighbor(user+1, total_similarity)
        list_of_similarity.append(neighbor_similarity)


def calculate_final_k_neighbor(user_id,list):
        c=0
        sum_of_similarity=0
        for i in range(0,len(list)):
            if i < k:
                c+=1
                final_k_neighbor.append(list[i])
                #print(final_k_neighbor[i].similarity)
                sum_of_similarity += final_k_neighbor[i].similarity
                #print(final_k_neighbor[i].similarity)
        #print(c)
        return sum_of_similarity

def cosine_similarity(user_id):

    test_matrix = open_test()
    specific_user = get_user_data(user_id, test_matrix)
    #print(specific_user.ratings,specific_user.movies)
    for user in range(0,len(train_matrix_data)):
        calculate_similarity(user,specific_user)
    list_of_similarity.sort(key=lambda x: x.similarity, reverse=True)
    sum=calculate_final_k_neighbor(user_id,list_of_similarity)


def calculate_prediction(user_id,mid,test_matrix):

    total_product=0
    cosine_similarity(user_id)
    sum = calculate_final_k_neighbor(user_id, list_of_similarity)
    #print(sum)
    user=get_user_data(user_id,test_matrix)
    for i in range(0,len(final_k_neighbor)):
        kid=final_k_neighbor[i].get_user_id()
        if(train_matrix_data[kid-1][mid-1])>0:
            total_product+=train_matrix_data[kid-1][mid-1]* final_k_neighbor[i].similarity
            #total_product+= (final_k_neighbor[i].similarity / sum) * train_matrix_data[kid-1][mid-1]
            #print(total_product)
    if sum !=0:
        prediction=total_product/sum
        prediction=int(round(prediction))
        if prediction > 5:
           prediction = 5
        elif prediction < 1:
           prediction= 1
    else:
        prediction= user.get_average()
    #print(prediction)
    return int(round(prediction))



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



def calculate_final_prediction(test_matrix, mp, output):

    uid= mp.get_users()
    mid=mp.get_movies()
    #print("len",mid[0])
    for val in range(0,len(uid)):
        user=uid[val]
        movie=mid[val]
        #cosine_similarity(user)
        predict=calculate_prediction(user,movie,test_matrix)
    ans = str(user)+" "+str(movie)+" "+str(predict)
    print(ans)
    output.write(ans+'\n')


def calling_method():

    output= open("CosineOutputtest5.txt",'w')
    test_matrix=open_test()
    k=100
    movies_to_predit = user_movie_prediction(test_matrix)
    #val=calculate_prediction(201,1,test_matrix)
    calculate_final_prediction(test_matrix, movies_to_predit,output)

calling_method()





#def calculate_adjusted_cosine_similarity(userid,movieid):

