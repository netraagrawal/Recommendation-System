import numpy as np
import math
from numpy import loadtxt
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
average_rating=0
pearson_k_neighbor_final=[]

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


def calculate_inverse_user_frequency(mid):

    mj=0
    for users in range(0,200):
        if(train_matrix_data[users][mid-1]!=0):
            mj+=1
    if(mj!=0):
        iuf_val=math.log10(200/mj)
        return iuf_val
    else:
        return 1


def final_k_neighbor_pearson(neighbor):
    c=0
    sum_of_similarity=0
    neighbor.sort(key=lambda x:x.similarity, reverse=True)
    #print(len(neighbor))
    for i in range(0,len(neighbor)):
        if i < k:
            c+=1
            #print(neighbor[i].similarity)
            pearson_k_neighbor_final.append(neighbor[i])
            sum_of_similarity += pearson_k_neighbor_final[i].similarity
            #print(pearson_k_neighbor_final[i].similarity)
    #print("len",len(pearson_k_neighbor_final))
    #print(c)
    return sum_of_similarity

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

def weighted_similarity(userid,mid):

    test_matrix=open_test()
    user_detail= get_user_data(userid,test_matrix)
    mid=user_detail.get_movies()
    rating=user_detail.get_ratings()
    average_rating=user_detail.get_average()
    pearson_k_neighbor=[]
    avg=calculate_average()
    for i in range(0, 200):
        # kobjid=final_k_neighbor[i].get_user_id()
        weight_n=0.0
        weight_d=0.0
        weight_dd=0.0
        for j in range(0,len(user_detail.get_movies())):
            #print(mid[j])
            if train_matrix_data[i][mid[j]-1] > 0:
                #print(mid[j])
                n_avg = avg[i+1]
                weight_n += (rating[j]-average_rating) * (train_matrix_data[i][mid[j]-1] - n_avg)
                weight_d += math.pow(rating[j]- average_rating,2)
                weight_dd += math.pow(train_matrix_data[i][mid[j]-1] - n_avg,2)
        #print(weight_d * weight_dd)
        if (weight_d * weight_dd) > 0:
            weight = weight_n / ((math.sqrt(weight_d) * math.sqrt(weight_dd)))

            # iuf
            iuf_res=calculate_inverse_user_frequency(mid)
            weight*=iuf_res

            # case modification
            weight*=math.pow(math.fabs(weight),2.5-1)

            #myownalgorithm
            weight*=

            weight_obj=neighbor(i+1,weight)
            pearson_k_neighbor.append(weight_obj)
    sum_of_weight = final_k_neighbor_pearson(pearson_k_neighbor)
    #print(sum_of_weight)
    return sum_of_weight


def pearson_correlation(userid,mid):

    sum_weight = weighted_similarity(userid, mid)
    #print(sum_weight)
    total=0.0
    mod_sum=0.0
    avg=calculate_average()
    test_matrix=open_test()
    user=get_user_data(userid,test_matrix)
    for i in range(0,len(pearson_k_neighbor_final)):
        kid= pearson_k_neighbor_final[i].user_id
        similarity= pearson_k_neighbor_final[i].similarity
        if train_matrix_data[kid-1][mid-1] > 0:

            total+= (train_matrix_data[kid-1][mid-1] - avg[kid]) * similarity
            mod_sum += math.fabs(similarity)
    if mod_sum!=0:
        total=user.get_average() + (total/mod_sum)
    else:
        total=user.get_average()

    if total>5:
        total=5
    elif total<=0:
        total=1
    return int(round(total))

#total=pearson_correlation(201,1)
#print(total)

def calculate_final_prediction(test_matrix, mp,output):

    uid= mp.get_users()
    mid=mp.get_movies()
    print("len",mid[0])
    for val in range(0,len(uid)):
        user=uid[val]
        movie=mid[val]
        #cosine_similarity(user)
        #predict=calculate_prediction(user,movie)
        ans = pearson_correlation(user, movie)
        predict = str(user)+" "+str(movie)+" "+str(ans)
        output.write(predict+"\n")
        print(ans)



def calling_method():

    output= open("PearsonOutputtest5.txt",'w')
    test_matrix=open_test()
    k=100
    movies_to_predit = user_movie_prediction(test_matrix)
    #val=calculate_prediction(201,1,test_matrix)
    calculate_final_prediction(test_matrix, movies_to_predit,output)

calling_method()