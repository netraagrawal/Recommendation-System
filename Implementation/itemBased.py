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


def open_test():
    test_matrix_data = np.loadtxt("test5.txt", delimiter=' ', dtype='int')
    return test_matrix_data


item_based_similarity=[]
itembased_final_k_neighbor=[]
k=100
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


item_similarity={}
def check(total_val,total_n,usr_no,mid):

    sim=0
    if total_val>0:
        sim=total_n/total_val
        #print("Sim ",sim)
        sim_obj=neighbor(usr_no+1,sim)
        item_based_similarity.append(sim_obj)
        item_similarity[usr_no+1]=mid


def sort_list(list):

    list.sort(key=lambda x: x.similarity, reverse=True)

def calculate_final_k(list):

    c=0
    sum_sim=0
    for i in range(0,len(list)):
        if i<k:
            itembased_final_k_neighbor.append(list[i])
            #print(itembased_final_k_neighbor[i].similarity)
            sum_sim+=itembased_final_k_neighbor[i].similarity
            c+=1
            #print(c)
    #print(sum_sim)
    return sum_sim

def calculate_item_based_similarity(userid,mid):

    test_matrix=open_test()
    details=get_user_data(userid,test_matrix)
    mlist=details.get_movies()
    avg=details.get_average()
    train_avg=calculate_average()
    for movies in range(len(details.get_movies())):

        total_n=0.0
        total_d=0.0
        total_dd=0.0
        for users in range(0,200):
            if (train_matrix_data[users][mlist[movies]-1] > 0) & (train_matrix_data[users][mid - 1] > 0):
                total_n+= (train_matrix_data[users][mlist[movies]-1] - train_avg[users+1])*(train_matrix_data[users][mid - 1] - train_avg[users+1])
                total_d+= math.pow(train_matrix_data[users][mlist[movies]-1] - train_avg[users+1],2)
                total_dd+= math.pow(train_matrix_data[users][mid-1] - train_avg[users+1],2)
        total=math.sqrt(total_d) * math.sqrt(total_dd)
        #print("total ",total)
        check(total,total_n,movies,mid)
    sort_list(item_based_similarity)

    calculate_final_k(item_based_similarity)



r_list={}
def avg_movie_r():

    for m in range(0,1000):
        total=0.0
        c=0
        for u in range(0,200):
            if train_matrix_data[u][m]!=0:
                total+=train_matrix_data[u][m]
                c+=1
        if c > 0:
            total=total/c
            r_list[m+1]=total
        else:
            r_list[m+1]=0.0

    return r_list

def calculate_result(i,user,mid,u_avg):

    n_val=0.0
    d_val=0.0
    total=0.0
    listr = avg_movie_r()
    mlist=user.get_movies()
    #print(listr)
    for val in range(0,len(itembased_final_k_neighbor)):
        neighbor=itembased_final_k_neighbor[val]
        neighbor_id=neighbor.get_user_id()

        if neighbor_id==mlist[i]:
            n_val+=itembased_final_k_neighbor[i].similarity * user.get_ratings()[i]
            d_val+= math.fabs(itembased_final_k_neighbor[i].similarity)
        if d_val>0:
            total=n_val/d_val
        elif listr[mid]!=0:
            total = listr[mid]
        else:
            total= u_avg

        #print(int(round(total)))
        return int(round(total))


def calculate_item_based(userid,mid,output):

    calculate_item_based_similarity(userid,mid)
    test_matrix=open_test()
    user=get_user_data(userid,test_matrix)
    u_avg=user.get_average()
    for i in range(0,len(user.get_movies())):
        result=calculate_result(i,user,mid,u_avg)
    ans = str(userid)+" "+str(mid)+" "+str(result)
    print(ans)
    output.write(ans+'\n')

#calculate_item_based(201,1)
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
        calculate_item_based(user,movie,output)


calling_method()