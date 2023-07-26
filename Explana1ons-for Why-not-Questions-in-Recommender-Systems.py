import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os 
from itertools import combinations

path=r"E:\Study Material\Tampere - Grad\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments\Assignment 1\ml-latest-small\ratings.csv"
data= pd.read_csv(path)

path1=r"E:\Study Material\Tampere - Grad\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments\Assignment 1\ml-latest-small\movies.csv"
movies_d= pd.read_csv(path1)


sys.path.append(os.path.abspath('E:\Study Material\Tampere - Grad\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments'))
from libs import Assignment1 as ur
from libs import Assignment2 as gr

#Average Group Recommendations
avg_g= gr.agg_group
avg_g20= avg_g.nlargest(n=20,columns='mean')

#Granualrity Case for Atomic 
#Why not matrix movie ==> item A
def why_not1(a):
    if a not in data['movieId'].values:
        print("Item does not exist in the database for the group") 
        return True
    return False

def tie(a,k):
    if avg_g.reset_index()[avg_g.reset_index()['movieId']==a]['mean'].values[0] in avg_g.head(k).values:
        print("There is a tie")

def top_k(a, k):
    top1_2k= avg_g[1*k: 2*k].reset_index()
    if a in top1_2k['movieId'].values :
        print("K maybe too low")   
        
def not_rating(a):
    if len(data[data['movieId']==a]['rating'])!=0:
        pass
    else:
        print("There are no rating scores for a")

#Peers of user
similarity= gr.sim[gr.sim['userId_x']<=3]
similarity_100= similarity[['userId_x','userId_y','sim']]

simm= pd.merge(similarity_100, data, left_on='userId_y', right_on='userId')
simm.drop(columns=['userId','timestamp'])
   
def peers(a):
    if len(simm[simm['movieId']==a]['rating'])==0:
        print('No peers rated this movie')
    else:
        print(simm[simm['movieId']==a][["userId_x", "userId_y","sim","rating"]])

def less_nump(a):
    nump_p= simm[simm['movieId']==a][["userId_x","userId_y","sim","rating"]].groupby('userId_x').agg({'rating':'count'})
    
    if len(nump_p[nump_p['rating']<5])!=0:
        print("Very few users rated this movie")
    
#Function --> Why not matrix movie ==> item A   
def why_not_main(a, k):   #movie should exist in avg_g
    if not why_not1(a):
        tie(a,k)
        top_k(a, k)
        not_rating(a)
        peers(a)
        less_nump(a)

 
#Position Absenteeism Case for Group
#Why not matrix movie first ==> item A first 
#Why not movie a at position of movie b

def why_not2(a, b):
    if a not in simm['movieId'].values and b in simm['movieId'].values:
        print("Peers rated item b and not a")
        
    a_like_avg = np.mean(simm[simm['movieId']==a]['rating'].values>=2.5)*100 
    b_like_avg = np.mean(simm[simm['movieId']==b]['rating'].values>=2.5) *100
    if a_like_avg<b_like_avg:
        print(f"Peers liked item b ({b_like_avg}%) and not a ({100-a_like_avg}%)")
    
    
    common_users = set(simm[simm['movieId']==a]['userId_y'].values).intersection(list(simm[simm['movieId']==b]['userId_y'].values))
    disagreement = np.mean(simm[(simm['movieId']==a) & (simm['userId_y'].isin(common_users)) ]['rating'].values >
                           simm[(simm['movieId']==b) & (simm['userId_y'].isin(common_users)) ]['rating'].values)*100
    if disagreement < 50:
        print(f"{100-disagreement}% of peers agree that movie b should be higher than movie a")


#Granularity Case for Group
#Why not action movies ==> genre A       
moviess= pd.merge(simm, movies_d, on='movieId')
mov= moviess.drop(columns=['userId','timestamp','title'])

#None of the peers has rated a specific genre (comedy, action etc.)
def com(a):
    if len(mov[mov['genres'].str.contains(a)])==0:
        print("None of the peers rated")  
        
#Only x of your peers like a specific genre (comedies, action etc.)        
def com_1(a):
    like_prop= np.mean(mov[mov['genres'].str.contains(a)]['rating'].values>3.5)*100
    if like_prop < 50:
        print(f"Only {like_prop}% of your peers liked the genre")
    
    data = pd.merge(gr.movies, movies_d, on='movieId')
    like_prop2 = np.mean(data[data['genres'].str.contains(a)].groupby('userId').agg({'rating':'mean'})['rating'].values>4)*100
    
    if like_prop2 < 50:
        print(f"Only {like_prop2} users in group liked the genre")
    
  
# Top 20 genres/movies
pd.merge(avg_g.nlargest(20, 'mean').reset_index(), movies_d, on='movieId')
