import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os 
from itertools import combinations

path=r"E:\Study Material\Tampere - Grad\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments\Assignment 1\ml-latest-small\ratings.csv"
data= pd.read_csv(path)

sys.path.append(os.path.abspath('E:\Study Material\Tampere - Grad\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments'))
from libs import Assignment2 as gr

#movies
mov= gr.movies

#Relevance Scores
rel= gr.relevance

#Disagreement Variance
dis_v= gr.var

#average aggregation
avg_agg= gr.agg_group

#least misery
least_misery= gr.lmm_group

#User Satisfaction
def user_sat(a, agg, mov_list):

#denominator
    movies_s= mov[~ mov['movieId'].isin(mov_list)].sort_values(by= 'rating', ascending= False)
    mov_u1= movies_s[movies_s['userId']==a].nlargest(n=20,columns='rating')
    mov_u1_s= mov_u1['rating'].sum()

#numerator
    mov_20= agg.nlargest(n=20, columns='mean').reset_index()
    num_g_sat= mov[(mov['userId']==a) & (mov['movieId'].isin(mov_20['movieId']))]['rating'].sum()
    
#formula    
    u_sat= num_g_sat/mov_u1_s
  
    return u_sat

#In first iteration, our score is the same as the one coming from average aggregation method
seq_s = avg_agg
mov_list = []


print(seq_s.nlargest(n=20, columns=['mean']))

# Top-20 recommendations in 5 different sequences
for i in range(4):
    u_list=[(i, user_sat(i, seq_s, mov_list)) for i in [1,2,3]]
    user_list= pd.DataFrame(u_list)
    user_list.columns= ['UserId', 'Satisfaction Score']
  
    #Computing Dissatisfaction
    max_sat = max(user_list['Satisfaction Score'])
    min_sat= min(user_list['Satisfaction Score'])
    g_dissat= max_sat-min_sat
      
    #Sequential Score
    mov_list= np.append(mov_list, seq_s.nlargest(n=20, columns=['mean']).reset_index()['movieId'].values)
    
    # seq_s= pd.merge(avg_agg.reset_index(), least_misery.reset_index(), on='movieId')
    # seq_s['seq_score']= (1-g_dissat)*seq_s['mean']+ g_dissat*seq_s['min']
    
    seq_s= pd.merge(avg_agg.reset_index(), rel[['movieId','Disagreement']], on='movieId')  #Disagreements computed in assignment 2
    seq_s['seq_score']= (1-g_dissat)*seq_s['mean']+ g_dissat*rel['Disagreement']
        
    seq_s['is_in'] = np.where(seq_s['movieId'].isin(mov_list), 0, 1)
    seq_s['mean']= seq_s['seq_score']*seq_s['is_in']
    
    print(user_list)
    print(seq_s.nlargest(n=20, columns=['mean']))
