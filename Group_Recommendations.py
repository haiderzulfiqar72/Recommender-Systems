import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os 
from itertools import combinations

path=r"E:\Study Material\Tampere - Grad\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments\Assignment 1\ml-latest-small\ratings.csv"
data= pd.read_csv(path)

sys.path.append(os.path.abspath('E:\Study Material\Tampere - Grad\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments'))
from libs import Assignment1 as ur

#similarity matrix
sim = ur.user_sim(data)

r_movies=[] 
#Group of user Id 1,2,3
users_g= np.array(np.arange(1,4))

for a in users_g:
    for p in data['movieId'].unique():
        
        #if user a has not watched movie p
        if p not in data[data['userId']==a]['movieId']:
            c= ur.predictionfunc(a,p,sim,data)
            r_movies.append((a,p,c))
       
movies= pd.DataFrame(r_movies)
movies.columns=['userId','movieId','rating']

#Average Method
#mean rating of movie p for the group
agg_mean = movies.groupby('movieId').agg({'rating':'mean', 'userId':'count'})
avg_avg= agg_mean.rename(columns={'rating':'mean', 'userId':'count'})
avg_group= avg_avg[avg_avg['count']==3]
agg_avg= avg_group.drop(columns=['count'])
agg_group= agg_avg.sort_values(by='mean', ascending=False)

#Top 20 movies for the group
print(agg_group.nlargest(n=20,columns='mean'))

#Least Misery Method
#min rating of movie p for the group
lm_min = movies.groupby('movieId').agg({'rating':'min', 'userId':'count'})
lm_remin= lm_min.rename(columns={'rating':'min', 'userId':'count'})
lm_group= lm_remin[lm_remin['count']==3]
lmm_lmm= lm_group.drop(columns=['count'])
lmm_group= lmm_lmm.sort_values(by='min', ascending=False)

#Top 20 movies for the group
print(lmm_group.nlargest(n=20,columns='min'))

#b)
#Computing Disagreements

#Method 1: Average Pairwise Disagreements

#defining a group of 3 users
for indx, (user1, user2) in enumerate(combinations([1,2,3],2)):
    
    #defining rating against each user for a specific movie ID
    user_Id1= movies[movies['userId']==user1][['movieId','rating']]
    user_Id2= movies[movies['userId']==user2][['movieId','rating']]
    
    user_1_2= pd.merge(user_Id1,user_Id2, on='movieId')
    
    #Computing summation of relevance for each user
    user_1_2['rel_'+str(indx)] = user_1_2.apply(lambda x: abs(x['rating_x'] - x['rating_y']), axis=1)
    user_1_2 = user_1_2[['movieId', 'rel_'+str(indx)]]
    
    if indx == 0:
        rel = user_1_2
    else:
        rel = pd.merge(rel, user_1_2, on='movieId')
 
#Computing relevance and disagreement along each user for the group        
rel_long = pd.wide_to_long(rel, ['rel_'], i='movieId', j='user_combination_index')
relevance = rel_long.rename(columns={'rel_':'rel'}).groupby('movieId').agg({'rel':'sum'}).reset_index()
relevance['Disagreement']= (2 / (len(users_g)*(len(users_g)-1))) * relevance['rel']


#Method 2: Disagreement Variance

#Computing mean rating for each user
user_mean =movies.groupby('userId').agg({'rating':'mean'}).reset_index().rename(columns={'rating':'user_mean_rating'})

#Computing summation of variance along each user
u_m= pd.merge(user_mean, movies, on= 'userId')
u_m['mean_square']= (u_m['rating']- u_m['user_mean_rating'])**2

#Computing disagreement for each movie for the group
var= u_m.groupby('movieId').agg({'mean_square':'mean'}).reset_index().rename(columns={'mean_square':'Disagreement'})

#Computation Suggestions for the Group
#Defining weights --> w1+w2= 1
w1= 0.75
w2= 0.25

#Suggestions via Pairwise Disagreement Method
consensus_func= pd.merge(agg_group, relevance, on='movieId')
consensus_func['Suggestion'] = w1 * consensus_func['mean'] + w2 * (1-consensus_func['Disagreement'])

#Top 20 movies for the group
print(consensus_func[['movieId','Suggestion']].nlargest(n=20,columns='Suggestion'))

#Suggestions via Disagreement Variance 
consensus_func1= pd.merge(agg_group, var, on='movieId')
consensus_func1['Suggestion'] = w1 * consensus_func1['mean'] + w2 * (1-consensus_func1['Disagreement'])

#Top 20 movies for the group
print(consensus_func1[['movieId','Suggestion']].nlargest(n=20,columns='Suggestion'))



#ROUGH WORK

#Pairwise Disagreement    
# user_Id1= movies[movies['userId']==1][['movieId','rating']]
# user_Id2= movies[movies['userId']==2][['movieId','rating']]
# user_Id3= movies[movies['userId']==3][['movieId','rating']]

# user_1_2= pd.merge(user_Id1,user_Id2, on='movieId')
# user_2_3= pd.merge(user_Id2,user_Id3, on='movieId')
# user_1_3= pd.merge(user_Id1,user_Id3, on='movieId')

# user_1_2['rel_1_2'] = user_1_2.apply(lambda x: abs(x['rating_x'] - x['rating_y']), axis=1)
# user_2_3['rel_2_3'] = user_2_3.apply(lambda x: abs(x['rating_x'] - x['rating_y']), axis=1)
# user_1_3['rel_1_3'] = user_1_3.apply(lambda x: abs(x['rating_x'] - x['rating_y']), axis=1)

# rel= pd.merge(pd.merge(user_1_2,user_2_3,on='movieId'),user_1_3,on='movieId')
# rel['rel']= rel['rel_1_2'] + rel['rel_2_3'] + rel['rel_1_3']
# relevence= rel[['movieId','rel']]

# relevence['Disagreement']= (2 / len(users_g)*(len(users_g)-1)) * relevence['rel']

#movies.groupby('movieId').agg({['userId':,'rating']

#DV
# rel_1= pd.merge(user_1_2, movies, on='movieId')
# rel_2= pd.merge(user_2_3, movies, on='movieId')
# rel_3= pd.merge(user_1_3, movies, on='movieId')

# rel_1['var_1']= (rel_1['rating'] - rel_1['mean_rating'])**2
# rel_2['var_2']= (rel_2['rating'] - rel_2['mean_rating'])**2
# rel_3['var_3']= (rel_3['rating'] - rel_3['mean_rating'])**2

# vr= pd.merge(pd.merge(rel_1,rel_2,on='movieId'),rel_3,on='movieId')
# vr['rel']= rel['rel_1_2'] + rel['rel_2_3'] + rel['rel_1_3']
# variance_comp= vr[['movieId','rel']]
# variance_comp['Disagreement']= (1 / len(users_g)) * variance_comp['rel']

# movies_df = pd.merge(iu, movies)
# # movies.insert('userId','movieId'()
# movies[movies['rating']<=5]
#df= data[['userId','movieId']]
# iu= df.loc[df['userId'] < 4]

# users_g= np.array(np.arange(1,4))
# scores=[]
# for i in users_g:
#     p= 10
#     pred= ur.predictionfunc(i,p,sim,data)
#     scores.append(pred)
#     print(pred)       
# print(np.mean(scores))
  


# # def gr(groups,p,sim,data):

# #     sim = ur.user_sim(data)
# #     users_g= np.array(np.arange(0,4))
# #     for i in users_g:
# #         p= iu['movieId']
# #         pred= ur.predictionfunc(i,p,sim,data)
# #     return pred
    
# a = 3
# b= 4
# c= 6

# p= 5
# q= 6
# r= 7

# pred= ur.predictionfunc(a,p,sim,data)
# pred1= ur.predictionfunc(b,p,sim,data)
# pred2= ur.predictionfunc(c,p,sim,data)

# pred3= ur.predictionfunc(a,q,sim,data)
# pred4= ur.predictionfunc(b,q,sim,data)
# pred5= ur.predictionfunc(c,q,sim,data)

# pred6= ur.predictionfunc(a,r,sim,data)
# pred7= ur.predictionfunc(b,r,sim,data)
# pred8= ur.predictionfunc(c,r,sim,data)


# p_m = np.array([pred,pred1,pred2])
# p_movie= np.average(p_m)

# q_m = np.array([pred3,pred4,pred5])
# q_movie= np.average(q_m)

# r_m = np.array([pred6,pred7,pred8])
# r_movie= np.average(r_m)

# def gr(groups):
    
#     users_g= np.array(np.arange(1,4)) #group of userID 0,1,2,3
#     p= 5 #item 5
#     for i in users_g:
#         sim = ur.user_sim(data)
#         pred= ur.predictionfunc(i,p,sim,data)
     
#     return gr

#     pred= ur.predictionfunc(a,p,sim,data)
    
