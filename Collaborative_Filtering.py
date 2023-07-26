import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data
path=r"E:\Study Material\Masters\Studies\Semester 1\Period II\Recommender Systems - DATA.ML.360\Assignments\Assignment 1\ml-latest-small\ratings.csv"
data= pd.read_csv(path)
print(data)

#data exploration
df= pd.DataFrame(data)
print(df.head())
print(df.info())
print(df.shape)
print(df[df.columns[2]].count())
df.hist(column='rating', bins=5)
print(df.nunique())
print(df.describe())

#Implement the user-based collaborative filtering approach, using the Pearson correlation function for computing similarities between users
agg_users = df.groupby('userId').agg(mean_rating = ('rating', 'mean'))
print(agg_users)

#self join on movieId for user user similarity --> rating of the equivalent movie that user a has watched against all a-users (looping over users)
df_join= pd.merge(df, df, how="inner",on='movieId')

#Rating Mean for each user x and y
r_mean=df_join.groupby(['userId_x','userId_y']).agg('mean')
r_mean.columns= (['movieId', 'rx_mean', 'timestamp_x', 'ry_mean', 'timestamp_y'])

#merge above two to get associated means in the initial dataframe and computing numerator and denominators of Pearson Correaltion Equation
new=pd.merge(df_join, r_mean, on=('userId_x','userId_y'))
new['numerator']=(new['rating_x']-new['rx_mean']) *  (new['rating_y']-new['ry_mean'])
new['denominator_1']= (new['rating_x']-new['rx_mean'])**2
new['denominator_2']= (new['rating_y']-new['ry_mean'])**2

#Pearson Correlation Function
sim= new.groupby(['userId_x','userId_y']).agg({'numerator':'sum','denominator_1':'sum', 'denominator_2': 'sum'}).reset_index()
sim['sim']= sim['numerator'] / np.sqrt(sim['denominator_1']*sim['denominator_2'])
sim=sim.fillna(0)

#Prediction Function
def predictionfunc(a,p):

    #user a mean
    r_a= df[df['userId']==a]['rating'].mean()

    # # set of users who have a similarity with user a
    # # n_p= sim[sim['userId_x']==a]['userId_y']
    
    # set of users that rate item p
    n_p= df[df['movieId']==p]['userId']

    num_list=[]
    den_list=[]

    for b in n_p:
        #user b mean
        r_b= df[df['userId']==b]['rating'].mean()
        
        #sim for user a and b
        sim_ab= sim[(sim['userId_x'] == a) & (sim['userId_y'] == b) ]['sim']
        
        #we get out of bounds error when len(sim_ab)=0
        if len(sim_ab)!=0:
            sim_ab= sim[(sim['userId_x'] == a) & (sim['userId_y'] == b) ]['sim'].values[0]
        else:
            sim_ab= 0
    
        #rating of user b for item p
        r_bp= df[(df['userId']==b) & (df['movieId']==p)]['rating'].values[0]
    
        num_sum= (sim_ab * (r_bp-r_b))
        den_sum= abs(sim_ab)
        
        num_list.append(num_sum)
        den_list.append(den_sum)

    pred= r_a + (sum(num_list)/sum(den_list))

    return pred

#10 most similar users to userId 3
sim[(sim['userId_x'] == 3)][['sim','userId_y']].nlargest(n=10,columns='sim')

#20 most relevant movies to userId 21
fav_movies=[]
for p in df['movieId'].unique():
    c= predictionfunc(2,p)
    fav_movies.append(c)
    #print(fav_movies)
    
movies= pd.DataFrame(fav_movies)
movies.columns=['rating']
movies[movies['rating']<=5].nlargest(n=20,columns='rating')

#Implement the item-based collaborative filtering approach, using the cosine similarity for computing similarities between items 

#self join on movieId for movie-movie similarity --> rating of a, b movies that user a has watched (looping over movies)
df_movie= pd.merge(df, df, how="inner",on='userId')

#Rating Mean for user u
u_mean=df.groupby('userId').agg(mean_rating = ('rating', 'mean'))

#merge above two to get associated means in the initial dataframe 
items= pd.merge(df_movie[['userId','movieId_x', 'rating_x', 'movieId_y','rating_y']], u_mean, on=('userId'))

#Computing numerator and denominator of cosine similarity function
items['num']=(items['rating_x']-items['mean_rating']) * (items['rating_y']-items['mean_rating'])
items['den_1']= (items['rating_x']-items['mean_rating'])**2
items['den_2']= (items['rating_y']-items['mean_rating'])**2

#cosine similarity equation
cos= items.groupby(['movieId_x','movieId_y']).agg({'num':'sum','den_1':'sum', 'den_2': 'sum'}).reset_index()
cos['cos']= cos['num'] / np.sqrt(cos['den_1']*cos['den_2'])
cos=cos.fillna(0)

#Prediction function for predicting movies scores
def preditem(a,p):
    
    #rating of a seen movie of user a
    user_a= df[df['userId']==a].loc[:,'movieId':'rating']
    
    #similariy between seen movie and an unseen movie
    sim_a= pd.merge(cos[cos['movieId_x']==p],user_a, how='inner', left_on=['movieId_y'],right_on=['movieId'])

    den= sum(abs(sim_a['cos']))
    
    if den == 0:
        return np.nan
    num= sum (sim_a['rating'] * sim_a['cos'])
    
    pred= num/den
    
    return pred

#20 most relevant movies to userId 21
relevant_movies=[]
relevant_ratings=[]

for p in df['movieId'].unique():
    r= preditem(22,p)
    relevant_movies.append(p)
    relevant_ratings.append(r)
    #print(rel_movies)
    
rel_movies= pd.DataFrame({'movieId':relevant_movies,'rating':relevant_ratings})   
rel_movies.nlargest(n=20,columns='rating')
