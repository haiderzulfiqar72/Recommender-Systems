# Recommender-Systems
This project implements various recommendation algorithms for collaborative filtering, group recommendations, sequential recommendations, and explores explanations for "why not" questions in recommender systems. It uses real-world movie rating data (MovieLens 100K) to provide personalized movie recommendations based on user preferences, group consensus, and sequential user behavior. The project showcases how different recommendation methods can be applied to cater to diverse user preferences and optimize the overall user experience.

1. Collaborative_Filtering.py:
This module demonstrates user-based collaborative filtering using the Pearson correlation function to compute user similarities. It predicts movie ratings for users and recommends movies based on the highest predicted ratings. The approach enables personalized movie recommendations and showcases how collaborative filtering can leverage user preferences for effective recommendations.

2. Group_Recommendations.py:
This module focuses on group-based recommendation methods and explores different techniques like average aggregation, least misery, and user satisfaction to provide movie suggestions for a group of users. It showcases how recommendations can be tailored to cater to the collective preferences and satisfaction of a group.

3. Sequential_Recommendation.py:
The sequential recommendation module provides a dynamic approach to movie recommendations. It uses sequential user behavior to refine movie suggestions over time, incorporating user satisfaction and preferences. The method adapts to users' changing tastes and offers personalized recommendations based on sequential movie ratings.

4. Explanations for Why not Questions in Recommender Systems.py:
This module addresses the "why not" questions in recommender systems, focusing on cases where certain movies or genres are not recommended. It explores various reasons, including insufficient data, peer preferences, disagreement among peers, and genre popularity within the user group. By analyzing these reasons, the system gains insights into improving recommendation accuracy and user satisfaction.
