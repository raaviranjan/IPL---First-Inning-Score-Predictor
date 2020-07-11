#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import pickle


ipl = pd.read_csv('ipl.csv')


ipl.head()


ipl.columns


columns_to_remove = ['mid','batsman', 'bowler','striker', 'non-striker']
ipl.drop(columns_to_remove,axis=1, inplace=True)


ipl.head()


ipl['bat_team'].unique()


present_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab','Royal Challengers Bangalore', 'Delhi Daredevils', 'Sunrisers Hyderabad']


ipl = ipl[(ipl['bat_team'].isin(present_teams)) & (ipl['bowl_team'].isin(present_teams))]


#To predict, we need atleast 5 over data. So removing first 5 over data form the dataset
ipl = ipl[ipl['overs']>=5.0]



ipl.head()


ipl['venue'].value_counts()


ipl['venue'] = ipl['venue'].replace(['Punjab Cricket Association Stadium, Mohali'],'Punjab Cricket Association IS Bindra Stadium, Mohali')



from datetime import datetime
ipl['date'] = ipl['date'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))



# --- Data Preprocessing ---
# Using OneHotEncoding to convert categorical features

encoded_ipl = pd.get_dummies(data = ipl, columns = ['venue','bat_team','bowl_team'])


encoded_ipl.columns


encoded_ipl = encoded_ipl[['date','bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','venue_Barabati Stadium', 'venue_Brabourne Stadium',
       'venue_Buffalo Park', 'venue_De Beers Diamond Oval',
       'venue_Dr DY Patil Sports Academy',
       'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens',
       'venue_Feroz Shah Kotla',
       'venue_Himachal Pradesh Cricket Association Stadium',
       'venue_Holkar Cricket Stadium',
       'venue_JSCA International Stadium Complex', 'venue_Kingsmead',
       'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
       'venue_Maharashtra Cricket Association Stadium',
       'venue_New Wanderers Stadium', 'venue_Newlands',
       'venue_OUTsurance Oval',
       'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
       'venue_Rajiv Gandhi International Stadium, Uppal',
       'venue_Sardar Patel Stadium, Motera', 'venue_Sawai Mansingh Stadium',
       'venue_Shaheed Veer Narayan Singh International Stadium',
       'venue_Sharjah Cricket Stadium', 'venue_Sheikh Zayed Stadium',
       "venue_St George's Park", 'venue_Subrata Roy Sahara Stadium',
       'venue_SuperSport Park', 'venue_Wankhede Stadium','runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'total']]





# --- Splitting the data ---
X_train = encoded_ipl.drop('total',axis = 1)[encoded_ipl['date'].dt.year<=2016]
X_test = encoded_ipl.drop('total',axis = 1)[encoded_ipl['date'].dt.year>=2017]


y_train = encoded_ipl[encoded_ipl['date'].dt.year<=2016]['total'].values
y_test = encoded_ipl[encoded_ipl['date'].dt.year>=2017]['total'].values


X_train.drop('date',axis=1,inplace = True)
X_test.drop('date',axis=1,inplace = True)


# --- Model Building ---
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(lr, open(filename, 'wb'))
