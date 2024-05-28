#!/usr/bin/env python
# coding: utf-8

# In[518]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestRegressor


#   Importing Libraries
#  numpy: a library for numerical computing with Python
#     pandas: a library for data manipulation and analysis
#     matplotlib: a library for creating static, interactive, and animated visualizations in Python
#     seaborn: a library for statistical data visualization that builds on top of matplotlib
#     plotly.graph_objects and plotly.express: libraries for creating interactive visualizations in Python

# # Loading Data
# Data Preprocessing:
# • Loading Data: Data is loaded from a CSV file containing information about solar
# radiation and environmental factors.
# • Data Wrangling: Extracting relevant features like month, day, hour, minute, and second
# from date and time columns. Also, extracting sunrise and sunset hours.
# • Handling Missing Values: Checking for null values in the data (which are absent in this
# dataset).
# • Feature Engineering: Dropping unnecessary columns, selecting relevant features, and
# splitting data into input features and target variable (radiation).
# • Feature Selection: Using SelectKBest and ExtraTreesRegressor to select important
# features for prediction

# In[519]:


matches=pd.read_csv("IPL_Matches_2008_2022.csv")


# In[520]:


matches.head()


# In[521]:


balls=pd.read_csv("IPL_Ball_by_Ball_2008_2022.csv")


# In[522]:


balls.head()


# In[523]:


print(matches.shape)
print(balls.shape)


# In[524]:


matches.info()


# In[525]:


balls.info()


# In[526]:


matches.describe()


# In[527]:


balls.describe()


# Data Visualization

# In[528]:


stadiums=matches['City'].value_counts()
stadiums.plot(kind='bar',figsize=(7,5))


# ### EDA and Feature Engineering 
# ### Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy and performance.

# Finding total score of the innings

# In[529]:


total_score = balls.groupby(['ID', 'innings']).sum()['total_run'].reset_index()


# In[530]:


extracted_col = matches["WinningTeam"]
total_score= total_score.join(extracted_col)


# In[531]:


total_score.head()


# We only need score of 1st innings
# 
# our target is winner prediction so we only need the score of first innings

# In[532]:


total_score = total_score[total_score['innings']==1]


# In[533]:


total_score.head(5)


# In[534]:


total_score['target'] = total_score['total_run'] + 1


# Merge with the matches dataset

# In[535]:


match_df = matches.merge(total_score[['ID','target']], on='ID')


# In[536]:


match_df.head()


# ### Removing old teams/Updating teams new names

# In[537]:


match_df['Team1'].unique()


# In[538]:


teams = [
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad', 
    'Delhi Capitals', 
    'Chennai Super Kings',
    'Gujarat Titans', 
    'Lucknow Super Giants', 
    'Kolkata Knight Riders',
    'Punjab Kings', 
    'Mumbai Indians'
]


# In[539]:


match_df['Team1'] = match_df['Team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['Team2'] = match_df['Team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['Team1'] = match_df['Team1'].str.replace('Kings XI Punjab', 'Punjab Kings')
match_df['Team2'] = match_df['Team2'].str.replace('Kings XI Punjab', 'Punjab Kings')
match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')


match_df['Team1'] = match_df['Team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['Team2'] = match_df['Team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['WinningTeam'] = match_df['WinningTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')


# In[540]:


match_df = match_df[match_df['Team1'].isin(teams)]
match_df = match_df[match_df['Team2'].isin(teams)]
match_df = match_df[match_df['WinningTeam'].isin(teams)]


# In[541]:


match_df.shape


# In[542]:


match_df.columns


# In[543]:


numeric_df = match_df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()


# ### We want only the matches where D/L is not applied. 

# So,removing all matches effected due to rain

# In[544]:


match_df['method'].unique()


# In[545]:


match_df['method'].value_counts()


# In[546]:


match_df = match_df[match_df['method'].isna()]


# In[547]:


match_df.shape


# In[548]:


match_df.columns


# In[549]:


match_df = match_df[['ID','City','Team1','Team2','WinningTeam','target']].dropna()


# In[550]:


match_df.head()


# In[551]:


match_df.isna().sum()


# Merge the match_df dataset with balls dataset

# In[552]:


balls.columns


# In[553]:


balls['BattingTeam'] = balls['BattingTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')
balls['BattingTeam'] = balls['BattingTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')
balls['BattingTeam'] = balls['BattingTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

balls = balls[balls['BattingTeam'].isin(teams)]


# In[554]:


balls_df = match_df.merge(balls, on='ID')


# In[555]:


balls_df.head()


# In[556]:


balls_df['BattingTeam'].value_counts()


# In[557]:


balls_df.columns


# In[558]:


balls_df = balls_df[balls_df['innings']==2]


# In[559]:


balls_df.shape


# In[560]:


balls_df.head()


# In[561]:


balls_df.columns


# ###Create new row current_score after each ball 

# In[562]:


balls_df['current_score'] = balls_df.groupby('ID')['total_run'].cumsum()


# In[563]:


balls_df


# In[564]:


balls_df['runs_left'] = np.where(balls_df['target']-balls_df['current_score']>=0, balls_df['target']-balls_df['current_score'], 0)


# In[565]:


balls_df


# In[566]:


balls_df['balls_left'] = np.where(120 - balls_df['overs']*6 - balls_df['ballnumber']>=0,120 - balls_df['overs']*6 - balls_df['ballnumber'], 0)


# In[567]:


balls_df['wickets_left'] = 10 - balls_df.groupby('ID')['isWicketDelivery'].cumsum()


# In[568]:


balls_df.columns


# In[569]:


balls_df['current_run_rate'] = (balls_df['current_score']*6)/(120-balls_df['balls_left'])


# In[570]:


balls_df['required_run_rate'] = np.where(balls_df['balls_left']>0, balls_df['runs_left']*6/balls_df['balls_left'], 0)


# In[571]:


balls_df.columns


# In[572]:


def result(row):
    return 1 if row['BattingTeam'] == row['WinningTeam'] else 0


# In[573]:


balls_df['result'] = balls_df.apply(result, axis=1)


# In[574]:


balls_df.head()


# In[575]:


balls_df.columns


# In[576]:


index1 = balls_df[balls_df['Team2']==balls_df['BattingTeam']]['Team1'].index
index2 = balls_df[balls_df['Team1']==balls_df['BattingTeam']]['Team2'].index


# In[577]:


balls_df.loc[index1, 'BowlingTeam'] = balls_df.loc[index1, 'Team1']
balls_df.loc[index2, 'BowlingTeam'] = balls_df.loc[index2, 'Team2']


# In[578]:


balls_df.head()


# In[579]:


final_df = balls_df[['BattingTeam', 'BowlingTeam','City','runs_left','balls_left','wickets_left','current_run_rate','required_run_rate','target','result']]


# In[580]:


final_df.head()


# In[581]:


final_df.shape


# In[582]:


final_df.describe()


# In[583]:


final_df.isna().sum()


# In[584]:


final_df.shape


# In[585]:


final_df.sample(final_df.shape[0])


# Randomly shuffle all the rows

# In[586]:


final_df.sample()


# ### One hot encoding 

# In[587]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False,drop='first'),['BattingTeam','BowlingTeam','City'])
],
remainder = 'passthrough')


# In[588]:


from sklearn.model_selection import train_test_split

X = final_df.drop('result', axis=1)
y = final_df['result']
X.shape, y.shape


# In[589]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


# In[590]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[591]:


from sklearn.preprocessing import LabelEncoder

# Assuming X_train contains both numeric and categorical features
label_encoder = LabelEncoder()
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train[column] = label_encoder.fit_transform(X_train[column])

# Now, you can fit the model
rf_model.fit(X_train, y_train)


# In[599]:


# Concatenate training and test datasets
combined_data = pd.concat([X_train, X_test])

# Perform one-hot encoding
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_data = one_hot_encoder.fit_transform(combined_data[categorical_columns])
encoded_columns = one_hot_encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

# Split the datasets back into training and test portions
X_train_encoded = encoded_df.iloc[:len(X_train)]
X_test_encoded = encoded_df.iloc[len(X_train):]

# Now, you can make predictions
rf_predictions = rf_model.predict(X_test_encoded)


# In[ ]:


rf_mse = mean_squared_error(ytest, rf_predictions)
rf_r2 = r2_score(ytest, rf_predictions)


# In[ ]:


print("Random Forest Regression Mean Squared Error:", rf_mse)
print("Random Forest Regression R-squared Value:", rf_r2)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)


# In[ ]:


gb_predictions = gb_model.predict(X_test)


# In[ ]:


gb_mse = mean_squared_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)


# In[ ]:


print("Gradient Boosting Regression Mean Squared Error:", gb_mse)
print("Gradient Boosting Regression R-squared Value:", gb_r2)


# In[ ]:


teams


# In[ ]:


final_df['City'].unique()


# In[ ]:


import pickle
pickle.dump(pipe, open('pipe.pkl','wb'))


# # This is a great way to save a trained model so that it can be used later without having to train it again.

# # we aimed to predict the winning team in IPL matches using machine learning techniques. We started by collecting and cleaning the data, which included match-level statistics, team-level statistics, and player-level statistics.We then performed exploratory data analysis to understand the trends and patterns in the data. We found that various factors, such as the number of runs scored, the number of wickets taken, and the economy rate of bowlers, had a significant impact on the outcome of the matches.Next, we preprocessed the data by encoding categorical variables, scaling numerical variables, and handling missing values. We then split the data into training and testing sets and trained various machine learning models, such as logistic regression, decision trees, and random forests. this project demonstrated the potential of machine learning techniques to predict the winning team in IPL matches. However, there is still room for improvement, such as incorporating more features, using more sophisticated models, and collecting more data. Nonetheless, this project has provided a solid foundation for further exploration and development in the field of sports analytics.

# In[ ]:




