import tweepy
import json
import csv
import re
import pandas as pd
import nltk
import textblob
#open the json file 

# with open("C:\\Users\\Sukhman Singh\\Desktop\\Capstone Project\\twitter_credentials.json", "r") as file:
#     credentials = json.load(file)

# print(credentials)

auth =  tweepy.auth.OAuthHandler('RHFekehJM9sXk5zcc7TDuEbeD', 'M2gI9MC8XVpnzzYu2X7phyquraY02SCsNOFseQWt6ZEmjIpduP')

auth.set_access_token('1085752350759804929-2iLtD4lsamgXJzXvEh4rvBCzakDJg8', 'gfKcQzcBoV07evCB0O8CIR4SWZVGMxvNpZpI9JTEiCosu')

#Build API to fetch desired tweet and save it in a CSV file.
api = tweepy.API(auth, wait_on_rate_limit=True)

tweets = open('saved_tweets_new.csv', 'a')
write_tweets = csv.writer(tweets)
for tweet in tweepy.Cursor(api.search, q = 'bitcoin', since= "2019-01-01", until ="2019-01-10" , lang = 'en').items():
    write_tweets.writerow([tweet.text.encode('utf-8')])
tweets.close()

# Reading the file and Preprocessing the data.
# Main goal is to clean out punctuations, Stopwords, other symbiloc Characters.


df = pd.read_csv("saved_tweets_tobeused.csv", sep=',', 
                  names = ['Date','text'])
print(df.head())
dff = pd.DataFrame(df['text'])
dff_dates = pd.DataFrame(df['Date'])
print(dff_dates.head())
print(dff.head())


dff = dff.dropna(axis = 0, how = 'any')
def clean_tweet(data): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", data).split())



def remove_non_ASCII(data):
    return ''.join([x if ord(x) < 128 else '' for x in data ])
dff['text'] = dff['text'].apply(lambda x:x.replace('\n','') )   
dff['text']= dff['text'].apply(clean_tweet)
dff['text']= dff['text'].apply(remove_non_ASCII)
dff['text'] = dff['text'].apply(lambda x: x.lower())
dff['text'] = dff['text'].apply(lambda x: x.replace(' rt ', ''))
print(dff.head())

# # #Now convert Dataframe to an Array
# # # df1 = df['text']

df_Array = dff.values
print(df_Array)
print(len(df_Array))
from textblob import TextBlob
def senti_analysis(tweet):
    
    return TextBlob(tweet).sentiment.polarity
i = 0
a = []
while (i<len(df_Array)):
    
    if TextBlob(str(df_Array[i])).sentiment.polarity > 0:
        a.append(1)
    elif TextBlob(str(df_Array[i])).sentiment.polarity == 0:
        a.append(0)
    else:
        a.append(-1)
    #print(analysis)
    i = i+1
    continue
#print(a)    
with open("testing7.csv", 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in a:
        writer.writerow([val])


#Classification Using SVC()
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report  
from sklearn.pipeline import Pipeline
data = pd.read_csv("testing7.csv")
data = data.dropna(axis = 0, how ='any')
data2 = pd.read_csv("Polarity.csv")

data3 = pd.read_csv("Ether_values.csv")
result = pd.concat([data3,data2,data], axis=1 )
result = result.dropna(axis = 0, how ='any')

print(result.head(20))

X = result.iloc[:, 0:2].values
Y = result.iloc[:, 2:].values
y =[]

for i in Y:
    y.append(int(i))
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# regressor = RandomForestRegressor()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test) 

# print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
# print(metrics.mean_squared_error(y_test, y_pred))

pipeline = Pipeline([('scaler', StandardScaler()), ('SVM', SVC())])

pipeline.fit(X_train,y_train)

Predictions = pipeline.predict(X_test)
  
c_m = confusion_matrix(y_test, Predictions)
c_r = classification_report(y_test, Predictions)
m_a_e = metrics.mean_absolute_error(y_test,Predictions)

print(c_m,'\n', c_r,'\n', m_a_e)

#Plot the Confusion matrix


import matplotlib.pyplot as plt 
import numpy as np
#%matplotlib inline
#<<%Matplotlib will not work in normal Sublime environment
plt.imshow(c_m, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Neutral','Positive']
plt.title('SVM Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)


for i in range(3):
    for j in range(3):
        plt.text(j,i, str(c_m[i][j]))
plt.show()          





