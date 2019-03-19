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

# auth =  tweepy.auth.OAuthHandler('RHFekehJM9sXk5zcc7TDuEbeD', 'M2gI9MC8XVpnzzYu2X7phyquraY02SCsNOFseQWt6ZEmjIpduP')

# auth.set_access_token('1085752350759804929-2iLtD4lsamgXJzXvEh4rvBCzakDJg8', 'gfKcQzcBoV07evCB0O8CIR4SWZVGMxvNpZpI9JTEiCosu')

# #Build API to fetch desired tweet and save it in a CSV file.
# api = tweepy.API(auth, wait_on_rate_limit=True)

# tweets = open('Bit_Tweets_Dated.csv', 'a')
# write_tweets = csv.writer(tweets)
# for tweet in tweepy.Cursor(api.search, q = 'bitcoin', lang = 'en').items():
#     write_tweets.writerow([tweet.text.encode('utf-8')])
# tweets.close()


#1510365360



df = pd.read_csv("Bit_Tweets_Dated.csv", sep=',', 
                   names = ['text'])
#print(df.head())
df = pd.DataFrame(df['text'])
# dff_dates = pd.DataFrame(df['Date'])
# print(dff_dates.head())
print(df.head())


df = df.dropna(axis = 0, how = 'any')
def clean_tweet(data): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", data).split())

df['text'] = df['text'].apply(lambda x:x.replace('\n','') )
df['text'] = df['text'].apply(clean_tweet)
print(df.info())


# Sentiment Analysis
from textblob import TextBlob
def senti_analysis(tweet):
    
    return TextBlob(tweet).sentiment.polarity

a =[]
i = 0
for i in df['text']:
    
    pol = senti_analysis(i)
    a.append(pol)

#print(a)   
df2 = pd.DataFrame({'POLARITY':a})
#print(df2.head(10))
#Building the data frame of Polarity in int form
POLS=[]
for j in df['text']:    
    if TextBlob(str(j)).sentiment.polarity > 0:
        POLS.append(1)
    elif TextBlob(str(j)).sentiment.polarity == 0:
        POLS.append(0)
    else:
        POLS.append(-1)
#     #print(analysis)


    

df3 = pd.DataFrame({'Senti':POLS})  
print() 

#1514

Bit_Vals =pd.read_csv("Bit_Values.csv")
Bit_Vals = Bit_Vals.dropna(axis = 0, how = 'any')
print(Bit_Vals.info())
df_time = Bit_Vals["Timestamp"]
BIT_Volume =Bit_Vals["Volume_(BTC)"]
BIT_Price = Bit_Vals["Weighted_Price"]
#print(BIT_Price.head(20))

#Building a Final Data Frame that will Be Used

""" DataFrame for Classification Problem """

Class_df = pd.concat([df_time,df2,BIT_Volume,BIT_Price,df3], axis=1 )
print(Class_df.tail(10))


Classification_DF = Class_df.dropna(axis = 0, how = 'any')
print(Classification_DF.tail(10))

""" DataFrame for Regression Problem """
Regress_df = pd.concat([df_time,df2,df3,BIT_Volume,BIT_Price], axis=1 )


Regress_DF = Regress_df.dropna(axis = 0, how = 'any')
print(Regress_DF.tail(10))

from sklearn.svm import SVC
from sklearn import metrics   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.model_selection





""" Plotting the Confusion Matrix
       UNCOMMENT TO USE """

# import matplotlib.pyplot as plt 
# import numpy as np
# #%matplotlib inline
# #<<%Matplotlib will not work in normal Sublime environment
# plt.imshow(c_m, interpolation='nearest', cmap=plt.cm.Wistia)
# classNames = ['Negative','Neutral','Positive']
# plt.title('SVM Confusion Matrix - Test Data')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# tick_marks = np.arange(len(classNames))
# plt.xticks(tick_marks, classNames, rotation=45)
# plt.yticks(tick_marks, classNames)


# for i in range(3):
#     for j in range(3):
#         plt.text(j,i, str(c_m[i][j]))
# plt.show()          





