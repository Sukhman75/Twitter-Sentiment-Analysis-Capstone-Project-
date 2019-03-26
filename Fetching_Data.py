#Import all the important Libraries.
import tweepy
import json
import csv
import re
import pandas as pd
import nltk
import numpy as np 
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

#Read the Tweets CSV file.
df = pd.read_csv("Bit_Tweets_Dated.csv", sep=',', 
                   names = ['text'])
#print(df.head())
df = pd.DataFrame(df['text'])
print(df.head())
#Clean out the empty row is any.
df = df.dropna(axis = 0, how = 'any')

def clean_tweet(data): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", data).split())

df['text'] = df['text'].apply(lambda x:x.replace('\n','') )
df['text'] = df['text'].apply(lambda x:x.replace("b'",''))
df['text'] = df['text'].apply(clean_tweet)
print(df.info())


# Sentiment Analysis
from textblob import TextBlob
def senti_analysis(tweet):
    """This function will return the Polarity of the sentiments of tweet """
    return TextBlob(tweet).sentiment.polarity
#Create a empty list to store the polarities of the sentiments
a =[]
i = 0
for i in df['text']:
    
    pol = senti_analysis(i)
    a.append(pol)

#print(a)   
#Convert the above list into a dataframe
df2 = pd.DataFrame({'POLARITY':a})
#print(df2.head(10))

POLS=[]
for j in df['text']:    
    if TextBlob(str(j)).sentiment.polarity > 0:
        POLS.append(1)
    # elif TextBlob(str(j)).sentiment.polarity == 0:
    #     POLS.append(0)
    else:
        POLS.append(-1)
    #print(analysis)
    
 
df3 = pd.DataFrame({'Senti':POLS})   
#print(df3.tail(20))

#Read the Historical Data of BITCOIN(Downloaded from cable)

Bit_Vals =pd.read_csv("Bit_Values.csv")
#Drop the empty rows if any.
#Bit_Vals = Bit_Vals.dropna(axis = 0, how = 'any')
print(Bit_Vals.head())
df_time = Bit_Vals["Timestamp"]
BIT_Volume =Bit_Vals["Volume_(BTC)"]
BIT_Open_Price = Bit_Vals["Open_Val"]
BIT_Price = Bit_Vals["Weighted_Price"]
BIT_Price = BIT_Price.fillna(BIT_Price.mean())

#print(BIT_Price.head(20))

#Building a Final Data Frame that will Be Used

""" DataFrame for Classification Problem """

Class_df = pd.concat([df_time,df2,BIT_Volume,BIT_Price,df3], axis=1 )
#print(Class_df.head(10))


Classification_DF = Class_df.dropna(axis = 0, how = 'any')
#print(Classification_DF.head(10))


# To convert dataframe to csv please UNCOMMENT the below
#Classification_DF.to_csv(Classification.csv, encoding='utf-8', index=False)



""" DataFrame for Regression Problem """
Regress_df = pd.concat([df_time,df2,BIT_Volume,BIT_Open_Price,BIT_Price], axis=1 )
#Regress_df = pd.concat([df_time,df2,BIT_Price], axis=1 )


Regress_df = Regress_df.dropna(axis = 0, how = 'any')
#print(Regress_df.tail(10))



# To convert dataframe to csv please UNCOMMENT the below
#Regress_DF.to_csv(regessor.csv, encoding='utf-8', index=False)


###############################################################################################
#Classification

from sklearn.svm import SVC
from sklearn import metrics   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn import preprocessing, metrics, svm, tree, ensemble

# X = Classification_DF.drop(["Senti"], axis =1)
# print(X.head())
# #X_F = X.iloc[:, 0:4].values 
# #print(X_F.head())
# Y = Classification_DF["Senti"]
# #Y = Classification_DF.iloc[:,4].values  
# #print(Y.head())
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

# pipeline = Pipeline([('scaler', StandardScaler()), ('SVM', SVC())])

# pipeline.fit(X_train,y_train)

# Predictions = pipeline.predict(X_test)

# c_m = metrics.confusion_matrix(y_test, Predictions)
# c_r = metrics.classification_report(y_test, Predictions)
# m_a_e = metrics.mean_absolute_error(y_test,Predictions)

# print(c_m,'\n', c_r,'\n', m_a_e)
# print(metrics.accuracy_score(y_test, Predictions))

# """ Plotting the Confusion Matrix
#        UNCOMMENT TO USE """

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


################
"""  K_FOLD for testing the best Classifier"""


# KF = sklearn.model_selection.StratifiedKFold( n_splits=10, shuffle=True)
# KF.get_n_splits(Y,X_F)
# #print(KF)
# # Build A function to validate the best classifier
# def folds(X, Y,Classifier, Kf):
#   y_pred = Y.copy()
#   for ii,jj in KF.split(X, Y):
#       X_train, X_test = X[ii], X[jj]
#       y_train = Y[ii]
#       clf = Classifier()
#       clf.fit(X_train,y_train)
#       y_pred[jj] = clf.predict(X_test)
#   return y_pred

# #folds(X, Y,ensemble.GradientBoostingClassifier, KF)
# # Print The Accuracy results of Different Classifier
# print(metrics.accuracy_score(Y, folds(X_F, Y, ensemble.GradientBoostingClassifier, KF)))
# print(metrics.accuracy_score(Y, folds(X_F, Y, ensemble.RandomForestClassifier, KF)))
# print(metrics.accuracy_score(Y, folds(X_F, Y, SVC, KF)))

#######################################################################################################
#Regression for Prediction of Bitcoin

X_feat = Regress_df.drop(["Weighted_Price"], axis=1)
X_feat = X_feat.values
Y_targ = Regress_df["Weighted_Price"].values
Y_targ = Y_targ.reshape(-1, 1)

# X_feat = Regress_df.iloc[:,0:1]
# Y_targ = Regress_df.iloc[:,2]
# Y_targ= np.ravel(Y_targ)
#Y_targ = Y_targ.reshape(-1, 1)

sc = StandardScaler()

X_feat = sc.fit_transform(X_feat)
Y_targ = sc.fit_transform(Y_targ)


#X_feat_train, X_feat_test, y_train, y_test = train_test_split(X_feat, Y_targ, test_size=0.2, random_state = 4)

#Mutiple Linear Regression
import statsmodels.api as sm
X_Scaled1 = sm.add_constant(X_feat)
model = sm.OLS(Y_targ, X_Scaled1)
fitt = model.fit()
print(fitt.summary())
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# RFR = ensemble.RandomForestRegressor(n_estimators = 400, random_state=4)

# model = RFR.fit(X_feat_train,y_train)
# Predict_y = RFR.predict(y_test)

# print(metrics.accuracy_score(y_test, Predict_y))
# print(metrics.mean_absolute_error(y_test, Predict_y))
"""  INCOMPLETE Neural Network Implementation of Regressio/Prediction Problem """

# from keras.models import Sequential
# from keras.layers import Dense

# model = Sequential()

# input_layer = Number of Features in Regression DataFrame
# #Adding Layers to the model
# model.add(Dense(150, activation='relu', input_shape=(n_cols,)))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(1))

# #Compile the model Taking Mean_Squared_error as the performance measure

# model.compile(optimizer='adam', loss ='mean_squared_error')

# #Fit/Train the created model
# model.fit(train_data, test_data, validation=0.2, epochs=, callbacks=[])









