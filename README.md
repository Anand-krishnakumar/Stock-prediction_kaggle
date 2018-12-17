# Stock-prediction_kaggle
Introduction

Using market data provided by Intrinio and news data provided by Thomson Reuters to predict the stock price performance in the competition hosted by two sigma. The abundance of data available today enables investors to mine over it and make better investment decisions. The challenge in this competition is cleaning the data to find which is useful and which is not rather than choosing the right algorithm to predict data because data in this set is abundant and finding the content takes a lot of effort. By going through the market data and joining it with the news data to find the correct trends in market price change, we are going to predict the stock market prices for the future. 

Since the data is not freely available to download, we are supposed to use Kaggle notebooks and develop a kernel for the project and cannot use other sources to code. A small sample of the market data and news data is given though that is free to download. We are planning to use various preprocessing techniques to extract useful information from those 4 million records to build a better stock market prediction model.

In this project we predict a signed confidence value [-1,1] which is multiplied by the market-adjusted return of a given asset(Company/Industry) over a ten-day window. If the stock prices of a particular asset are predicted to be high over the next ten days compared to the entire market, then it will be assigned a large positive confidence value say 1.0 whereas if you expect the asset to have a negative return then you assign a large negative value say -1.0. 

Problem Definition and algorithm

The data that we have is related to the financial sector of the world and is pretty interesting as we can mine valuable information from that to predict change in stock markets which is pretty important these days as all the trends in prices of day-to-day products depends on the stock value of these major companies. If you have the power to predict that you can invest money in those companies to get good gains. Thus, its so important and interesting. The two sources of data are:
Market data provided by Intrinio (2007 to present) contains financial market information such as opening price, closing price, trading volume, calculated returns etc. 
News data provided by Thomson Reuters (2007 to present) which contains data about news articles/alerts published about assets, such article details, sentiment and other commentary.

Some columns in market data are :
time(datetime64[ns, UTC]) - the current time assetCode(object) - a unique id of an asset
assetName(category) - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
universe(float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
volume(float64) - trading volume in shares for the day
close(float64) - the close price for the day (not adjusted for splits or dividends)
The columns in news data are:
time(datetime64[ns, UTC]) - UTC timestamp showing when the data was available on the feed (second precision)
sourceTimestamp(datetime64[ns, UTC]) - UTC timestamp of this news item when it was created
firstCreated(datetime64[ns, UTC]) - UTC timestamp for the first version of the item
sentenceCount(int16) - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.
wordCount(int32) - the total number of lexical tokens (words and punctuation) in the news item
sentimentClass(int8) - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.
sentimentNegative(float32) - probability that the sentiment of the news item was negative for the asset
sentimentNeutral(float32) - probability that the sentiment of the news item was neutral for the asset
sentimentPositive(float32) - probability that the sentiment of the news item was positive for the asset
sentimentWordCount(int32) - the number of lexical tokens in the sections of the item text 
volumeCounts7D(int16) - same as above, but for 7 days
Since Gradient Boosting algorithms are good with categorical datasets for predicting the future, we have used LGB and XGB for predicting.
Experimental Evaluation 
Methodology
The main goal of the project is to design a model that is capable of predicting stock price changes using economic data from the company and news article titles.  The economic data provides details about the opening and closing stock prices for millions of companies over the course of several years. It also documents certain details about the company such as volumes of shares traded that day. The news article dataset includes the titles of articles as well as some pertain information about the article, such as positivity of the articles sentiment or its novelty. We hypothesize that the gradient boosting model developed will predict the opening prices of company stock with at least 70% accuracy using the specified parameters. 
The data has some errors and outliers which must first be accounted for through preprocessing. Certain dates had exceptionally high changes in prices that could not be explained by actual events. It is assumed that these events are possible errors in documentation. To remove the errors and any outliers, The data was isolated to any datasets with price_diff that did not deviate from the mean price by more than 100. The second preprocessing step is to remove all data before 2010. This removes the influences of the financial crisis in 2008, thereby better isolating the effects of our parameters.
Below are few graphs which we developed for easier preprocessing by employing visual analysis:
 

For the modelling procedures, the data split into a 10% testing set. The first model uses the LGB algorithm with parameters:
learning_rate': 0.01, 
'max_depth': 12,
 'boosting': 'gbdt',
 'objective': 'binary',
 'metric': 'auc',
 'is_training_metric': True,
 'seed': 42
The second model employs XGB algorithm with the following parameters:
n_jobs=8,
n_estimators=300,
max_depth=12,
booster='gbtree',
 learning_rate=0.01,
 objective='binary:logistic',
 eta=0.15,
 random_state=42
Both models are tested against the test set to evaluate the accuracy of the models using the builtin function within the XGB and LGB functions.

Results



Related Work
Our project is based on the work of Andrew Lukyanenko who posted a solution to kaggle under the name EDA, feature engineering and everything[1]. Lukyanenko’s methodology used LGB and resulted in an accuracy rating of roughly 60%. Our project strove to improve on Lukyanenko’s work by employing XGB as well, comparing the accuracy results of both.
Future Work
Try to get better results by further preprocessing the dataset and also use multiple other machine learning algorithms to see how better they fare.

Conclusion
Accuracy of LGB : 55.01 percent,    Accuracy of XGB: 55.67 percent
Precision of LGB : .55 		   Precision of XGB: .56
Recall of LGB : 0.62			   Recall of XGB: 0.63
The results reflect a correlation between specific media topics and price changes in stocks. The accuracy of the results suggests a connection between headlines and the stock evaluation of companies. This would make sense for a number of reasons. We noticed that some key phrases were earning update related such as “second quarter”. A company's quarterly earnings may be a metric people use to make trade decisions. There were also strong correlations with phrases such as “research roundup” and “buzz stocks”, suggesting some trades being done as a result of prospective changes in a company, such as new research findings in that field or social excitement about a company. This information has implications on stock evaluation, particularly for companies that are involved in daytrade. There are also noteworthy questions about the media and its influence on company stock values as well as media impact on the economy as a whole. 
Bibliography
[1]	Lukyanenko, Andrew. “EDA, Feature Engineering and Everything.” Kaggle.com, 2018, www.kaggle.com/artgor/eda-feature-engineering-and-everything.
[2]	XGBoost Documentation https://xgboost.readthedocs.io/en/latest/index.html
[3]	StockMarket Data https://money.cnn.com/data/markets/


