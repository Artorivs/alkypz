import os
import pandas as pd
from textblob import TextBlob
from googletrans import Translator

## root directory
## the path before the current directory
root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path_data = os.path.join(root, 'data-tweet')
translator = Translator(service_urls=['translate.googleapis.com'])

## read the data
df_vacc_us = pd.read_csv(os.path.join(path_data, 'US COVID-19 Tweets.csv'), usecols=['datetime', 'text'])
df_vacc_id = pd.read_csv(os.path.join(path_data, 'indonesia-TRANSLATED-covid-sentiment.csv'), usecols=['date', 'translated'])
df_vacc_in = pd.read_csv(os.path.join(path_data, 'india_covid_tweets_trans2.csv'))

df_vacc_id = (df_vacc_id.rename(columns={'translated': 'text'})
                        .dropna(subset=['text'])
                        .assign(country='id')
              )

df_vacc_us.datetime = pd.to_datetime(df_vacc_us.datetime)
df_vacc_us['date'] = df_vacc_us.datetime.dt.date
df_vacc_us = df_vacc_us.assign(country='us')
df_vacc_us.drop(columns=['datetime'], inplace=True)

df_vacc_in = (df_vacc_in.melt(id_vars=['date'], 
                             value_vars=[i for i in df_vacc_in.columns if i.endswith('hashtags') != True and i != 'date'], 
                             var_name='userid', value_name='text'
                             )
                        .drop(columns=['userid'])
                        .assign(country='in')
              )


def clean_nonenglish_tweet(dataframe: pd.DataFrame, 
                           target_col: str
                          ) -> pd.DataFrame:
    '''
    This function is to clean the non-english tweet
    '''
    
    dataframe.target_col = dataframe.target_col.str.lower()
    dataframe = dataframe.dropna(subset=[target_col])
    dataframe['lang'] = dataframe[target_col].apply(lambda x: translator.detect(x).lang)
    dataframe = dataframe[dataframe.lang == 'en']
    
    return dataframe


def sentiment_analysis(dataframe: pd.DataFrame, 
                       target_col: str
                      ) -> pd.DataFrame:
    '''
    This function is to perform the sentiment analysis
    '''
    
    dataframe[target_col] = dataframe[target_col].str.lower() # lower case
    dataframe['polarity'] = dataframe[target_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    dataframe['subjectivity'] = dataframe[target_col].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    return dataframe


df_vacc_us = clean_nonenglish_tweet(df_vacc_us, 'text')
df_vacc_in = clean_nonenglish_tweet(df_vacc_in, 'text')
df_vacc_id = clean_nonenglish_tweet(df_vacc_id, 'translated')


df_vacc_id = sentiment_analysis(df_vacc_id, 'text')
df_vacc_in = sentiment_analysis(df_vacc_in, 'text')
df_vacc_us = sentiment_analysis(df_vacc_us, 'text')

df_tweets = pd.concat([df_vacc_id, df_vacc_in, df_vacc_us], ignore_index=True)

## compute the polarity and subjectivity of each tweet
df_tweets['polarity'] = df_tweets['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df_tweets['subjectivity'] = df_tweets['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

df_tweets.to_csv(os.path.join(path_data, 'tweets.csv'), index=False)
