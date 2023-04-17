import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re


def preprocess(file_path,
                fillna = 'ukw',
                remove_punc = True,
                lower_case = True,
                remove_stopword = True,
                steam = False,

            ):
    
    df = pd.read_csv(file_path)
    df.fillna(fillna, inplace=True)
    X = df['title'] + ' eoft ' + df['text']
    y = df['label']


    X = X.apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))  if remove_punc else X
    X = X.apply(lambda x: x.lower()) if lower_case else X

    #tokenizing
    X = X.apply(lambda x: nltk.word_tokenize(x)) 


    X = X.apply(lambda x: [word for word in x if word not in stopwords.words('english')]) if remove_stopword else X

    X = X.apply(lambda x: [])



