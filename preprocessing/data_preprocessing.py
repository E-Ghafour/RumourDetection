import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re


def preprocess(file_path,
                fillna = 'ukw',
                remove_punc = True,
                lower_case = True,
                steam = True,
                remove_stopword = True,

            ):
    
    df = pd.read_csv(file_path)
    df.fillna(fillna, inplace=True)
    df['content'] = df['title'] + ' __eoft__ ' + df['text']
    

