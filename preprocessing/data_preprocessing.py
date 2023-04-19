import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
import argparse
from tqdm import tqdm
import pickle 
tqdm.pandas()

# nltk.download('stopwords')

def preprocess(file_path,
                fillna = 'ukw',
                remove_punc = True,
                lower_case = True,
                remove_stopword = True,
                steam = False,
            ):
    
    print('reading csv...')
    df = pd.read_csv(file_path)
    df.fillna(fillna, inplace=True)
    X = df['title'] + ' eoft ' + df['text']
    y = df['label']

    print('remove punc...')
    X = X.progress_apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))  if remove_punc else X

    print('lower case...')
    X = X.progress_apply(lambda x: x.lower()) if lower_case else X

    #tokenizing
    print('tokenizing...')
    X = X.progress_apply(lambda x: nltk.word_tokenize(x)) 

    print('remove stopwords...')
    X = X.progress_apply(lambda x: [word for word in x if word not in stopwords.words('english')]) if remove_stopword else X

    print('steaming...')
    ps = nltk.PorterStemmer()
    X = X.progress_apply(lambda x: [ps.stem(word) for word in X]) if steam else X

    return X.values, y.values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()
    file_name = args.file_name

    X, y = preprocess(file_name)

    with open('X_train.data', 'wb') as f_train:
        pickle.dump(X, f_train)

    with open('y_train.data', 'rb') as fy_train:
        pickle.dump(y, fy_train)