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
# nltk.download('punkt')

def preprocess(file_path,
                fillna = 'ukw',
                remove_punc = True,
                lower_case = True,
                remove_stopwords = True,
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
    X = X.progress_apply(lambda x: [word for word in x if word not in stopwords.words('english')]) if remove_stopwords else X

    print('steaming...')
    ps = nltk.PorterStemmer()
    X = X.progress_apply(lambda x: [ps.stem(word) for word in X]) if steam else X

    return X.values, y.values

def save_data(data, path):
    with open(path, 'wb') as ff:
        pickle.dump(data, ff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()
    file_name = args.file_name

    X, y = preprocess(file_name)

    save_data(X)

    save_data(y)
