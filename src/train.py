import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from joblib import dump, load


def main():
    # Read Data
    df = pd.read_csv('data/emotions.csv',
                     names=['emotion', 'text', 'unnamed'])
    df = df[['emotion', 'text']]

    # Map emotions
    emotions = df['emotion'].unique()
    emotion_map = {k:v for v, k in zip(range(len(emotions)), emotions)}
    df['target'] = df['emotion'].map(emotion_map)

    # Train final model

    print('Training final model...')

    x = df['text'].values
    y = df['target'].values

    best_model = Pipeline([('vect', TfidfVectorizer(stop_words="english",
                                                    min_df=1,
                                                    ngram_range=(1, 2))),
                           ('clf', MultinomialNB(alpha=0.75))
                          ])

    best_model.fit(x, y)

    # Save best model
    model_filepath = 'out/final_model.joblib'
    dump(best_model, model_filepath) 

    print('Final model saved to ', model_filepath)

    # Save emotion map
    inverse_emotion_map = {v: k for k, v in emotion_map.items()}

    inv_emotion_map_path = 'out/inv_emotion_map.json'

    with open(inv_emotion_map_path, 'w') as f:
        json.dump(inverse_emotion_map, f)

    print('Inverse emotion map saved to ', inv_emotion_map_path)


if __name__ == '__main__':
    main()
