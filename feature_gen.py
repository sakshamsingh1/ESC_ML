import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

path = '/Users/sakshamsingh/Desktop/ML_F23/project/ESC-50/meta/esc50.csv'
df = pd.read_csv(path)
cols = ['filename', 'fold', 'target', 'category']
df = df[cols]

def extract_features(file_path):
    y, sr = librosa.load(file_path)

    # Extract features
    features = {
        'mfccs': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'spectral_centroids': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'spectral_contrast': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        'tonnetz': np.mean(librosa.feature.tonnetz(y=y, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'spectral_flatness': np.mean(librosa.feature.spectral_flatness(y=y))
    }
    return features


feature_columns = ['mfccs', 'spectral_centroids', 'chroma_stft', 'spectral_contrast', 'tonnetz',
                   'zero_crossing_rate', 'spectral_rolloff', 'spectral_bandwidth', 'spectral_flatness']
# for feature in feature_columns:
#     df[feature] = None

base_path = '/Users/sakshamsingh/Desktop/ML_F23/project/ESC-50/audio/'
def process_row(row):
    file_path = os.path.join(base_path, row['filename'])
    return extract_features(file_path)

# Run feature extraction in parallel and store results
results = Parallel(n_jobs=-1)(delayed(process_row)(row) for index, row in tqdm(df.iterrows(), total=len(df)))

# Combine results with the original DataFrame
features_df = pd.DataFrame(results)
df = pd.concat([df, features_df], axis=1)
# df = df.dropna(axis=1, how='any')

df.to_csv('data/esc50_features.csv', index=False)
