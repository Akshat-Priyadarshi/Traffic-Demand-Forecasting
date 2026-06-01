import pandas as pd
import numpy as np
import pygeohash as pgh

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw traffic data into an advanced spatial-temporal matrix.
    """
    df = df.copy()
    
    # 1. The Keystone Spatial Features: Decoding Geohashes
    # We convert the string into exact physical coordinates.
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) and x != 'Missing' else np.nan)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) and x != 'Missing' else np.nan)
    
    # 2. Cyclical Temporal Features
    # Traffic operates in 24-hour cycles. Sin/Cos transformations ensure 
    # the model understands that 23:55 is only 10 minutes away from 00:05.
    df['hour'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else 0)
    df['minute'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]) if pd.notnull(x) else 0)
    df['time_in_mins'] = df['hour'] * 60 + df['minute']
    
    # 1440 minutes in a day
    df['time_sin'] = np.sin(2 * np.pi * df['time_in_mins'] / 1440)
    df['time_cos'] = np.cos(2 * np.pi * df['time_in_mins'] / 1440)
    
    # 3. Binary and Categorical Clean-up
    df['LargeVehicles'] = df['LargeVehicles'].map({'Not Allowed': 0, 'Allowed': 1})
    df['Landmarks'] = df['Landmarks'].map({'No': 0, 'Yes': 1})
    
    # Fill remaining categoricals for Tree Models
    cat_cols = ['RoadType', 'Weather']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Missing')
        
    # Drop raw timestamp as it is now mathematically represented
    df = df.drop(columns=['timestamp', 'hour', 'minute'])
    
    return df