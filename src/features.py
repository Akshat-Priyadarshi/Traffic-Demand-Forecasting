import pandas as pd
import numpy as np
import pygeohash as pgh

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Exact Spatial Coordinates
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) and x != 'Missing' else np.nan)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) and x != 'Missing' else np.nan)
    
    # 2. Cyclical Temporal Features (Restored for seamless continuous time)
    df['hour'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else 0)
    df['minute'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]) if pd.notnull(x) else 0)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)
    
    # 3. Binary and Categorical Clean-up
    df['LargeVehicles'] = df['LargeVehicles'].map({'Not Allowed': 0, 'Allowed': 1})
    df['Landmarks'] = df['Landmarks'].map({'No': 0, 'Yes': 1})
    
    for col in ['RoadType', 'Weather']:
        df[col] = df[col].astype(str).fillna('Missing')
        
    df = df.drop(columns=['timestamp'])
    
    return df