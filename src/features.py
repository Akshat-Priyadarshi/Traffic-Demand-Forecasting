import pandas as pd
import numpy as np
import pygeohash as pgh

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Exact Spatial Coordinates
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) and x != 'Missing' else np.nan)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) and x != 'Missing' else np.nan)
    
    # 2. Discrete Temporal Features (Trees love raw integers, not sin/cos)
    df['hour'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else 0)
    df['minute'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]) if pd.notnull(x) else 0)
    df['time_in_mins'] = df['hour'] * 60 + df['minute']
    
    # 3. Binary and Categorical Clean-up
    df['LargeVehicles'] = df['LargeVehicles'].map({'Not Allowed': 0, 'Allowed': 1})
    df['Landmarks'] = df['Landmarks'].map({'No': 0, 'Yes': 1})
    
    cat_cols = ['RoadType', 'Weather']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Missing')
        
    df = df.drop(columns=['timestamp'])
    
    return df