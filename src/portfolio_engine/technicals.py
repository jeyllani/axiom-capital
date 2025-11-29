import pandas as pd
import numpy as np

def get_technical_indicators(series):
    """
    Calculates technical indicators (RSI, MACD, Stochastic) for a given series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series of prices/values.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['RSI', 'MACD', 'Signal', '%K', '%D']
    """
    df = pd.DataFrame(index=series.index)
    df['Close'] = series
    
    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Stochastic Oscillator (14, 3, 3)
    low_14 = df['Close'].rolling(window=14).min()
    high_14 = df['Close'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    return df[['RSI', 'MACD', 'Signal', '%K', '%D']].dropna()
