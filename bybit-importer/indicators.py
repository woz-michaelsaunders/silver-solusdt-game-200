def calculate_rsi(df, window=14):
    """Calculate Relative Strength Index (RSI) for the given data."""
    delta = df['close'].diff()  # Price change between consecutive periods
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # Average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Average loss
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    #print(rsi)
    return rsi

def calculate_ichimoku(data):
    """
    Calculate Ichimoku Cloud components.
    :param data: DataFrame containing 'High', 'Low', and 'Close' columns.
    :return: DataFrame with Ichimoku components added.
    """
    # Tenkan-sen (Conversion Line)
    data['Tenkan_sen'] = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2

    # Kijun-sen (Base Line)
    data['Kijun_sen'] = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2

    # Senkou Span A (Leading Span A)
    data['Senkou_Span_A'] = ((data['Tenkan_sen'] + data['Kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    data['Senkou_Span_B'] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)

    # Chikou Span (Lagging Span)
    data['Chikou_Span'] = data['close'].shift(-26)
    return data

def calculate_bollinger_bands(data, window=20, multiplier=2):
    """
    Calculate Bollinger Bands for the given data.

    Args:
        data (pd.DataFrame): DataFrame with a 'Close' column.
        window (int): Window size for the moving average and standard deviation.
        multiplier (float): Multiplier for the standard deviation.

    Returns:
        pd.DataFrame: DataFrame with 'Middle Band', 'Upper Band', and 'Lower Band'.
    """
    # Calculate Middle Band (Simple Moving Average)
    data['Middle Band'] = data['close'].rolling(window=window).mean()

    # Calculate Standard Deviation
    data['Standard Deviation'] = data['close'].rolling(window=window).std()

    # Calculate Upper and Lower Bands
    data['Upper Band'] = data['Middle Band'] + (multiplier * data['Standard Deviation'])
    data['Lower Band'] = data['Middle Band'] - (multiplier * data['Standard Deviation'])

    # Drop NaN values caused by rolling calculations
    #data = data.dropna()
    return data

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    # Compute EMAs
    data['EMA_fast'] = data['close'].ewm(span=fast_period, adjust=False).mean()
    data['EMA_slow'] = data['close'].ewm(span=slow_period, adjust=False).mean()
    # MACD Line
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    # Signal Line
    data['Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    # MACD Histogram
    data['Histogram'] = data['MACD'] - data['Signal']
    return data

