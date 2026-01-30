"""
Factor Library - Collection of factor calculation functions
"""
import pandas as pd
import numpy as np
from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)


class FactorLibrary:
    """Library of factor calculation functions"""
    
    @staticmethod
    def momentum(df: pd.DataFrame, window: int = 20, price_col: str = 'close') -> pd.Series:
        """
        Calculate momentum factor (price change over window)
        
        Args:
            df: Price DataFrame (must be sorted by date)
            window: Lookback window
            price_col: Price column to use
            
        Returns:
            Series with momentum values
        """
        return df.groupby('stock_code')[price_col].pct_change(window)
    
    @staticmethod
    def volatility(df: pd.DataFrame, window: int = 20, price_col: str = 'close') -> pd.Series:
        """
        Calculate volatility factor (rolling std of returns)
        
        Args:
            df: Price DataFrame
            window: Rolling window
            price_col: Price column to use
            
        Returns:
            Series with volatility values
        """
        returns = df.groupby('stock_code')[price_col].pct_change()
        return returns.groupby(df['stock_code']).rolling(window).std().reset_index(0, drop=True)
    
    @staticmethod
    def rsi(df: pd.DataFrame, window: int = 14, price_col: str = 'close') -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            df: Price DataFrame
            window: RSI window
            price_col: Price column to use
            
        Returns:
            Series with RSI values
        """
        delta = df.groupby('stock_code')[price_col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.groupby(df['stock_code']).rolling(window).mean().reset_index(0, drop=True)
        avg_loss = loss.groupby(df['stock_code']).rolling(window).mean().reset_index(0, drop=True)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def ma_cross(df: pd.DataFrame, short_window: int = 5, long_window: int = 20,
                price_col: str = 'close') -> pd.Series:
        """
        Calculate moving average crossover signal
        
        Args:
            df: Price DataFrame
            short_window: Short MA window
            long_window: Long MA window
            price_col: Price column to use
            
        Returns:
            Series with MA cross signal (1 = bullish, -1 = bearish, 0 = neutral)
        """
        short_ma = df.groupby('stock_code')[price_col].rolling(short_window).mean().reset_index(0, drop=True)
        long_ma = df.groupby('stock_code')[price_col].rolling(long_window).mean().reset_index(0, drop=True)
        
        signal = np.where(short_ma > long_ma, 1, -1)
        return pd.Series(signal, index=df.index)
    
    @staticmethod
    def volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate volume ratio (current volume / average volume)
        
        Args:
            df: Price DataFrame with volume column
            window: Window for average volume
            
        Returns:
            Series with volume ratio
        """
        vol_col = 'vol' if 'vol' in df.columns else 'volume'
        avg_volume = df.groupby('stock_code')[vol_col].rolling(window).mean().reset_index(0, drop=True)
        return df[vol_col] / avg_volume
    
    @staticmethod
    def price_to_high(df: pd.DataFrame, window: int = 20, price_col: str = 'close') -> pd.Series:
        """
        Calculate price distance to recent high
        
        Args:
            df: Price DataFrame
            window: Lookback window
            price_col: Price column to use
            
        Returns:
            Series with price to high ratio
        """
        rolling_high = df.groupby('stock_code')[price_col].rolling(window).max().reset_index(0, drop=True)
        return df[price_col] / rolling_high
    
    @staticmethod
    def price_to_low(df: pd.DataFrame, window: int = 20, price_col: str = 'close') -> pd.Series:
        """
        Calculate price distance to recent low
        
        Args:
            df: Price DataFrame
            window: Lookback window
            price_col: Price column to use
            
        Returns:
            Series with price to low ratio
        """
        rolling_low = df.groupby('stock_code')[price_col].rolling(window).min().reset_index(0, drop=True)
        return df[price_col] / rolling_low
    
    @staticmethod
    def turnover_rate(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate average turnover rate
        
        Args:
            df: Price DataFrame with volume and shares outstanding
            window: Rolling window
            
        Returns:
            Series with turnover rate
        """
        # Placeholder - requires shares outstanding data
        # turnover = volume / shares_outstanding
        if 'turnover' in df.columns:
            return df.groupby('stock_code')['turnover'].rolling(window).mean().reset_index(0, drop=True)
        return pd.Series(np.nan, index=df.index)
    
    @staticmethod
    def beta(df: pd.DataFrame, market_returns: pd.Series, window: int = 60,
            price_col: str = 'close') -> pd.Series:
        """
        Calculate beta (stock vs market)
        
        Args:
            df: Price DataFrame
            market_returns: Market return series
            window: Rolling window
            price_col: Price column to use
            
        Returns:
            Series with beta values
        """
        stock_returns = df.groupby('stock_code')[price_col].pct_change()
        
        # Align market returns with stock data
        df_aligned = df.copy()
        df_aligned['market_return'] = df_aligned['trade_date'].map(market_returns)
        df_aligned['stock_return'] = stock_returns
        
        def rolling_beta(group):
            return group['stock_return'].rolling(window).cov(group['market_return']) / \
                   group['market_return'].rolling(window).var()
        
        return df_aligned.groupby('stock_code').apply(rolling_beta).reset_index(0, drop=True)
    
    @staticmethod
    def rank_normalize(series: pd.Series, by_date: bool = True, 
                      date_col: pd.Series = None) -> pd.Series:
        """
        Rank normalize factor values (cross-sectional)
        
        Args:
            series: Factor values
            by_date: Whether to normalize within each date
            date_col: Date column for grouping
            
        Returns:
            Normalized series (0 to 1)
        """
        if by_date and date_col is not None:
            return series.groupby(date_col).rank(pct=True)
        return series.rank(pct=True)
    
    @staticmethod
    def zscore_normalize(series: pd.Series, by_date: bool = True,
                        date_col: pd.Series = None) -> pd.Series:
        """
        Z-score normalize factor values
        
        Args:
            series: Factor values
            by_date: Whether to normalize within each date
            date_col: Date column for grouping
            
        Returns:
            Z-score normalized series
        """
        if by_date and date_col is not None:
            return series.groupby(date_col).transform(lambda x: (x - x.mean()) / x.std())
        return (series - series.mean()) / series.std()
    
    @staticmethod
    def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99,
                 by_date: bool = True, date_col: pd.Series = None) -> pd.Series:
        """
        Winsorize factor values to handle outliers
        
        Args:
            series: Factor values
            lower: Lower percentile
            upper: Upper percentile
            by_date: Whether to winsorize within each date
            date_col: Date column for grouping
            
        Returns:
            Winsorized series
        """
        def winsorize_group(x):
            lower_bound = x.quantile(lower)
            upper_bound = x.quantile(upper)
            return x.clip(lower_bound, upper_bound)
        
        if by_date and date_col is not None:
            return series.groupby(date_col).transform(winsorize_group)
        return winsorize_group(series)
    
    @classmethod
    def get_all_factors(cls) -> Dict[str, Callable]:
        """
        Get dictionary of all available factor functions
        
        Returns:
            Dictionary mapping factor names to functions
        """
        return {
            'momentum_20': lambda df: cls.momentum(df, 20),
            'momentum_60': lambda df: cls.momentum(df, 60),
            'volatility_20': lambda df: cls.volatility(df, 20),
            'volatility_60': lambda df: cls.volatility(df, 60),
            'rsi_14': lambda df: cls.rsi(df, 14),
            'volume_ratio_20': lambda df: cls.volume_ratio(df, 20),
            'price_to_high_20': lambda df: cls.price_to_high(df, 20),
            'price_to_low_20': lambda df: cls.price_to_low(df, 20),
            'ma_cross_5_20': lambda df: cls.ma_cross(df, 5, 20),
        }
