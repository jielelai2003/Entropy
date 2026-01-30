"""
Data Loader - Handles data loading and preprocessing
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses market data"""
    
    def __init__(self, db_manager):
        """
        Initialize data loader
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        
    def load_price_data(self, stock_codes: Optional[List[str]] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       table_name: str = 'stock_daily_price') -> pd.DataFrame:
        """
        Load price data from database
        
        Args:
            stock_codes: List of stock codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            table_name: Name of the table to load from
            
        Returns:
            DataFrame with price data
        """
        df = self.db.load_stock_prices(stock_codes, start_date, end_date, table_name=table_name)
        
        if df.empty:
            logger.warning("No price data loaded")
            return df
            
        # Calculate adjusted prices
        if 'adj_factor' in df.columns:
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[f'adj_{col}'] = df[col] * df['adj_factor']
        
        logger.info(f"Loaded price data: {len(df)} records, {df['stock_code'].nunique()} stocks")
        return df
        
    def get_returns(self, df: pd.DataFrame, periods: int = 1, 
                   price_col: str = 'adj_close') -> pd.DataFrame:
        """
        Calculate returns for each stock
        
        Args:
            df: Price DataFrame
            periods: Number of periods for return calculation
            price_col: Price column to use
            
        Returns:
            DataFrame with returns column added
        """
        df = df.copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        df[f'return_{periods}d'] = df.groupby('stock_code')[price_col].pct_change(periods)
        return df
        
    def pivot_data(self, df: pd.DataFrame, value_col: str = 'adj_close') -> pd.DataFrame:
        """
        Pivot data to wide format (dates x stocks)
        
        Args:
            df: Long format DataFrame
            value_col: Column to pivot
            
        Returns:
            Wide format DataFrame
        """
        pivot_df = df.pivot(index='trade_date', columns='stock_code', values=value_col)
        return pivot_df
        
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        Get list of trading dates
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading dates
        """
        df = self.db.load_stock_prices(start_date=start_date, end_date=end_date)
        if df.empty:
            return []
        dates = sorted(df['trade_date'].unique())
        return [d.strftime('%Y-%m-%d') for d in dates]
        
    def filter_universe(self, df: pd.DataFrame, universe: str = 'all',
                       min_price: float = 1.0, min_volume: float = 0) -> pd.DataFrame:
        """
        Filter stock universe based on criteria
        
        Args:
            df: Price DataFrame
            universe: Universe type (all, hs300, zz500, custom)
            min_price: Minimum price filter
            min_volume: Minimum volume filter
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        # Price filter
        if 'adj_close' in df.columns:
            df = df[df['adj_close'] >= min_price]
        
        # Volume filter
        if 'volume' in df.columns and min_volume > 0:
            df = df[df['volume'] >= min_volume]
        
        # Universe filter (placeholder - implement based on your index constituents)
        if universe == 'hs300':
            # Filter for HS300 constituents
            pass
        elif universe == 'zz500':
            # Filter for ZZ500 constituents
            pass
            
        logger.info(f"Filtered universe: {df['stock_code'].nunique()} stocks")
        return df
        
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing data
        
        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing data (ffill, bfill, drop)
            
        Returns:
            DataFrame with missing data handled
        """
        df = df.copy()
        
        if method == 'ffill':
            df = df.groupby('stock_code').fillna(method='ffill')
        elif method == 'bfill':
            df = df.groupby('stock_code').fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna()
            
        return df
        
    def resample_data(self, df: pd.DataFrame, frequency: str = 'W') -> pd.DataFrame:
        """
        Resample data to different frequency
        
        Args:
            df: Daily price DataFrame
            frequency: Target frequency (W=weekly, M=monthly)
            
        Returns:
            Resampled DataFrame
        """
        df = df.copy()
        df = df.set_index('trade_date')
        
        resampled_list = []
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code]
            
            resampled = stock_df.resample(frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
            
            if 'adj_close' in stock_df.columns:
                resampled['adj_close'] = stock_df['adj_close'].resample(frequency).last()
                
            resampled['stock_code'] = stock_code
            resampled_list.append(resampled)
            
        result = pd.concat(resampled_list).reset_index()
        logger.info(f"Resampled to {frequency}: {len(result)} records")
        return result
        
    def align_data(self, *dfs: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Align multiple DataFrames to same dates and stocks
        
        Args:
            *dfs: Variable number of DataFrames to align
            
        Returns:
            List of aligned DataFrames
        """
        if not dfs:
            return []
            
        # Get common dates and stocks
        common_dates = set(dfs[0]['trade_date'].unique())
        common_stocks = set(dfs[0]['stock_code'].unique())
        
        for df in dfs[1:]:
            common_dates &= set(df['trade_date'].unique())
            common_stocks &= set(df['stock_code'].unique())
            
        # Filter all DataFrames
        aligned = []
        for df in dfs:
            filtered = df[
                df['trade_date'].isin(common_dates) & 
                df['stock_code'].isin(common_stocks)
            ].copy()
            aligned.append(filtered)
            
        logger.info(f"Aligned data: {len(common_dates)} dates, {len(common_stocks)} stocks")
        return aligned
