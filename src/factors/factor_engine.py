"""
Factor Engine - Orchestrates factor calculation and management
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from .factor_library import FactorLibrary
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FactorEngine:
    """Manages factor calculation and processing"""
    
    def __init__(self, db_manager):
        """
        Initialize factor engine
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self.factor_lib = FactorLibrary()
        self.calculated_factors = {}
        
    def calculate_factors(self, df: pd.DataFrame, factor_names: List[str],
                         normalize: bool = True, winsorize: bool = True) -> pd.DataFrame:
        """
        Calculate multiple factors
        
        Args:
            df: Price DataFrame
            factor_names: List of factor names to calculate
            normalize: Whether to normalize factors
            winsorize: Whether to winsorize factors
            
        Returns:
            DataFrame with calculated factors
        """
        df = df.copy()
        df = df.sort_values(['stock_code', 'trade_date'])
        
        available_factors = self.factor_lib.get_all_factors()
        
        logger.info(f"Calculating {len(factor_names)} factors...")
        
        for factor_name in tqdm(factor_names, desc="Calculating factors"):
            if factor_name in available_factors:
                try:
                    factor_values = available_factors[factor_name](df)
                    df[factor_name] = factor_values
                    
                    # Winsorize
                    if winsorize:
                        df[factor_name] = self.factor_lib.winsorize(
                            df[factor_name], 
                            by_date=True, 
                            date_col=df['trade_date']
                        )
                    
                    # Normalize
                    if normalize:
                        df[factor_name] = self.factor_lib.zscore_normalize(
                            df[factor_name],
                            by_date=True,
                            date_col=df['trade_date']
                        )
                    
                    logger.debug(f"Calculated factor: {factor_name}")
                except Exception as e:
                    logger.error(f"Error calculating {factor_name}: {e}")
            else:
                logger.warning(f"Factor {factor_name} not found in library")
        
        self.calculated_factors = df
        return df
    
    def add_custom_factor(self, name: str, func: Callable, df: pd.DataFrame,
                         normalize: bool = True, winsorize: bool = True) -> pd.DataFrame:
        """
        Add custom factor calculation
        
        Args:
            name: Factor name
            func: Function that takes DataFrame and returns Series
            df: Price DataFrame
            normalize: Whether to normalize
            winsorize: Whether to winsorize
            
        Returns:
            DataFrame with new factor added
        """
        df = df.copy()
        
        try:
            factor_values = func(df)
            df[name] = factor_values
            
            if winsorize:
                df[name] = self.factor_lib.winsorize(
                    df[name],
                    by_date=True,
                    date_col=df['trade_date']
                )
            
            if normalize:
                df[name] = self.factor_lib.zscore_normalize(
                    df[name],
                    by_date=True,
                    date_col=df['trade_date']
                )
            
            logger.info(f"Added custom factor: {name}")
        except Exception as e:
            logger.error(f"Error adding custom factor {name}: {e}")
        
        return df
    
    def combine_factors(self, df: pd.DataFrame, factor_names: List[str],
                       weights: Optional[List[float]] = None) -> pd.Series:
        """
        Combine multiple factors into composite score
        
        Args:
            df: DataFrame with calculated factors
            factor_names: List of factor names to combine
            weights: Optional weights for each factor (default: equal weight)
            
        Returns:
            Series with composite factor scores
        """
        if weights is None:
            weights = [1.0 / len(factor_names)] * len(factor_names)
        
        if len(weights) != len(factor_names):
            raise ValueError("Number of weights must match number of factors")
        
        composite = pd.Series(0, index=df.index)
        
        for factor_name, weight in zip(factor_names, weights):
            if factor_name in df.columns:
                composite += df[factor_name].fillna(0) * weight
            else:
                logger.warning(f"Factor {factor_name} not found in DataFrame")
        
        return composite
    
    def save_factors(self, df: pd.DataFrame, factor_names: List[str]):
        """
        Save calculated factors to database
        
        Args:
            df: DataFrame with calculated factors
            factor_names: List of factor names to save
        """
        records = []
        
        for factor_name in factor_names:
            if factor_name not in df.columns:
                logger.warning(f"Factor {factor_name} not found, skipping")
                continue
            
            factor_df = df[['stock_code', 'trade_date', factor_name]].copy()
            factor_df.columns = ['stock_code', 'trade_date', 'factor_value']
            factor_df['factor_name'] = factor_name
            
            records.append(factor_df)
        
        if records:
            all_factors = pd.concat(records, ignore_index=True)
            all_factors = all_factors.dropna(subset=['factor_value'])
            self.db.save_factor_data(all_factors)
            logger.info(f"Saved {len(factor_names)} factors to database")
    
    def load_factors(self, factor_names: List[str], stock_codes: Optional[List[str]] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load factors from database
        
        Args:
            factor_names: List of factor names to load
            stock_codes: Optional list of stock codes
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with factors in wide format
        """
        df = self.db.load_factor_data(factor_names, stock_codes, start_date, end_date)
        
        if df.empty:
            logger.warning("No factor data loaded")
            return df
        
        # Pivot to wide format
        pivot_df = df.pivot_table(
            index=['stock_code', 'trade_date'],
            columns='factor_name',
            values='factor_value'
        ).reset_index()
        
        return pivot_df
    
    def analyze_factor_correlation(self, df: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """
        Analyze correlation between factors
        
        Args:
            df: DataFrame with calculated factors
            factor_names: List of factor names to analyze
            
        Returns:
            Correlation matrix
        """
        factor_data = df[factor_names].copy()
        corr_matrix = factor_data.corr()
        
        logger.info("Factor correlation matrix:")
        logger.info(f"\n{corr_matrix}")
        
        return corr_matrix
    
    def get_factor_coverage(self, df: pd.DataFrame, factor_names: List[str]) -> Dict[str, float]:
        """
        Calculate factor coverage (non-null percentage)
        
        Args:
            df: DataFrame with calculated factors
            factor_names: List of factor names
            
        Returns:
            Dictionary of factor coverage percentages
        """
        coverage = {}
        
        for factor_name in factor_names:
            if factor_name in df.columns:
                coverage[factor_name] = (1 - df[factor_name].isna().mean()) * 100
            else:
                coverage[factor_name] = 0.0
        
        logger.info("Factor coverage:")
        for name, cov in coverage.items():
            logger.info(f"  {name}: {cov:.2f}%")
        
        return coverage
