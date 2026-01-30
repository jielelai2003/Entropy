"""
Backtest Engine - Core backtesting logic
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from .portfolio import Portfolio
from .performance import PerformanceAnalyzer
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 1000000.0,
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.0001):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Initial capital
            commission_rate: Commission rate
            slippage_rate: Slippage rate
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        self.portfolio = None
        self.results = None
        
    def run_backtest(self, data: pd.DataFrame, 
                    signal_column: str,
                    top_n: int = 50,
                    rebalance_freq: str = '1W',
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run backtest with factor signals
        
        Args:
            data: DataFrame with price data and factor signals
            signal_column: Column name containing factor signals
            top_n: Number of top stocks to hold
            rebalance_freq: Rebalancing frequency ('1D', '1W', '1M')
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate
        )
        
        # Filter date range
        df = data.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        if start_date:
            df = df[df['trade_date'] >= start_date]
        if end_date:
            df = df[df['trade_date'] <= end_date]
        
        if df.empty:
            logger.error("No data in specified date range")
            return {}
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(df, rebalance_freq)
        
        logger.info(f"Running backtest from {df['trade_date'].min()} to {df['trade_date'].max()}")
        logger.info(f"Rebalancing {len(rebalance_dates)} times")
        
        # Run backtest
        for date in tqdm(rebalance_dates, desc="Backtesting"):
            # Get data for this date
            date_data = df[df['trade_date'] == date].copy()
            
            if date_data.empty:
                continue
            
            # Rank stocks by signal
            date_data = date_data.dropna(subset=[signal_column])
            date_data = date_data.sort_values(signal_column, ascending=False)
            
            # Select top N stocks
            selected_stocks = date_data.head(top_n)
            
            # Equal weight portfolio
            target_weights = {
                row['stock_code']: 1.0 / top_n
                for _, row in selected_stocks.iterrows()
            }
            
            # Get prices
            prices = {
                row['stock_code']: row['close']
                for _, row in selected_stocks.iterrows()
            }
            
            # Rebalance portfolio
            self.portfolio.rebalance(target_weights, prices, str(date))
            
            # Record state
            self.portfolio.record_state(str(date), prices)
        
        # Get results
        portfolio_df = self.portfolio.get_history_df()
        trades_df = self.portfolio.get_trades_df()
        
        # Calculate performance
        metrics = PerformanceAnalyzer.analyze(portfolio_df)
        
        self.results = {
            'portfolio_history': portfolio_df,
            'trades': trades_df,
            'metrics': metrics
        }
        
        logger.info("Backtest completed")
        return self.results
    
    def run_custom_strategy(self, data: pd.DataFrame,
                           strategy_func: Callable,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict:
        """
        Run backtest with custom strategy function
        
        Args:
            data: DataFrame with price data
            strategy_func: Function that takes (date, data, portfolio) and returns target_weights
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate
        )
        
        # Filter date range
        df = data.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        if start_date:
            df = df[df['trade_date'] >= start_date]
        if end_date:
            df = df[df['trade_date'] <= end_date]
        
        # Get unique dates
        dates = sorted(df['trade_date'].unique())
        
        logger.info(f"Running custom strategy backtest")
        
        for date in tqdm(dates, desc="Backtesting"):
            # Get data up to this date
            historical_data = df[df['trade_date'] <= date]
            current_data = df[df['trade_date'] == date]
            
            if current_data.empty:
                continue
            
            # Get target weights from strategy
            target_weights = strategy_func(date, historical_data, self.portfolio)
            
            if not target_weights:
                continue
            
            # Get prices
            prices = {
                row['stock_code']: row['close']
                for _, row in current_data.iterrows()
                if row['stock_code'] in target_weights
            }
            
            # Rebalance
            self.portfolio.rebalance(target_weights, prices, str(date))
            
            # Record state
            all_prices = {
                row['stock_code']: row['close']
                for _, row in current_data.iterrows()
            }
            self.portfolio.record_state(str(date), all_prices)
        
        # Get results
        portfolio_df = self.portfolio.get_history_df()
        trades_df = self.portfolio.get_trades_df()
        metrics = PerformanceAnalyzer.analyze(portfolio_df)
        
        self.results = {
            'portfolio_history': portfolio_df,
            'trades': trades_df,
            'metrics': metrics
        }
        
        return self.results
    
    def _get_rebalance_dates(self, df: pd.DataFrame, freq: str) -> List:
        """Get rebalancing dates based on frequency"""
        dates = pd.to_datetime(df['trade_date'].unique())
        dates = pd.Series(dates).sort_values()
        
        if freq == '1D':
            return dates.tolist()
        elif freq == '1W':
            # Weekly rebalance (every Monday or first trading day of week)
            rebalance = dates.to_frame(name='date')
            rebalance['week'] = rebalance['date'].dt.isocalendar().week
            rebalance['year'] = rebalance['date'].dt.year
            return rebalance.groupby(['year', 'week'])['date'].first().tolist()
        elif freq == '1M':
            # Monthly rebalance (first trading day of month)
            rebalance = dates.to_frame(name='date')
            rebalance['month'] = rebalance['date'].dt.month
            rebalance['year'] = rebalance['date'].dt.year
            return rebalance.groupby(['year', 'month'])['date'].first().tolist()
        else:
            logger.warning(f"Unknown frequency {freq}, using daily")
            return dates.tolist()
    
    def get_results(self) -> Dict:
        """Get backtest results"""
        return self.results
    
    def print_summary(self):
        """Print backtest summary"""
        if not self.results:
            logger.warning("No results to display")
            return
        
        metrics = self.results['metrics']
        PerformanceAnalyzer.print_metrics(metrics)
        
        trades_df = self.results['trades']
        if not trades_df.empty:
            print(f"Total Trades: {len(trades_df)}")
            print(f"Buy Trades:   {len(trades_df[trades_df['action'] == 'BUY'])}")
            print(f"Sell Trades:  {len(trades_df[trades_df['action'] == 'SELL'])}")
