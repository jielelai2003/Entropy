"""
Performance Analyzer - Calculates performance metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyzes backtest performance"""
    
    @staticmethod
    def calculate_returns(portfolio_values: pd.Series) -> pd.Series:
        """Calculate daily returns"""
        return portfolio_values.pct_change().fillna(0)
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_total_return(portfolio_values: pd.Series) -> float:
        """Calculate total return"""
        if len(portfolio_values) == 0:
            return 0.0
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    
    @staticmethod
    def calculate_annual_return(portfolio_values: pd.Series, trading_days: int = 252) -> float:
        """Calculate annualized return"""
        total_return = PerformanceAnalyzer.calculate_total_return(portfolio_values)
        n_days = len(portfolio_values)
        if n_days == 0:
            return 0.0
        years = n_days / trading_days
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(trading_days)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03,
                              trading_days: int = 252) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / trading_days
        if excess_returns.std() == 0:
            return 0.0
        return np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03,
                               trading_days: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / trading_days
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(trading_days) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = portfolio_values
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
    
    @staticmethod
    def calculate_profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        return gains / losses
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                                   trading_days: int = 252) -> float:
        """Calculate information ratio"""
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(trading_days)
        
        if tracking_error == 0:
            return 0.0
        
        return (active_returns.mean() * trading_days) / tracking_error
    
    @staticmethod
    def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta"""
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 0.0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(returns: pd.Series, market_returns: pd.Series,
                       risk_free_rate: float = 0.03, trading_days: int = 252) -> float:
        """Calculate alpha"""
        beta = PerformanceAnalyzer.calculate_beta(returns, market_returns)
        
        portfolio_return = returns.mean() * trading_days
        market_return = market_returns.mean() * trading_days
        
        alpha = portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
        return alpha
    
    @classmethod
    def analyze(cls, portfolio_df: pd.DataFrame, 
                benchmark_returns: Optional[pd.Series] = None,
                risk_free_rate: float = 0.03) -> Dict[str, float]:
        """
        Comprehensive performance analysis
        
        Args:
            portfolio_df: DataFrame with portfolio history
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        if portfolio_df.empty:
            logger.warning("Empty portfolio data")
            return {}
        
        portfolio_values = portfolio_df['portfolio_value']
        returns = cls.calculate_returns(portfolio_values)
        
        metrics = {
            'total_return': cls.calculate_total_return(portfolio_values),
            'annual_return': cls.calculate_annual_return(portfolio_values),
            'volatility': cls.calculate_volatility(returns),
            'sharpe_ratio': cls.calculate_sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': cls.calculate_sortino_ratio(returns, risk_free_rate),
            'max_drawdown': cls.calculate_max_drawdown(portfolio_values),
            'win_rate': cls.calculate_win_rate(returns),
            'profit_factor': cls.calculate_profit_factor(returns),
        }
        
        # Calmar ratio
        metrics['calmar_ratio'] = cls.calculate_calmar_ratio(
            metrics['annual_return'],
            metrics['max_drawdown']
        )
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
            metrics['beta'] = cls.calculate_beta(returns, aligned_benchmark)
            metrics['alpha'] = cls.calculate_alpha(returns, aligned_benchmark, risk_free_rate)
            metrics['information_ratio'] = cls.calculate_information_ratio(returns, aligned_benchmark)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Print performance metrics in formatted way"""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        
        print(f"\nReturn Metrics:")
        print(f"  Total Return:        {metrics.get('total_return', 0)*100:>8.2f}%")
        print(f"  Annual Return:       {metrics.get('annual_return', 0)*100:>8.2f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Volatility:          {metrics.get('volatility', 0)*100:>8.2f}%")
        print(f"  Max Drawdown:        {metrics.get('max_drawdown', 0)*100:>8.2f}%")
        
        print(f"\nRisk-Adjusted Returns:")
        print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):>8.2f}")
        print(f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):>8.2f}")
        
        print(f"\nTrading Metrics:")
        print(f"  Win Rate:            {metrics.get('win_rate', 0)*100:>8.2f}%")
        print(f"  Profit Factor:       {metrics.get('profit_factor', 0):>8.2f}")
        
        if 'beta' in metrics:
            print(f"\nBenchmark-Relative:")
            print(f"  Beta:                {metrics.get('beta', 0):>8.2f}")
            print(f"  Alpha:               {metrics.get('alpha', 0)*100:>8.2f}%")
            print(f"  Information Ratio:   {metrics.get('information_ratio', 0):>8.2f}")
        
        print("="*50 + "\n")
