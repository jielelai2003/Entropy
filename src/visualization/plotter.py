"""
Plotter - Visualization utilities for backtest results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Plotter:
    """Visualization tools for backtest analysis"""
    
    @staticmethod
    def plot_portfolio_value(portfolio_df: pd.DataFrame, 
                            benchmark_df: Optional[pd.DataFrame] = None,
                            save_path: Optional[str] = None):
        """
        Plot portfolio value over time
        
        Args:
            portfolio_df: Portfolio history DataFrame
            benchmark_df: Optional benchmark DataFrame
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot portfolio value
        ax.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 
                label='Portfolio', linewidth=2, color='#2E86AB')
        
        # Plot benchmark if provided
        if benchmark_df is not None:
            ax.plot(benchmark_df['date'], benchmark_df['value'], 
                   label='Benchmark', linewidth=2, color='#A23B72', alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved portfolio value plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_returns(portfolio_df: pd.DataFrame,
                    benchmark_df: Optional[pd.DataFrame] = None,
                    save_path: Optional[str] = None):
        """
        Plot cumulative returns
        
        Args:
            portfolio_df: Portfolio history DataFrame
            benchmark_df: Optional benchmark DataFrame
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Calculate returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().fillna(0)
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Plot portfolio returns
        ax.plot(portfolio_df['date'], cumulative_returns * 100,
               label='Portfolio', linewidth=2, color='#2E86AB')
        
        # Plot benchmark if provided
        if benchmark_df is not None:
            benchmark_returns = benchmark_df['value'].pct_change().fillna(0)
            benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1
            ax.plot(benchmark_df['date'], benchmark_cumulative * 100,
                   label='Benchmark', linewidth=2, color='#A23B72', alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Returns (%)', fontsize=12)
        ax.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved returns plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_drawdown(portfolio_df: pd.DataFrame,
                     save_path: Optional[str] = None):
        """
        Plot drawdown over time
        
        Args:
            portfolio_df: Portfolio history DataFrame
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Calculate drawdown
        portfolio_value = portfolio_df['portfolio_value']
        running_max = portfolio_value.expanding().max()
        drawdown = (portfolio_value - running_max) / running_max * 100
        
        # Plot drawdown
        ax.fill_between(portfolio_df['date'], drawdown, 0, 
                        color='#C73E1D', alpha=0.3)
        ax.plot(portfolio_df['date'], drawdown, 
               color='#C73E1D', linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved drawdown plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_rolling_metrics(portfolio_df: pd.DataFrame,
                            window: int = 60,
                            save_path: Optional[str] = None):
        """
        Plot rolling Sharpe ratio and volatility
        
        Args:
            portfolio_df: Portfolio history DataFrame
            window: Rolling window size
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Calculate returns
        returns = portfolio_df['portfolio_value'].pct_change().fillna(0)
        
        # Rolling Sharpe ratio
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
        ax1.plot(portfolio_df['date'], rolling_sharpe, 
                color='#2E86AB', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Sharpe Ratio', fontsize=12)
        ax1.set_title(f'Rolling Sharpe Ratio ({window}-day)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        ax2.plot(portfolio_df['date'], rolling_vol,
                color='#F18F01', linewidth=2)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volatility (%)', fontsize=12)
        ax2.set_title(f'Rolling Volatility ({window}-day)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved rolling metrics plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_monthly_returns(portfolio_df: pd.DataFrame,
                            save_path: Optional[str] = None):
        """
        Plot monthly returns heatmap
        
        Args:
            portfolio_df: Portfolio history DataFrame
            save_path: Path to save figure
        """
        # Calculate monthly returns
        df = portfolio_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        monthly_returns = df['portfolio_value'].resample('M').last().pct_change() * 100
        
        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='month', columns='year', values='return')
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Month', fontsize=12)
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved monthly returns heatmap to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_trade_analysis(trades_df: pd.DataFrame,
                           save_path: Optional[str] = None):
        """
        Plot trade analysis
        
        Args:
            trades_df: Trades DataFrame
            save_path: Path to save figure
        """
        if trades_df.empty:
            logger.warning("No trades to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Trade count over time
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_by_date = trades_df.groupby('date').size()
        ax1.plot(trades_by_date.index, trades_by_date.values, 
                color='#2E86AB', linewidth=2)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Number of Trades', fontsize=11)
        ax1.set_title('Trade Count Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Buy vs Sell
        action_counts = trades_df['action'].value_counts()
        ax2.bar(action_counts.index, action_counts.values, 
               color=['#06A77D', '#C73E1D'])
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Buy vs Sell Trades', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Commission costs over time
        commission_by_date = trades_df.groupby('date')['commission'].sum()
        ax3.plot(commission_by_date.index, commission_by_date.values,
                color='#F18F01', linewidth=2)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Commission Cost', fontsize=11)
        ax3.set_title('Commission Costs Over Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Top traded stocks
        top_stocks = trades_df['stock_code'].value_counts().head(10)
        ax4.barh(range(len(top_stocks)), top_stocks.values, color='#A23B72')
        ax4.set_yticks(range(len(top_stocks)))
        ax4.set_yticklabels(top_stocks.index)
        ax4.set_xlabel('Trade Count', fontsize=11)
        ax4.set_title('Top 10 Traded Stocks', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trade analysis plot to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_factor_distribution(factor_data: pd.DataFrame,
                                 factor_name: str,
                                 save_path: Optional[str] = None):
        """
        Plot factor distribution
        
        Args:
            factor_data: DataFrame with factor values
            factor_name: Name of factor column
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        factor_values = factor_data[factor_name].dropna()
        ax1.hist(factor_values, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.set_xlabel(factor_name, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'{factor_name} Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        ax2.boxplot(factor_values, vert=True)
        ax2.set_ylabel(factor_name, fontsize=12)
        ax2.set_title(f'{factor_name} Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved factor distribution plot to {save_path}")
        
        plt.show()
    
    @classmethod
    def plot_all(cls, results: Dict, output_dir: Optional[str] = None):
        """
        Generate all plots
        
        Args:
            results: Backtest results dictionary
            output_dir: Directory to save plots
        """
        portfolio_df = results.get('portfolio_history')
        trades_df = results.get('trades')
        
        if portfolio_df is None or portfolio_df.empty:
            logger.warning("No portfolio data to plot")
            return
        
        # Portfolio value
        save_path = f"{output_dir}/portfolio_value.png" if output_dir else None
        cls.plot_portfolio_value(portfolio_df, save_path=save_path)
        
        # Returns
        save_path = f"{output_dir}/returns.png" if output_dir else None
        cls.plot_returns(portfolio_df, save_path=save_path)
        
        # Drawdown
        save_path = f"{output_dir}/drawdown.png" if output_dir else None
        cls.plot_drawdown(portfolio_df, save_path=save_path)
        
        # Rolling metrics
        save_path = f"{output_dir}/rolling_metrics.png" if output_dir else None
        cls.plot_rolling_metrics(portfolio_df, save_path=save_path)
        
        # Monthly returns
        save_path = f"{output_dir}/monthly_returns.png" if output_dir else None
        cls.plot_monthly_returns(portfolio_df, save_path=save_path)
        
        # Trade analysis
        if trades_df is not None and not trades_df.empty:
            save_path = f"{output_dir}/trade_analysis.png" if output_dir else None
            cls.plot_trade_analysis(trades_df, save_path=save_path)
        
        logger.info("All plots generated")
