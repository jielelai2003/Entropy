"""
Example: Multi-Factor Backtesting Framework Usage

This example demonstrates how to use the Entropy framework for:
1. Loading data from database
2. Calculating factors
3. Running backtest
4. Analyzing and visualizing results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging
from src.database.db_manager import DatabaseManager
from src.data.data_loader import DataLoader
from src.factors.factor_engine import FactorEngine
from src.backtest.backtest_engine import BacktestEngine
from src.visualization.plotter import Plotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example workflow"""
    
    # ========== 1. Initialize Components ==========
    logger.info("Initializing components...")
    
    db_manager = DatabaseManager('config.yaml')
    data_loader = DataLoader(db_manager)
    factor_engine = FactorEngine(db_manager)
    backtest_engine = BacktestEngine(
        initial_capital=1000000,
        commission_rate=0.0003,
        slippage_rate=0.0001
    )
    
    # ========== 2. Load Data ==========
    logger.info("Loading data from database...")
    
    # Load stock data for a date range
    data = data_loader.load_price_data(
        start_date='2020-01-01',
        end_date='2023-12-31',
        stock_codes=None,  # None = all stocks
        table_name='stock_daily_price'
    )
    
    if data.empty:
        logger.error("No data loaded. Please check database connection and data availability.")
        return
    
    logger.info(f"Loaded {len(data)} rows of data")
    logger.info(f"Date range: {data['trade_date'].min()} to {data['trade_date'].max()}")
    logger.info(f"Number of stocks: {data['stock_code'].nunique()}")
    
    # ========== 3. Calculate Factors ==========
    logger.info("Calculating factors...")
    
    # Calculate multiple factors (use predefined factor names from library)
    factor_names = ['momentum_20', 'volatility_20', 'rsi_14', 'volume_ratio_20']
    data = factor_engine.calculate_factors(data, factor_names, normalize=True, winsorize=True)
    
    # Create composite factor (equal weight)
    data['composite_factor'] = factor_engine.combine_factors(
        data,
        factor_names=['momentum_20', 'volatility_20', 'rsi_14'],
        weights=[0.4, 0.3, 0.3]
    )
    
    logger.info("Factors calculated successfully")
    
    # ========== 4. Run Backtest ==========
    logger.info("Running backtest...")
    
    results = backtest_engine.run_backtest(
        data=data,
        signal_column='composite_factor',  # Use composite factor as signal
        top_n=50,  # Hold top 50 stocks
        rebalance_freq='1W',  # Weekly rebalancing
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    if not results:
        logger.error("Backtest failed")
        return
    
    logger.info("Backtest completed successfully")
    
    # ========== 5. Analyze Results ==========
    logger.info("Analyzing results...")
    
    # Print performance summary
    backtest_engine.print_summary()
    
    # Get detailed results
    portfolio_history = results['portfolio_history']
    trades = results['trades']
    metrics = results['metrics']
    
    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    portfolio_history.to_csv('results/portfolio_history.csv', index=False)
    trades.to_csv('results/trades.csv', index=False)
    
    # Save metrics to file
    with open('results/metrics.txt', 'w') as f:
        f.write("BACKTEST PERFORMANCE METRICS\n")
        f.write("="*50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info("Results saved to 'results/' directory")
    
    # ========== 6. Visualize Results ==========
    logger.info("Generating visualizations...")
    
    # Create output directory for plots
    os.makedirs('results/plots', exist_ok=True)
    
    # Generate all plots
    Plotter.plot_all(results, output_dir='results/plots')
    
    logger.info("Visualizations saved to 'results/plots/' directory")
    
    # ========== 7. Save to Database (Optional) ==========
    logger.info("Saving results to database...")
    
    try:
        # Save portfolio history
        db_manager.save_backtest_results(
            portfolio_history,
            table_name='backtest_portfolio_history'
        )
        
        # Save trades
        db_manager.save_backtest_results(
            trades,
            table_name='backtest_trades'
        )
        
        logger.info("Results saved to database successfully")
    except Exception as e:
        logger.warning(f"Failed to save to database: {e}")
    
    logger.info("="*50)
    logger.info("BACKTEST COMPLETE!")
    logger.info("="*50)


def example_custom_strategy():
    """Example of using custom strategy function"""
    
    logger.info("Running custom strategy example...")
    
    # Initialize components
    db_manager = DatabaseManager('config.yaml')
    data_loader = DataLoader(db_manager)
    backtest_engine = BacktestEngine(initial_capital=1000000)
    
    # Load data
    data = data_loader.load_price_data(
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Define custom strategy
    def my_strategy(date, historical_data, portfolio):
        """
        Custom strategy: Buy stocks with highest momentum
        
        Args:
            date: Current date
            historical_data: All data up to current date
            portfolio: Current portfolio state
            
        Returns:
            Dictionary of target weights {stock_code: weight}
        """
        # Get current date data
        current_data = historical_data[historical_data['trade_date'] == date]
        
        if current_data.empty:
            return {}
        
        # Calculate 20-day momentum
        current_data = current_data.copy()
        current_data['momentum'] = current_data.groupby('stock_code')['adj_close'].pct_change(20)
        
        # Select top 30 stocks by momentum
        top_stocks = current_data.nlargest(30, 'momentum')
        
        # Equal weight
        weights = {
            row['stock_code']: 1.0 / 30
            for _, row in top_stocks.iterrows()
        }
        
        return weights
    
    # Run backtest with custom strategy
    results = backtest_engine.run_custom_strategy(
        data=data,
        strategy_func=my_strategy,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Print results
    backtest_engine.print_summary()
    
    logger.info("Custom strategy backtest complete")


if __name__ == '__main__':
    # Run main example
    main()
    
    # Uncomment to run custom strategy example
    # example_custom_strategy()
