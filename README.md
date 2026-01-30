# Entropy - Multi-Factor Backtesting Framework

A professional-grade Python framework for quantitative multi-factor backtesting with database integration, factor calculation, portfolio management, and comprehensive visualization.

## Features

- **Database Management**: SQLite/PostgreSQL/MySQL support with connection pooling
- **Data Pipeline**: Efficient data loading and preprocessing
- **Factor Library**: 15+ built-in technical factors (momentum, volatility, RSI, MACD, etc.)
- **Factor Engine**: Calculate, combine, and normalize factors
- **Backtesting Engine**: Event-driven backtesting with realistic transaction costs
- **Portfolio Management**: Position tracking, rebalancing, commission/slippage modeling
- **Performance Analytics**: 15+ metrics (Sharpe, Sortino, Calmar, Alpha, Beta, etc.)
- **Visualization**: Publication-quality plots for analysis

## Architecture

```
Entropy/
├── src/
│   ├── database/          # Database connection and management
│   ├── data/              # Data loading and preprocessing
│   ├── factors/           # Factor calculation library
│   ├── backtest/          # Backtesting engine and portfolio
│   └── visualization/     # Plotting and analysis tools
├── examples/              # Usage examples
├── config.yaml            # Configuration file
└── requirements.txt       # Dependencies
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Entropy

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Database

Edit `config.yaml`:

```yaml
database:
  type: sqlite  # or postgresql, mysql
  sqlite:
    path: data/stocks.db
```

### 2. Run Example

```python
python examples/example_backtest.py
```

### 3. Basic Usage

```python
from src.database.db_manager import DatabaseManager
from src.data.data_loader import DataLoader
from src.factors.factor_engine import FactorEngine
from src.backtest.backtest_engine import BacktestEngine
from src.visualization.plotter import Plotter

# Initialize
db = DatabaseManager('config.yaml')
loader = DataLoader(db)
factor_engine = FactorEngine()
backtest = BacktestEngine(initial_capital=1000000)

# Load data
data = loader.load_stock_data(
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Calculate factors
data = factor_engine.add_factor(data, 'momentum', window=20)
data = factor_engine.add_factor(data, 'rsi', window=14)

# Create composite factor
factor_engine.create_composite_factor(
    data,
    factor_names=['momentum_20', 'rsi_14'],
    weights=[0.6, 0.4],
    composite_name='my_factor'
)

# Run backtest
results = backtest.run_backtest(
    data=data,
    signal_column='my_factor',
    top_n=50,
    rebalance_freq='1W'
)

# Analyze results
backtest.print_summary()
Plotter.plot_all(results, output_dir='results/plots')
```

## Available Factors

### Momentum Factors
- `momentum`: Price momentum over N periods
- `roc`: Rate of change
- `macd`: Moving Average Convergence Divergence

### Volatility Factors
- `volatility`: Rolling standard deviation
- `atr`: Average True Range
- `bollinger_width`: Bollinger Band width

### Trend Factors
- `sma`: Simple Moving Average
- `ema`: Exponential Moving Average
- `adx`: Average Directional Index

### Oscillators
- `rsi`: Relative Strength Index
- `stochastic`: Stochastic Oscillator
- `cci`: Commodity Channel Index

### Volume Factors
- `volume_ratio`: Volume relative to average
- `obv`: On-Balance Volume
- `mfi`: Money Flow Index

## Performance Metrics

### Return Metrics
- Total Return
- Annualized Return

### Risk Metrics
- Volatility (annualized)
- Maximum Drawdown

### Risk-Adjusted Returns
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Trading Metrics
- Win Rate
- Profit Factor

### Benchmark-Relative
- Beta
- Alpha
- Information Ratio

## Custom Strategy Example

```python
def my_strategy(date, historical_data, portfolio):
    """Custom strategy function"""
    current_data = historical_data[historical_data['trade_date'] == date]
    
    # Your logic here
    current_data['signal'] = calculate_my_signal(current_data)
    
    # Select top stocks
    top_stocks = current_data.nlargest(30, 'signal')
    
    # Return target weights
    return {
        row['stock_code']: 1.0 / 30
        for _, row in top_stocks.iterrows()
    }

# Run backtest
results = backtest.run_custom_strategy(
    data=data,
    strategy_func=my_strategy
)
```

## Database Schema

### Stock Daily Data
```sql
CREATE TABLE stock_daily (
    trade_date DATE,
    stock_code VARCHAR(20),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    adj_close FLOAT,
    PRIMARY KEY (trade_date, stock_code)
);
```

## Configuration Options

### Backtest Parameters
- `initial_capital`: Starting capital (default: 1,000,000)
- `commission_rate`: Commission per trade (default: 0.0003)
- `slippage_rate`: Slippage per trade (default: 0.0001)
- `top_n`: Number of stocks to hold (default: 50)
- `rebalance_freq`: Rebalancing frequency ('1D', '1W', '1M')

### Factor Parameters
- `window`: Lookback period for calculations
- `method`: Calculation method ('rank', 'zscore', 'minmax')
- `weights`: Weights for composite factors

## Output

### Results Directory Structure
```
results/
├── portfolio_history.csv      # Portfolio value over time
├── trades.csv                 # All trades executed
├── metrics.txt                # Performance metrics
└── plots/
    ├── portfolio_value.png
    ├── returns.png
    ├── drawdown.png
    ├── rolling_metrics.png
    ├── monthly_returns.png
    └── trade_analysis.png
```

## Best Practices

1. **Data Quality**: Ensure clean, adjusted price data
2. **Factor Design**: Test factors individually before combining
3. **Transaction Costs**: Use realistic commission and slippage rates
4. **Rebalancing**: Balance between turnover and signal capture
5. **Overfitting**: Use out-of-sample testing and cross-validation
6. **Risk Management**: Monitor drawdowns and position concentration

## Extending the Framework

### Add Custom Factor

```python
from src.factors.factor_library import FactorLibrary

@staticmethod
def my_custom_factor(df, window=20):
    """Calculate custom factor"""
    # Your calculation logic
    return df.groupby('stock_code')['close'].transform(
        lambda x: x.rolling(window).apply(my_calculation)
    )

# Register factor
FactorLibrary.my_custom_factor = my_custom_factor
```

### Add Custom Metric

```python
from src.backtest.performance import PerformanceAnalyzer

@staticmethod
def my_custom_metric(returns):
    """Calculate custom metric"""
    # Your calculation logic
    return custom_value

# Use in analysis
PerformanceAnalyzer.my_custom_metric = my_custom_metric
```

## Performance Optimization

- Use vectorized pandas operations
- Batch database queries
- Cache factor calculations
- Parallel processing for multiple backtests

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check config.yaml settings
   - Verify database credentials
   - Ensure database server is running

2. **Empty Data**
   - Verify date range has data
   - Check stock_codes filter
   - Confirm database table exists

3. **Factor Calculation Error**
   - Ensure sufficient historical data
   - Check for missing values
   - Verify column names

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.

---

**Built with Python 3.8+ | Pandas | NumPy | Matplotlib | SQLAlchemy**
