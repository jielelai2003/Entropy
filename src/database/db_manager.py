"""
Database Manager - Handles all database operations
"""
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, Date, Index
from sqlalchemy.orm import sessionmaker
from typing import Optional, Dict, Any, List, Union
import logging
import yaml

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, config: Union[Dict[str, Any], str]):
        """
        Initialize database manager
        
        Args:
            config: Database configuration dictionary OR path to YAML config file
        """
        if isinstance(config, str):
            # Load from YAML file
            with open(config, 'r') as f:
                full_config = yaml.safe_load(f)
                self.config = full_config.get('database', {})
        else:
            self.config = config
            
        self.engine = None
        self.Session = None
        self.metadata = MetaData()
        self._connect()
        
    def _connect(self):
        """Establish database connection"""
        db_type = self.config.get('type', 'sqlite')
        
        if db_type == 'postgresql':
            connection_string = (
                f"postgresql://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
        elif db_type == 'mysql':
            connection_string = (
                f"mysql+pymysql://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            )
        else:  # sqlite
            connection_string = f"sqlite:///{self.config.get('sqlite_path', './data/quant.db')}"
        
        self.engine = create_engine(connection_string, echo=False, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Connected to {db_type} database")
        
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        # Stock price data table
        Table('stock_prices', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('stock_code', String(20), nullable=False),
              Column('trade_date', Date, nullable=False),
              Column('open', Float),
              Column('high', Float),
              Column('low', Float),
              Column('close', Float),
              Column('volume', Float),
              Column('amount', Float),
              Column('adj_factor', Float),
              Index('idx_stock_date', 'stock_code', 'trade_date')
        )
        
        # Factor data table
        Table('factor_data', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('stock_code', String(20), nullable=False),
              Column('trade_date', Date, nullable=False),
              Column('factor_name', String(50), nullable=False),
              Column('factor_value', Float),
              Index('idx_factor_stock_date', 'factor_name', 'stock_code', 'trade_date')
        )
        
        # Backtest results table
        Table('backtest_results', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('strategy_name', String(100), nullable=False),
              Column('trade_date', Date, nullable=False),
              Column('portfolio_value', Float),
              Column('cash', Float),
              Column('positions', String(5000)),
              Column('daily_return', Float),
              Index('idx_strategy_date', 'strategy_name', 'trade_date')
        )
        
        # Performance metrics table
        Table('performance_metrics', self.metadata,
              Column('id', Integer, primary_key=True),
              Column('strategy_name', String(100), nullable=False),
              Column('start_date', Date),
              Column('end_date', Date),
              Column('total_return', Float),
              Column('annual_return', Float),
              Column('sharpe_ratio', Float),
              Column('max_drawdown', Float),
              Column('win_rate', Float),
              Column('calmar_ratio', Float),
              Column('sortino_ratio', Float),
              Index('idx_strategy_name', 'strategy_name')
        )
        
        self.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")
        
    def save_stock_prices(self, df: pd.DataFrame, table_name: str = 'stock_prices'):
        """
        Save stock price data to database
        
        Args:
            df: DataFrame with columns [stock_code, trade_date, open, high, low, close, volume, amount, adj_factor]
            table_name: Name of the table to save to (default: 'stock_prices')
        """
        df.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
        logger.info(f"Saved {len(df)} price records to {table_name}")
        
    def load_stock_prices(self, stock_codes: Optional[List[str]] = None, 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         table_name: str = 'stock_prices') -> pd.DataFrame:
        """
        Load stock price data from database
        
        Args:
            stock_codes: List of stock codes to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            table_name: Name of the table to load from (default: 'stock_prices')
            
        Returns:
            DataFrame with price data
        """
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = {}
        
        if stock_codes:
            placeholders = ','.join([f':code_{i}' for i in range(len(stock_codes))])
            query += f" AND stock_code IN ({placeholders})"
            params.update({f'code_{i}': code for i, code in enumerate(stock_codes)})
            
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY stock_code, trade_date"
        
        df = pd.read_sql(text(query), self.engine, params=params)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        logger.info(f"Loaded {len(df)} price records from {table_name}")
        return df
        
    def save_factor_data(self, df: pd.DataFrame, table_name: str = 'factor_data'):
        """
        Save factor data to database
        
        Args:
            df: DataFrame with columns [stock_code, trade_date, factor_name, factor_value]
            table_name: Name of the table to save to (default: 'factor_data')
        """
        df.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
        logger.info(f"Saved {len(df)} factor records to {table_name}")
        
    def load_factor_data(self, factor_names: Optional[List[str]] = None,
                        stock_codes: Optional[List[str]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        table_name: str = 'factor_data') -> pd.DataFrame:
        """
        Load factor data from database
        
        Args:
            factor_names: List of factor names to load
            stock_codes: List of stock codes to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            table_name: Name of the table to load from (default: 'factor_data')
            
        Returns:
            DataFrame with factor data
        """
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = {}
        
        if factor_names:
            placeholders = ','.join([f':factor_{i}' for i in range(len(factor_names))])
            query += f" AND factor_name IN ({placeholders})"
            params.update({f'factor_{i}': name for i, name in enumerate(factor_names)})
            
        if stock_codes:
            placeholders = ','.join([f':code_{i}' for i in range(len(stock_codes))])
            query += f" AND stock_code IN ({placeholders})"
            params.update({f'code_{i}': code for i, code in enumerate(stock_codes)})
            
        if start_date:
            query += " AND trade_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            query += " AND trade_date <= :end_date"
            params['end_date'] = end_date
            
        query += " ORDER BY factor_name, stock_code, trade_date"
        
        df = pd.read_sql(text(query), self.engine, params=params)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        logger.info(f"Loaded {len(df)} factor records from {table_name}")
        return df
        
    def save_backtest_results(self, df: pd.DataFrame, strategy_name: str, table_name: str = 'backtest_results'):
        """
        Save backtest results to database
        
        Args:
            df: DataFrame with backtest results
            strategy_name: Name of the strategy
            table_name: Name of the table to save to (default: 'backtest_results')
        """
        df['strategy_name'] = strategy_name
        df.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
        logger.info(f"Saved backtest results for {strategy_name} to {table_name}")
        
    def save_performance_metrics(self, metrics: Dict[str, float], strategy_name: str,
                                start_date: str, end_date: str, table_name: str = 'performance_metrics'):
        """
        Save performance metrics to database
        
        Args:
            metrics: Dictionary of performance metrics
            strategy_name: Name of the strategy
            start_date: Backtest start date
            end_date: Backtest end date
            table_name: Name of the table to save to (default: 'performance_metrics')
        """
        data = {
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            **metrics
        }
        pd.DataFrame([data]).to_sql(table_name, self.engine, 
                                    if_exists='append', index=False)
        logger.info(f"Saved performance metrics for {strategy_name} to {table_name}")
        
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute custom SQL query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        return pd.read_sql(text(query), self.engine, params=params or {})
        
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
