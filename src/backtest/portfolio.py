"""
Portfolio - Manages portfolio positions and transactions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages portfolio state and transactions"""
    
    def __init__(self, initial_capital: float = 1000000.0, 
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.0001):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Initial capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        self.positions = {}  # {stock_code: shares}
        self.position_values = {}  # {stock_code: value}
        
        self.history = []
        self.trades = []
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            prices: Dictionary of current prices {stock_code: price}
            
        Returns:
            Total portfolio value
        """
        position_value = sum(
            self.positions.get(stock, 0) * prices.get(stock, 0)
            for stock in self.positions
        )
        return self.cash + position_value
    
    def get_position_value(self, stock_code: str, price: float) -> float:
        """Get value of a specific position"""
        return self.positions.get(stock_code, 0) * price
    
    def buy(self, stock_code: str, shares: int, price: float, date: str):
        """
        Buy shares
        
        Args:
            stock_code: Stock code
            shares: Number of shares to buy
            price: Purchase price
            date: Trade date
        """
        if shares <= 0:
            return
        
        # Apply slippage (buy at higher price)
        actual_price = price * (1 + self.slippage_rate)
        
        # Calculate cost
        cost = shares * actual_price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        if total_cost > self.cash:
            # Adjust shares to available cash
            max_shares = int(self.cash / (actual_price * (1 + self.commission_rate)))
            if max_shares <= 0:
                logger.warning(f"Insufficient cash to buy {stock_code}")
                return
            shares = max_shares
            cost = shares * actual_price
            commission = cost * self.commission_rate
            total_cost = cost + commission
        
        # Execute trade
        self.cash -= total_cost
        self.positions[stock_code] = self.positions.get(stock_code, 0) + shares
        
        # Record trade
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'BUY',
            'shares': shares,
            'price': actual_price,
            'commission': commission,
            'total_cost': total_cost
        })
        
        logger.debug(f"BUY {shares} shares of {stock_code} at {actual_price:.2f}")
    
    def sell(self, stock_code: str, shares: int, price: float, date: str):
        """
        Sell shares
        
        Args:
            stock_code: Stock code
            shares: Number of shares to sell
            price: Sale price
            date: Trade date
        """
        if shares <= 0:
            return
        
        current_position = self.positions.get(stock_code, 0)
        if current_position <= 0:
            logger.warning(f"No position in {stock_code} to sell")
            return
        
        # Adjust shares to available position
        shares = min(shares, current_position)
        
        # Apply slippage (sell at lower price)
        actual_price = price * (1 - self.slippage_rate)
        
        # Calculate proceeds
        proceeds = shares * actual_price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission
        
        # Execute trade
        self.cash += net_proceeds
        self.positions[stock_code] -= shares
        
        if self.positions[stock_code] == 0:
            del self.positions[stock_code]
        
        # Record trade
        self.trades.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'SELL',
            'shares': shares,
            'price': actual_price,
            'commission': commission,
            'net_proceeds': net_proceeds
        })
        
        logger.debug(f"SELL {shares} shares of {stock_code} at {actual_price:.2f}")
    
    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float], date: str):
        """
        Rebalance portfolio to target weights
        
        Args:
            target_weights: Dictionary of target weights {stock_code: weight}
            prices: Dictionary of current prices {stock_code: price}
            date: Trade date
        """
        portfolio_value = self.get_portfolio_value(prices)
        
        # Calculate target positions
        target_positions = {}
        for stock, weight in target_weights.items():
            if stock in prices and prices[stock] > 0:
                target_value = portfolio_value * weight
                target_shares = int(target_value / prices[stock])
                target_positions[stock] = target_shares
        
        # Sell positions not in target or over-weighted
        for stock in list(self.positions.keys()):
            current_shares = self.positions[stock]
            target_shares = target_positions.get(stock, 0)
            
            if target_shares < current_shares:
                shares_to_sell = current_shares - target_shares
                if stock in prices:
                    self.sell(stock, shares_to_sell, prices[stock], date)
        
        # Buy positions that are under-weighted
        for stock, target_shares in target_positions.items():
            current_shares = self.positions.get(stock, 0)
            
            if target_shares > current_shares:
                shares_to_buy = target_shares - current_shares
                if stock in prices:
                    self.buy(stock, shares_to_buy, prices[stock], date)
    
    def record_state(self, date: str, prices: Dict[str, float]):
        """
        Record current portfolio state
        
        Args:
            date: Current date
            prices: Current prices
        """
        portfolio_value = self.get_portfolio_value(prices)
        
        self.history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'num_positions': len(self.positions)
        })
    
    def get_history_df(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.position_values = {}
        self.history = []
        self.trades = []
