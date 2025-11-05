#!/usr/bin/env python3
"""
🏦 FinVision Pro - Advanced Financial Intelligence & Algorithmic Trading System
AI-powered market analysis, portfolio optimization, and risk management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced Financial ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import xgboost as xgb
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.rolling import RollingOLS
import yfinance as yf

# Advanced Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import mplfinance as mpf

class AdvancedFinancialIntelligence:
    """
    🏦 Advanced Financial Intelligence Engine
    Features:
    - Multi-asset Price Prediction
    - Portfolio Optimization & Risk Management
    - Algorithmic Trading Strategies
    - Market Regime Detection
    - Risk Analytics & VaR Calculation
    - Sentiment Analysis Integration
    - Blockchain & Crypto Analytics
    - Real-time Market Monitoring
    """
    
    def __init__(self):
        self.market_data = None
        self.portfolio_data = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.visualizations = {}
        
        # Financial instruments
        self.assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'SPY', 'QQQ', 'GLD']
        self.crypto_assets = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD']
        
    def generate_synthetic_financial_data(self, n_days=1000, start_date='2018-01-01'):
        """
        Generate sophisticated synthetic financial market data
        """
        print("🎲 Generating advanced synthetic financial data...")
        
        np.random.seed(42)
        
        # Generate date range
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        
        # Market regime parameters
        regimes = ['Bull', 'Bear', 'Sideways', 'Volatile']
        regime_durations = [200, 150, 180, 120]  # Days per regime
        current_regime = 0
        regime_start = 0
        
        market_data = {}
        
        for asset in self.assets + self.crypto_assets:
            prices = [100 if asset in self.crypto_assets else 50]  # Starting price
            volumes = [1000000]
            returns = [0]
            
            regime_idx = 0
            regime_days = 0
            
            for i in range(1, n_days):
                # Check for regime change
                if regime_days >= regime_durations[regime_idx]:
                    regime_idx = (regime_idx + 1) % len(regimes)
                    regime_days = 0
                
                regime = regimes[regime_idx]
                regime_days += 1
                
                # Base parameters based on regime
                if regime == 'Bull':
                    drift = 0.0008
                    volatility = 0.015
                    volume_trend = 1.02
                elif regime == 'Bear':
                    drift = -0.0006
                    volatility = 0.025
                    volume_trend = 1.05
                elif regime == 'Sideways':
                    drift = 0.0001
                    volatility = 0.008
                    volume_trend = 0.98
                else:  # Volatile
                    drift = 0.0002
                    volatility = 0.04
                    volume_trend = 1.10
                
                # Asset-specific adjustments
                if asset in self.crypto_assets:
                    volatility *= 1.5
                    drift *= 1.2
                
                # Generate price movement (Geometric Brownian Motion)
                daily_return = np.random.normal(drift, volatility)
                new_price = prices[-1] * np.exp(daily_return)
                
                # Add some jumps (black swan events)
                if np.random.random() < 0.005:  # 0.5% chance of jump
                    jump_size = np.random.normal(0, 0.08)  # ±8% jump
                    new_price *= (1 + jump_size)
                
                # Generate volume with autocorrelation
                new_volume = volumes[-1] * np.random.normal(volume_trend, 0.2)
                new_volume = max(100000, new_volume)  # Minimum volume
                
                prices.append(new_price)
                volumes.append(new_volume)
                returns.append(daily_return)
            
            # Create DataFrame for this asset
            asset_df = pd.DataFrame({
                'date': dates,
                'asset': asset,
                'close': prices,
                'volume': volumes,
                'daily_return': returns,
                'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
                'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices]
            })
            
            # Calculate technical indicators
            asset_df = self._calculate_technical_indicators(asset_df)
            market_data[asset] = asset_df
        
        # Combine all assets
        self.market_data = pd.concat(market_data.values(), ignore_index=True)
        
        # Add market microstructure features
        self._add_market_microstructure()
        
        print(f"✅ Generated {len(self.market_data)} market records across {len(self.assets + self.crypto_assets)} assets")
        return self.market_data
    
    def _calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volatility measures
        df['volatility_20'] = df['daily_return'].rolling(window=20).std()
        df['volatility_50'] = df['daily_return'].rolling(window=50).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        return df
    
    def _add_market_microstructure(self):
        """Add market microstructure features"""
        # Calculate spread (bid-ask spread simulation)
        self.market_data['spread'] = self.market_data['close'] * np.random.normal(0.001, 0.0005)
        
        # Market depth (simulated)
        self.market_data['market_depth'] = np.random.lognormal(10, 1, len(self.market_data))
        
        # Order flow imbalance
        self.market_data['order_imbalance'] = np.random.normal(0, 0.1, len(self.market_data))
        
        # Volatility clustering (GARCH-like behavior)
        self.market_data['volatility_cluster'] = self.market_data['volatility_20'].shift(1) * 0.9 + np.random.normal(0, 0.01)
    
    def price_prediction_engine(self):
        """
        Advanced price prediction using ensemble ML models
        """
        print("\n🔮 Training advanced price prediction models...")
        
        prediction_results = {}
        
        for asset in self.assets[:3]:  # Focus on first 3 assets for demonstration
            print(f"📈 Training models for {asset}...")
            
            asset_data = self.market_data[self.market_data['asset'] == asset].copy()
            asset_data = asset_data.dropna()
            
            # Features for prediction
            feature_columns = [
                'sma_20', 'sma_50', 'rsi', 'macd', 'macd_histogram',
                'volatility_20', 'volume_ratio', 'momentum_10', 'bb_position',
                'spread', 'market_depth', 'order_imbalance'
            ]
            
            # Create lagged features
            for lag in [1, 2, 3, 5]:
                for feature in ['daily_return', 'volume_ratio', 'rsi']:
                    asset_data[f'{feature}_lag_{lag}'] = asset_data[feature].shift(lag)
                    feature_columns.append(f'{feature}_lag_{lag}')
            
            # Target: Next day return
            asset_data['target'] = asset_data['daily_return'].shift(-1)
            asset_data = asset_data.dropna()
            
            X = asset_data[feature_columns]
            y = asset_data['target']
            
            # Time series split validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
            }
            
            cv_scores = {}
            feature_importances = {}
            
            for name, model in models.items():
                scores = []
                for train_idx, test_idx in tscv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    scores.append(r2_score(y_test, y_pred))
                
                cv_scores[name] = np.mean(scores)
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importances[name] = dict(zip(feature_columns, model.feature_importances_))
            
            best_model_name = max(cv_scores, key=cv_scores.get)
            
            prediction_results[asset] = {
                'best_model': best_model_name,
                'cv_r2_score': cv_scores[best_model_name],
                'all_scores': cv_scores,
                'feature_importance': feature_importances.get(best_model_name, {}),
                'predictive_features': sorted(
                    feature_importances.get(best_model_name, {}).items(),
                    key=lambda x: x[1], reverse=True
                )[:5]  # Top 5 features
            }
        
        self.results['price_prediction'] = prediction_results
        print("✅ Price prediction models trained successfully")
        return prediction_results
    
    def portfolio_optimization(self, initial_capital=100000):
        """
        Advanced portfolio optimization using Modern Portfolio Theory and ML
        """
        print("\n💼 Performing portfolio optimization...")
        
        # Prepare returns data
        returns_data = self.market_data.pivot(index='date', columns='asset', values='daily_return')
        returns_data = returns_data.dropna()
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean() * 252  # Annualized
        cov_matrix = returns_data.cov() * 252  # Annualized
        
        # Markowitz Portfolio Optimization
        def portfolio_stats(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility
            return portfolio_return, portfolio_volatility, sharpe_ratio
        
        def objective_function(weights):
            return -portfolio_stats(weights)[2]  # Maximize Sharpe ratio
        
        # Constraints
        n_assets = len(expected_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # weights sum to 1
        bounds = tuple((0.01, 0.3) for _ in range(n_assets))  # No short selling, max 30% per asset
        
        # Initial guess (equal weights)
        init_guess = n_assets * [1.0 / n_assets]
        
        # Optimization
        optimized_results = minimize(
            objective_function, init_guess,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        optimal_weights = optimized_results.x
        
        # Calculate optimal portfolio metrics
        opt_return, opt_volatility, opt_sharpe = portfolio_stats(optimal_weights)
        
        # Risk metrics
        portfolio_returns = returns_data.dot(optimal_weights)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Create portfolio DataFrame
        portfolio_allocation = pd.DataFrame({
            'asset': expected_returns.index,
            'weight': optimal_weights,
            'expected_return': expected_returns.values
        }).sort_values('weight', ascending=False)
        
        portfolio_analysis = {
            'optimal_weights': dict(zip(portfolio_allocation['asset'], portfolio_allocation['weight'])),
            'expected_annual_return': opt_return,
            'expected_annual_volatility': opt_volatility,
            'sharpe_ratio': opt_sharpe,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'portfolio_allocation': portfolio_allocation.to_dict('records')
        }
        
        self.results['portfolio_optimization'] = portfolio_analysis
        print(f"✅ Portfolio optimization completed - Sharpe Ratio: {opt_sharpe:.3f}")
        return portfolio_analysis
    
    def market_regime_detection(self):
        """
        Detect market regimes using clustering and ML
        """
        print("\n🔄 Performing market regime detection...")
        
        # Prepare market state data
        market_features = self.market_data.groupby('date').agg({
            'daily_return': ['mean', 'std'],
            'volume': 'sum',
            'volatility_20': 'mean',
            'rsi': 'mean'
        }).reset_index()
        
        market_features.columns = ['date', 'market_return', 'market_volatility', 'total_volume', 'avg_volatility', 'avg_rsi']
        
        # Additional features
        market_features['volatility_regime'] = (market_features['avg_volatility'] > market_features['avg_volatility'].rolling(50).mean()).astype(int)
        market_features['trend_strength'] = market_features['market_return'].rolling(20).std()
        
        # Features for clustering
        regime_features = ['market_return', 'market_volatility', 'avg_volatility', 'avg_rsi', 'trend_strength']
        X_regime = market_features[regime_features].dropna()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_regime)
        
        # K-means clustering for regime detection
        kmeans = KMeans(n_clusters=4, random_state=42)
        regime_labels = kmeans.fit_predict(X_scaled)
        
        # Analyze each regime
        regime_analysis = {}
        for regime_id in range(4):
            regime_data = market_features.iloc[np.where(regime_labels == regime_id)[0]]
            
            regime_profile = {
                'size': len(regime_data),
                'avg_return': regime_data['market_return'].mean(),
                'avg_volatility': regime_data['market_volatility'].mean(),
                'avg_rsi': regime_data['avg_rsi'].mean(),
                'typical_duration': self._calculate_regime_duration(regime_labels, regime_id)
            }
            
            # Label regimes based on characteristics
            if regime_profile['avg_return'] > 0.001 and regime_profile['avg_volatility'] < 0.02:
                regime_name = "Bull Market"
            elif regime_profile['avg_return'] < -0.001 and regime_profile['avg_volatility'] > 0.025:
                regime_name = "Bear Market"
            elif regime_profile['avg_volatility'] > 0.03:
                regime_name = "High Volatility"
            else:
                regime_name = "Sideways Market"
            
            regime_analysis[regime_name] = regime_profile
        
        # Add regime labels to market data
        market_features['regime'] = regime_labels
        market_features['regime_name'] = [list(regime_analysis.keys())[r] for r in regime_labels]
        
        self.results['market_regimes'] = regime_analysis
        print(f"✅ Market regime detection completed - {len(regime_analysis)} regimes identified")
        return regime_analysis
    
    def _calculate_regime_duration(self, labels, regime_id):
        """Calculate typical duration of market regimes"""
        durations = []
        current_duration = 0
        
        for label in labels:
            if label == regime_id:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def risk_analytics(self):
        """
        Comprehensive risk analytics including VaR, CVaR, and stress testing
        """
        print("\n⚠️ Performing comprehensive risk analytics...")
        
        # Calculate portfolio returns (equal weights for demonstration)
        returns_data = self.market_data.pivot(index='date', columns='asset', values='daily_return').dropna()
        portfolio_returns = returns_data.mean(axis=1)  # Equal-weighted portfolio
        
        # Value at Risk (VaR) calculations
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Stress testing - historical scenarios
        stress_periods = {
            'covid_crash': portfolio_returns['2020-02-20':'2020-03-23'],
            'volatility_spike': portfolio_returns.nsmallest(20),  # Worst 20 days
            'bull_market': portfolio_returns.nlargest(20)  # Best 20 days
        }
        
        stress_results = {}
        for scenario, period_returns in stress_periods.items():
            stress_results[scenario] = {
                'avg_return': period_returns.mean(),
                'volatility': period_returns.std(),
                'min_return': period_returns.min(),
                'max_return': period_returns.max()
            }
        
        # Correlation analysis
        correlation_matrix = returns_data.corr()
        
        risk_analysis = {
            'value_at_risk': {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99
            },
            'drawdown_analysis': {
                'max_drawdown': max_drawdown,
                'drawdown_duration': self._calculate_drawdown_duration(drawdown)
            },
            'stress_testing': stress_results,
            'correlation_analysis': {
                'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                'highest_correlation': correlation_matrix.stack().nlargest(6).iloc[1:].to_dict()  # Skip self-correlation
            }
        }
        
        self.results['risk_analytics'] = risk_analysis
        print(f"🚨 Risk analytics completed - Max Drawdown: {max_drawdown:.2%}")
        return risk_analysis
    
    def _calculate_drawdown_duration(self, drawdown_series):
        """Calculate average drawdown duration"""
        in_drawdown = False
        durations = []
        current_duration = 0
        
        for dd in drawdown_series:
            if dd < -0.01:  # Considered in drawdown if >1% below peak
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    durations.append(current_duration)
                    in_drawdown = False
                    current_duration = 0
        
        return np.mean(durations) if durations else 0
    
    def algorithmic_trading_strategies(self):
        """
        Implement and backtest algorithmic trading strategies
        """
        print("\n🤖 Developing algorithmic trading strategies...")
        
        strategy_results = {}
        
        # Strategy 1: Momentum Strategy
        strategy_results['momentum'] = self._backtest_momentum_strategy()
        
        # Strategy 2: Mean Reversion Strategy
        strategy_results['mean_reversion'] = self._backtest_mean_reversion_strategy()
        
        # Strategy 3: Machine Learning Strategy
        strategy_results['ml_strategy'] = self._backtest_ml_strategy()
        
        # Compare strategies
        comparison = {}
        for strategy_name, results in strategy_results.items():
            comparison[strategy_name] = {
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate']
            }
        
        self.results['trading_strategies'] = {
            'strategy_results': strategy_results,
            'comparison': comparison,
            'best_strategy': max(comparison, key=lambda x: comparison[x]['sharpe_ratio'])
        }
        
        print("✅ Algorithmic trading strategies backtested successfully")
        return strategy_results
    
    def _backtest_momentum_strategy(self):
        """Backtest momentum-based trading strategy"""
        # Simple momentum strategy: Buy when price above 50-day MA, sell when below
        asset_data = self.market_data[self.market_data['asset'] == 'AAPL'].copy()
        asset_data['position'] = np.where(asset_data['close'] > asset_data['sma_50'], 1, -1)
        asset_data['strategy_returns'] = asset_data['position'].shift(1) * asset_data['daily_return']
        
        total_return = (1 + asset_data['strategy_returns'].fillna(0)).cumprod().iloc[-1] - 1
        sharpe = asset_data['strategy_returns'].mean() / asset_data['strategy_returns'].std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self._calculate_max_drawdown(asset_data['strategy_returns']),
            'win_rate': (asset_data['strategy_returns'] > 0).mean()
        }
    
    def _backtest_mean_reversion_strategy(self):
        """Backtest mean reversion trading strategy"""
        # Mean reversion: Buy when RSI < 30, sell when RSI > 70
        asset_data = self.market_data[self.market_data['asset'] == 'AAPL'].copy()
        asset_data['position'] = np.where(asset_data['rsi'] < 30, 1, np.where(asset_data['rsi'] > 70, -1, 0))
        asset_data['strategy_returns'] = asset_data['position'].shift(1) * asset_data['daily_return']
        
        total_return = (1 + asset_data['strategy_returns'].fillna(0)).cumprod().iloc[-1] - 1
        sharpe = asset_data['strategy_returns'].mean() / asset_data['strategy_returns'].std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self._calculate_max_drawdown(asset_data['strategy_returns']),
            'win_rate': (asset_data['strategy_returns'] > 0).mean()
        }
    
    def _backtest_ml_strategy(self):
        """Backtest ML-based trading strategy"""
        # Simplified ML strategy using prediction signals
        asset_data = self.market_data[self.market_data['asset'] == 'AAPL'].copy()
        
        # Use momentum and volatility as signals (simplified ML)
        asset_data['ml_signal'] = np.where(
            (asset_data['momentum_10'] > 0.02) & (asset_data['volatility_20'] < 0.02), 1,
            np.where(
                (asset_data['momentum_10'] < -0.02) & (asset_data['volatility_20'] > 0.03), -1, 0
            )
        )
        
        asset_data['strategy_returns'] = asset_data['ml_signal'].shift(1) * asset_data['daily_return']
        
        total_return = (1 + asset_data['strategy_returns'].fillna(0)).cumprod().iloc[-1] - 1
        sharpe = asset_data['strategy_returns'].mean() / asset_data['strategy_returns'].std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self._calculate_max_drawdown(asset_data['strategy_returns']),
            'win_rate': (asset_data['strategy_returns'] > 0).mean()
        }
    
    def _calculate_max_drawdown(self, returns_series):
        """Calculate maximum drawdown from returns series"""
        cumulative_returns = (1 + returns_series.fillna(0)).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def create_financial_visualizations(self):
        """
        Create advanced financial visualizations and dashboards
        """
        print("\n📊 Creating advanced financial visualizations...")
        
        # 1. Price Prediction Performance
        prediction_results = self.results['price_prediction']
        assets = list(prediction_results.keys())
        r2_scores = [prediction_results[asset]['cv_r2_score'] for asset in assets]
        
        fig_prediction = px.bar(x=assets, y=r2_scores,
                              title='Price Prediction Model Performance (R² Scores)',
                              labels={'x': 'Asset', 'y': 'R² Score'})
        
        # 2. Portfolio Allocation
        portfolio_allocation = self.results['portfolio_optimization']['portfolio_allocation']
        allocation_df = pd.DataFrame(portfolio_allocation)
        
        fig_allocation = px.pie(allocation_df, values='weight', names='asset',
                              title='Optimal Portfolio Allocation')
        
        # 3. Risk-Return Scatter Plot
        returns_data = self.market_data.groupby('asset').agg({
            'daily_return': ['mean', 'std']
        }).reset_index()
        returns_data.columns = ['asset', 'avg_return', 'volatility']
        returns_data['annual_return'] = returns_data['avg_return'] * 252
        returns_data['annual_volatility'] = returns_data['volatility'] * np.sqrt(252)
        
        fig_risk_return = px.scatter(returns_data, x='annual_volatility', y='annual_return',
                                   text='asset', title='Risk-Return Profile of Assets',
                                   labels={'annual_volatility': 'Annual Volatility', 
                                          'annual_return': 'Annual Return'})
        
        # 4. Correlation Heatmap
        corr_data = self.market_data.pivot(index='date', columns='asset', values='daily_return').corr()
        fig_correlation = ff.create_annotated_heatmap(
            z=corr_data.values,
            x=corr_data.columns.tolist(),
            y=corr_data.columns.tolist(),
            annotation_text=corr_data.round(2).values,
            colorscale='RdBu_r'
        )
        fig_correlation.update_layout(title='Asset Correlation Matrix')
        
        self.visualizations = {
            'prediction_performance': fig_prediction,
            'portfolio_allocation': fig_allocation,
            'risk_return': fig_risk_return,
            'correlation_matrix': fig_correlation
        }
        
        print("✅ Advanced financial visualizations created")
        return self.visualizations
    
    def generate_financial_insights(self):
        """
        Generate comprehensive financial insights and investment recommendations
        """
        print("\n📈 Generating financial insights report...")
        
        insights = {
            'market_analysis': [],
            'investment_opportunities': [],
            'risk_assessment': [],
            'trading_strategy_insights': [],
            'portfolio_recommendations': []
        }
        
        # Market analysis
        if 'price_prediction' in self.results:
            best_asset = max(self.results['price_prediction'].items(), 
                           key=lambda x: x[1]['cv_r2_score'])
            insights['market_analysis'].extend([
                f"📈 Best predictable asset: {best_asset[0]} (R²: {best_asset[1]['cv_r2_score']:.3f})",
                f"🎯 Top predictive features: {', '.join([f[0] for f in best_asset[1]['predictive_features'][:3]])}",
                f"🔍 Market regimes detected: {len(self.results['market_regimes'])} distinct patterns"
            ])
        
        # Investment opportunities
        if 'portfolio_optimization' in self.results:
            top_assets = self.results['portfolio_optimization']['portfolio_allocation'][:3]
            insights['investment_opportunities'].extend([
                f"💼 Top portfolio allocations: {', '.join([f\"{a['asset']} ({a['weight']:.1%})\" for a in top_assets])}",
                f"📊 Expected portfolio return: {self.results['portfolio_optimization']['expected_annual_return']:.2%}",
                f"⚡ Portfolio Sharpe ratio: {self.results['portfolio_optimization']['sharpe_ratio']:.3f}"
            ])
        
        # Risk assessment
        if 'risk_analytics' in self.results:
            risk_data = self.results['risk_analytics']
            insights['risk_assessment'].extend([
                f"⚠️  95% VaR: {risk_data['value_at_risk']['var_95']:.2%}",
                f"🚨 Maximum drawdown: {risk_data['drawdown_analysis']['max_drawdown']:.2%}",
                f"📉 Expected shortfall (95%): {risk_data['value_at_risk']['cvar_95']:.2%}"
            ])
        
        # Trading strategy insights
        if 'trading_strategies' in self.results:
            best_strategy = self.results['trading_strategies']['best_strategy']
            strategy_perf = self.results['trading_strategies']['comparison'][best_strategy]
            insights['trading_strategy_insights'].extend([
                f"🤖 Best performing strategy: {best_strategy}",
                f"📈 Strategy return: {strategy_perf['total_return']:.2%}",
                f"🎯 Strategy Sharpe ratio: {strategy_perf['sharpe_ratio']:.3f}"
            ])
        
        # Portfolio recommendations
        insights['portfolio_recommendations'].extend([
            "🎯 Consider dynamic asset allocation based on market regimes",
            "💡 Implement risk parity approach for better risk-adjusted returns",
            "🛡️ Use options strategies for downside protection in volatile markets",
            "🌍 Diversify across uncorrelated asset classes and geographies",
            "🤖 Deploy algorithmic strategies for systematic alpha generation",
            "📊 Regularly rebalance portfolio based on changing market conditions"
        ])
        
        self.results['financial_insights'] = insights
        
        print("✅ Comprehensive financial insights report generated")
        return insights
    
    def save_financial_analysis(self):
        """
        Save all financial analysis results and visualizations
        """
        print("\n💾 Saving financial analysis results...")
        
        # Save market data
        self.market_data.to_csv('sample_data/market_datasets/comprehensive_market_data.csv', index=False)
        
        # Save analysis results
        results_summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'assets_analyzed': len(self.market_data['asset'].unique()),
            'time_period': f"{self.market_data['date'].min()} to {self.market_data['date'].max()}",
            'key_metrics': {
                'average_daily_return': self.market_data['daily_return'].mean(),
                'average_volatility': self.market_data['volatility_20'].mean(),
                'best_prediction_r2': max([r['cv_r2_score'] for r in self.results['price_prediction'].values()]),
                'optimal_portfolio_sharpe': self.results['portfolio_optimization']['sharpe_ratio']
            },
            'financial_insights': self.results.get('financial_insights', {})
        }
        
        import json
        with open('sample_data/analysis_results/financial_analysis_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save visualizations as HTML
        for viz_name, fig in self.visualizations.items():
            fig.write_html(f'sample_data/analysis_results/{viz_name}_visualization.html')
        
        print("✅ All financial analysis results saved")
    
    def run_complete_financial_analysis(self):
        """
        Run the complete advanced financial analytics pipeline
        """
        print("🏦 STARTING ADVANCED FINANCIAL ANALYTICS PIPELINE")
        print("=" * 60)
        
        # Step 1: Generate financial data
        self.generate_synthetic_financial_data(1000)
        
        # Step 2: Price prediction
        self.price_prediction_engine()
        
        # Step 3: Portfolio optimization
        self.portfolio_optimization()
        
        # Step 4: Market regime detection
        self.market_regime_detection()
        
        # Step 5: Risk analytics
        self.risk_analytics()
        
        # Step 6: Algorithmic trading strategies
        self.algorithmic_trading_strategies()
        
        # Step 7: Visualizations
        self.create_financial_visualizations()
        
        # Step 8: Financial insights
        insights = self.generate_financial_insights()
        
        # Step 9: Save results
        self.save_financial_analysis()
        
        print("\n🎉 FINANCIAL ANALYTICS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print key financial insights
        print("\n📊 KEY FINANCIAL FINDINGS:")
        print("-" * 25)
        for finding in insights['market_analysis'][:2]:
            print(f"• {finding}")
        
        print("\n🎯 INVESTMENT RECOMMENDATIONS:")
        print("-" * 25)
        for recommendation in insights['portfolio_recommendations'][:3]:
            print(f"• {recommendation}")
        
        return self.results

def main():
    """
    Main function to demonstrate the advanced financial intelligence engine
    """
    # Initialize the financial analytics engine
    financial_engine = AdvancedFinancialIntelligence()
    
    # Run complete financial analysis
    results = financial_engine.run_complete_financial_analysis()
    
    print(f"\n📁 Financial analysis results saved in 'sample_data/' directory")
    print("🔍 Open the HTML files in your browser to view interactive financial dashboards")

if __name__ == "__main__":
    main()
