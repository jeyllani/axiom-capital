import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

class PortfolioDataLoader:
    """
    Centralized Data Loader for Portfolio Analysis.
    Handles loading, cleaning, filtering, and sector mapping.
    """
    def __init__(self, config):
        """
        Initialize with configuration dictionary.
        Required keys: 'data_dir', 'input_file', 'test_start', 'test_end', 'lookback_months'
        Optional keys: 'n_stocks', 'random_seed', 'coverage_threshold'
        """
        self.config = config
        self.df = None
        self.returns_matrix = None
        self.market_caps_matrix = None
        self.sector_map = None
        self.name_map = None
        self.rf_series = None
        
    def load_data(self):
        """
        Dispatcher method to load data based on configured source.
        """
        source = self.config.get('source', 'CRSP').upper()
        
        if source == 'YFINANCE':
            self._run_strategy_yfinance()
        else:
            self._run_strategy_crsp()

    def _run_strategy_crsp(self):
        """
        Original CRSP Loading Strategy.
        """
        input_path = Path(self.config['data_dir']) / self.config['input_file']
        if not input_path.exists():
            raise FileNotFoundError(f"Data file not found: {input_path}")
            
        # 1. Load Raw Data
        df = pl.read_parquet(input_path)
        
        # 2. Determine Date Range (Test Period + Lookback)
        test_start_dt = pd.Timestamp(self.config['test_start'])
        lookback_months = self.config.get('lookback_months', 24)
        
        # Buffer for lookback (approx 2 years + 1 month)
        initial_start = test_start_dt - pd.DateOffset(months=lookback_months + 12) 
        # Note: We load extra history to ensure valid lookback windows
        
        # Filter by Date
        # Convert to polars date for filtering
        # df = df.filter(pl.col('date') >= pl.date(initial_start.year, initial_start.month, initial_start.day))
        # (Optional: Filter strictly if dataset is huge, but usually fine to keep all history for liquid universe selection)
        
        # 3. Universe Selection (Liquidity/Coverage)
        # We select stocks that have enough history BEFORE the test start
        # to ensure they are investable at t=0.
        
        # Define "Pre-Test" period for selection
        pre_test_end = test_start_dt - pd.DateOffset(days=1)
        pre_test_start = pre_test_end - pd.DateOffset(years=2)
        
        df_pre = df.filter((pl.col('date') >= pl.date(pre_test_start.year, pre_test_start.month, pre_test_start.day)) & 
                           (pl.col('date') <= pl.date(pre_test_end.year, pre_test_end.month, pre_test_end.day)))
                           
        coverage = df_pre.group_by('Ticker').agg(pl.col('DlyRet').count().alias('n'))
        threshold = self.config.get('coverage_threshold', 0.80) * (252*2) # Approx 80% of 2 years
        valid_tickers = coverage.filter(pl.col('n') > 100).select('Ticker').to_series().to_list()
        
        if not valid_tickers:
            raise ValueError(f"No assets found for universe selection period ({pre_test_start.date()} to {pre_test_end.date()}). "
                             f"Check your 'test_start' date. Data available from {df['date'].min().date()}.")
            
        # 4. Sector Filtering (if requested in config)
        # We need to map sectors first to filter by them
        temp_sector_map = self._create_sector_map(df)
        
        requested_sectors = self.config.get('sectors', None) # List of sectors or None
        if requested_sectors is not None:
            # Filter tickers that belong to requested sectors
            valid_tickers = [t for t in valid_tickers if temp_sector_map.get(t, 'Other') in requested_sectors]
            
        # 5. Random Sampling (if n_stocks specified)
        n_stocks = self.config.get('n_stocks', len(valid_tickers))
        if n_stocks < len(valid_tickers):
            np.random.seed(self.config.get('random_seed', 42))
            selected_tickers = np.random.choice(valid_tickers, size=n_stocks, replace=False).tolist()
        else:
            selected_tickers = valid_tickers
            
        # 6. Filter Main DataFrame
        df_universe = df.filter(pl.col('Ticker').is_in(selected_tickers))
        
        # 7. Prepare Matrices (Pandas)
        df_pd = df_universe.select(['date', 'Ticker', 'DlyRet', 'DlyCap']).to_pandas()
        df_pd['date'] = pd.to_datetime(df_pd['date'])
        df_pd = df_pd.sort_values(['Ticker', 'date']).drop_duplicates(subset=['date', 'Ticker'], keep='last')
        df_pd['year_month'] = df_pd['date'].dt.to_period('M')
        
        # Monthly Returns
        monthly_returns = df_pd.groupby(['Ticker', 'year_month'])['DlyRet'].apply(lambda x: (1 + x).prod() - 1).reset_index()
        monthly_returns['date'] = monthly_returns['year_month'].dt.to_timestamp('M')
        self.returns_matrix = monthly_returns.pivot(index='date', columns='Ticker', values='DlyRet').sort_index()
        
        # Monthly Market Caps
        monthly_caps = df_pd.groupby(['Ticker', 'year_month'])['DlyCap'].last().reset_index()
        monthly_caps['date'] = monthly_caps['year_month'].dt.to_timestamp('M')
        self.market_caps_matrix = monthly_caps.pivot(index='date', columns='Ticker', values='DlyCap').sort_index()
    
        # =================================================================
        # FILTER: Test Period + Lookback Buffer
        # =================================================================
        test_start = pd.to_datetime(self.config['test_start'])
        test_end = pd.to_datetime(self.config['test_end'])
        lookback = self.config.get('lookback_months', 24)
        
        # Buffer: Start date must be earlier to allow lookback calculation at t=0
        # We add ~2 months extra buffer to be safe
        buffer_start = test_start - pd.DateOffset(months=lookback + 2)
        
        if self.returns_matrix is not None:
            self.returns_matrix = self.returns_matrix.loc[buffer_start:test_end]
        if self.market_caps_matrix is not None:
            self.market_caps_matrix = self.market_caps_matrix.loc[buffer_start:test_end]
            
        print(f"📅 Date Range Loaded (CRSP): {self.returns_matrix.index[0].date()} to {self.returns_matrix.index[-1].date()} (Includes {lookback}m lookback)")

        # Final Sector Map (for selected tickers)
        self.sector_map = {t: temp_sector_map.get(t, 'Other') for t in selected_tickers}
        
        # Create Name Map
        temp_name_map = self._create_name_map(df)
        self.name_map = {t: temp_name_map.get(t, t) for t in selected_tickers}
        
        # 8. Load Risk Free Rate (if configured)
        if 'rf_file' in self.config:
            rf_path = Path(self.config['data_dir']) / self.config['rf_file']
            if rf_path.exists():
                rf_df = pd.read_parquet(rf_path)
                # Ensure date index
                if 'date' in rf_df.columns:
                    rf_df = rf_df.set_index('date')
                
                # Resample to Monthly (Compounding)
                # rf_df is daily decimal.
                # We need to align with monthly_returns['year_month'] logic or just resample 'M'
                # monthly_returns uses 'M' (End of Month)
                
                # Create year_month for grouping to match returns exactly
                rf_df['year_month'] = rf_df.index.to_period('M')
                rf_monthly = rf_df.groupby('year_month')['RiskFree'].apply(lambda x: (1 + x).prod() - 1)
                
                # Convert index back to timestamp (End of Month) to match returns_matrix
                rf_monthly.index = rf_monthly.index.to_timestamp('M')
                
                # Reindex to match returns_matrix
                self.rf_series = rf_monthly.reindex(self.returns_matrix.index).fillna(0.0) # Fill missing with 0? or ffill?
                # Usually RF shouldn't be missing. If it is, 0 is a safe fallback for "excess return".
                
                print(f"Risk Free Rate Loaded (CRSP). Shape: {self.rf_series.shape}")
            else:
                print(f"WARNING: RF file configured but not found: {rf_path}")
        
        print(f"Data Loaded (CRSP): {len(selected_tickers)} assets. Returns shape: {self.returns_matrix.shape}")

    def _run_strategy_yfinance(self):
        """
        Stratégie YFinance avec respect de la configuration (n_stocks, sectors).
        """
        # 1. Chemins : Utilisation directe de la config ou fallback intelligent
        data_dir = Path(self.config.get('data_dir', 'data/yfinance'))
        input_file = self.config.get('input_file', 'financial_universe_clean.parquet')
        rf_file = self.config.get('rf_file', 'risk_free.parquet')
        
        yf_path = data_dir / input_file
        rf_path = data_dir / rf_file
        
        # Fallback si chemin relatif incorrect (cas fréquent Streamlit vs Notebook)
        if not yf_path.exists():
            # Try resolving relative to current working directory if it starts with ..
            if str(data_dir).startswith('..'):
                # We are likely in notebooks/ or pages/ and data_dir is ../data/...
                # But if we are in root, ../data might be wrong.
                # Let's try absolute path resolution based on project root assumption
                try:
                    # Assuming we are in notebooks/ or pages/ and project root is one level up
                    possible_roots = [Path('.'), Path('..'), Path('../..')]
                    for root in possible_roots:
                        candidate = root / str(data_dir).replace('../', '') / input_file
                        if candidate.exists():
                            yf_path = candidate
                            rf_path = root / str(data_dir).replace('../', '') / rf_file
                            break
                except:
                    pass

        print(f"Loading YFinance Data from: {yf_path}")
        
        if not yf_path.exists():
            raise FileNotFoundError(f"YFinance data not found at: {yf_path}")
        
        # 2. Chargement Brut
        df = pl.read_parquet(yf_path)
        
        # Normalisation
        if 'Date' in df.columns: df = df.rename({'Date': 'date'})
        if 'sector' in df.columns and 'Sector' not in df.columns: df = df.rename({'sector': 'Sector'})

        # =================================================================
        # 3. FILTRAGE
        # =================================================================
        
        # A. Liste de tous les tickers
        all_tickers = df.select('Ticker').unique().to_series().to_list()
        
        # B. Filtre Sectoriel (si demandé dans config)
        requested_sectors = self.config.get('sectors', None)
        if requested_sectors is not None:
            temp_map = df.select(['Ticker', 'Sector']).unique().to_pandas().set_index('Ticker')['Sector'].to_dict()
            all_tickers = [t for t in all_tickers if temp_map.get(t, 'Other') in requested_sectors]

        # C. Sélection (Top Market Cap ou Random)
        selection_method = self.config.get('selection_method', 'top_market_cap') # 'top_market_cap' or 'random'
        n_stocks = self.config.get('n_stocks', len(all_tickers))
        
        if n_stocks < len(all_tickers):
            if selection_method == 'random':
                print(f"🎲 Randomly selecting {n_stocks} stocks.")
                if 'random_seed' in self.config:
                    np.random.seed(self.config['random_seed'])
                selected_tickers = np.random.choice(all_tickers, size=n_stocks, replace=False).tolist()
            else:
                # Top Market Cap (Default)
                try:
                    cap_stats = df.filter(pl.col('Ticker').is_in(all_tickers)) \
                                  .group_by('Ticker') \
                                  .agg(pl.col('DlyCap').mean().alias('AvgCap')) \
                                  .sort('AvgCap', descending=True) \
                                  .head(n_stocks)
                    
                    selected_tickers = cap_stats.select('Ticker').to_series().to_list()
                    print(f"🔝 Top {n_stocks} stocks selected by Market Cap.")
                except Exception as e:
                    print(f"⚠️ Market Cap sort failed ({e}), falling back to random sampling.")
                    selected_tickers = np.random.choice(all_tickers, size=n_stocks, replace=False).tolist()
        else:
            selected_tickers = all_tickers
            
        # D. Application du filtre
        df = df.filter(pl.col('Ticker').is_in(selected_tickers))
        
        # --- Extract Maps ---
            # --- Extract Maps ---
        try:
            # Sector Map
            if 'Sector' in df.columns:
                self.sector_map = df.select(['Ticker', 'Sector']).unique().to_pandas().set_index('Ticker')['Sector'].to_dict()
            elif 'sector' in df.columns:
                 self.sector_map = df.select(['Ticker', 'sector']).unique().to_pandas().set_index('Ticker')['sector'].to_dict()
            
            # Name Map
            # Try multiple columns for Name
            name_col = None
            for col in ['shortName', 'longName', 'SecurityNm', 'ShortName', 'LongName', 'name', 'Name']:
                if col in df.columns:
                    name_col = col
                    break
            
            if name_col:
                print(f"✅ Found Name Column: {name_col}")
                self.name_map = df.select(['Ticker', name_col]).unique().to_pandas().set_index('Ticker')[name_col].to_dict()
            else:
                print("⚠️ No Name Column found (checked shortName, longName, SecurityNm, etc.). Using Ticker.")
                self.name_map = {t: t for t in selected_tickers}
                
        except Exception as e:
            print(f"⚠️ Error extracting metadata maps: {e}")
            self.name_map = {t: t for t in selected_tickers}

        # =================================================================
        # 4. Préparation des Matrices
        # =================================================================
        # Ensure Sector is included if present, as it was in the original code
        cols_to_select = ['date', 'Ticker', 'DlyRet', 'DlyCap']
        if 'Sector' in df.columns:
            cols_to_select.append('Sector')
            
        df_pd = df.select(cols_to_select).to_pandas()
        df_pd['date'] = pd.to_datetime(df_pd['date'])
        
        # Gestion de la Fréquence (Daily vs Monthly)
        frequency = self.config.get('frequency', 'monthly').lower()
        
        if frequency == 'daily':
            # Returns Daily
            self.returns_matrix = df_pd.pivot(index='date', columns='Ticker', values='DlyRet').sort_index()
            self.market_caps_matrix = df_pd.pivot(index='date', columns='Ticker', values='DlyCap').sort_index()
            print("📅 Frequency: DAILY")
        else:
            # Returns Monthly (Aggregation)
            df_pd['year_month'] = df_pd['date'].dt.to_period('M')
            
            monthly_returns = df_pd.groupby(['Ticker', 'year_month'])['DlyRet'].apply(lambda x: (1 + x).prod() - 1).reset_index()
            monthly_returns['date'] = monthly_returns['year_month'].dt.to_timestamp('M')
            self.returns_matrix = monthly_returns.pivot(index='date', columns='Ticker', values='DlyRet').sort_index()
            
            monthly_caps = df_pd.groupby(['Ticker', 'year_month'])['DlyCap'].last().reset_index()
            monthly_caps['date'] = monthly_caps['year_month'].dt.to_timestamp('M')
            self.market_caps_matrix = monthly_caps.pivot(index='date', columns='Ticker', values='DlyCap').sort_index()
            print("📅 Frequency: MONTHLY (Aggregated)")
        
        # =================================================================
        # 5. Filtrage Temporel (Test Period + Lookback Buffer)
        # =================================================================
        test_start = pd.to_datetime(self.config['test_start'])
        test_end = pd.to_datetime(self.config['test_end'])
        lookback = self.config.get('lookback_months', 24)
        
        # Buffer: Start date must be earlier to allow lookback calculation at t=0
        # We add ~2 months extra buffer to be safe
        buffer_start = test_start - pd.DateOffset(months=lookback + 2)
        
        if self.returns_matrix is not None:
            self.returns_matrix = self.returns_matrix.loc[buffer_start:test_end]
            
        if self.market_caps_matrix is not None:
            self.market_caps_matrix = self.market_caps_matrix.loc[buffer_start:test_end]
            
        print(f"📅 Date Range Loaded: {self.returns_matrix.index[0].date()} to {self.returns_matrix.index[-1].date()} (Includes {lookback}m lookback)")
        
        sector_df = df_pd.groupby('Ticker')['Sector'].last()
        self.sector_map = sector_df.to_dict()
        # self.name_map = {t: t for t in selected_tickers}  <-- REMOVED: This was overwriting the extracted map!
        
        # Load RF
        if rf_path.exists():
            rf_df = pl.read_parquet(rf_path).to_pandas()
            if 'Date' in rf_df.columns: rf_df = rf_df.rename(columns={'Date': 'date'})
            rf_df['date'] = pd.to_datetime(rf_df['date'])
            rf_df = rf_df.set_index('date').sort_index()
            
            # Resample to match returns frequency
            if frequency == 'daily':
                self.rf_series = rf_df['RiskFree'].reindex(self.returns_matrix.index).fillna(method='ffill')
            else:
                rf_df['year_month'] = rf_df.index.to_period('M')
                rf_monthly = rf_df.groupby('year_month')['RiskFree'].apply(lambda x: (1 + x).prod() - 1)
                rf_monthly.index = rf_monthly.index.to_timestamp('M')
                self.rf_series = rf_monthly.reindex(self.returns_matrix.index).fillna(0.0)
        else:
            self.rf_series = pd.Series(0.0, index=self.returns_matrix.index)

        print(f"✅ Data Loaded (YFinance): {len(self.returns_matrix.columns)} assets. Returns shape: {self.returns_matrix.shape}")

    def _create_sector_map(self, df):
        """
        Creates Ticker -> Sector mapping using SIC codes.
        """
        if 'SICCD' not in df.columns:
            return {}
            
        # Extract unique Ticker-SIC pairs
        # We take the most recent SIC for each ticker (if it changed)
        # For simplicity, we take the mode or last known.
        # Here we take unique pairs.
        ticker_sic = df.select(['Ticker', 'SICCD']).unique(subset=['Ticker']).to_pandas()
        
        mapping = {}
        for _, row in ticker_sic.iterrows():
            mapping[row['Ticker']] = self._get_sector_from_sic(row['SICCD'])
            
        return mapping

    def _create_name_map(self, df):
        """
        Creates Ticker -> SecurityNm mapping.
        """
        if 'SecurityNm' not in df.columns:
            return {}
            
        # Extract unique Ticker-Name pairs (take last known)
        ticker_name = df.select(['Ticker', 'SecurityNm']).unique(subset=['Ticker']).to_pandas()
        
        mapping = {}
        for _, row in ticker_name.iterrows():
            # Clean Name: "IAMGOLD CORP; COM NONE; CONS" -> "IAMGOLD CORP"
            raw_name = row['SecurityNm']
            clean_name = raw_name.split(';')[0].strip()
            mapping[row['Ticker']] = clean_name
            
        return mapping

    def _get_sector_from_sic(self, sic_code):
        try:
            sic = int(sic_code)
        except:
            return 'Other'
            
        # Energy (1300-1399, 2900-2999)
        if (1300 <= sic <= 1399) or (2900 <= sic <= 2999): return 'Energy'
        
        # Materials (1000-1299, 1400-1499, 2600-2699, 2800-2829, 2840-2899, 3200-3299)
        if (1000 <= sic <= 1499) or (2600 <= sic <= 2699) or (2800 <= sic <= 2829) or (2840 <= sic <= 2899) or (3200 <= sic <= 3299): return 'Materials'
        
        # Industrials (1500-1799, 3400-3569, 3580-3599, 3700-3799, 4000-4799)
        if (1500 <= sic <= 1799) or (3400 <= sic <= 3569) or (3580 <= sic <= 3599) or (3700 <= sic <= 3799) or (4000 <= sic <= 4799): return 'Industrials'
        
        # Consumer Discretionary (2200-2399, 2500-2599, 3000-3199, 3630-3659, 3710-3719, 5200-5399, 5500-5999, 7000-7999)
        if (2200 <= sic <= 2399) or (2500 <= sic <= 2599) or (3000 <= sic <= 3199) or (3630 <= sic <= 3659) or (3710 <= sic <= 3719) or (5200 <= sic <= 5399) or (5500 <= sic <= 5999) or (7000 <= sic <= 7999): return 'Consumer Discretionary'
        
        # Consumer Staples (2000-2199, 2840-2844, 5400-5499)
        if (2000 <= sic <= 2199) or (5400 <= sic <= 5499): return 'Consumer Staples'
        
        # Health Care (2830-2836, 3840-3851, 8000-8099)
        if (2830 <= sic <= 2836) or (3840 <= sic <= 3851) or (8000 <= sic <= 8099): return 'Health Care'
        
        # Financials (6000-6499, 6700-6799)
        if (6000 <= sic <= 6499) or (6700 <= sic <= 6799): return 'Financials'
        
        # Information Technology (3570-3579, 3600-3629, 3660-3699, 7370-7379)
        if (3570 <= sic <= 3579) or (3600 <= sic <= 3629) or (3660 <= sic <= 3699) or (7370 <= sic <= 7379): return 'Information Technology'
        
        # Communication Services (4800-4899)
        if (4800 <= sic <= 4899): return 'Communication Services'
        
        # Utilities (4900-4999)
        if (4900 <= sic <= 4999): return 'Utilities'
        
        # Real Estate (6500-6599)
        if (6500 <= sic <= 6599): return 'Real Estate'
        
        return 'Other'

    def get_matrices(self):
        """
        Returns (returns_matrix, market_caps_matrix, sector_map, name_map)
        """
        if self.returns_matrix is None:
            self.load_data()
        return self.returns_matrix, self.market_caps_matrix, self.sector_map, self.name_map

    def get_rf_data(self):
        """
        Returns the Risk Free Rate Series (aligned with returns).
        """
        if self.returns_matrix is None:
            self.load_data()
            
        if self.rf_series is None:
            # Return zeros if not available
            return pd.Series(0.0, index=self.returns_matrix.index)
            
        return self.rf_series
