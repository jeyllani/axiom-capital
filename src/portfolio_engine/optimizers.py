from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import riskfolio as rp
import scipy.optimize as sco
import warnings

class BaseOptimizer(ABC):
    """
    Abstract base class for portfolio optimization.
    """
    
    def __init__(self, cov_method='ledoit', max_turnover=None, risk_aversion=None, long_only=True, solver='CLARABEL'):
        self.cov_method = cov_method
        self.max_turnover = max_turnover
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.solver = solver
        self.ridge_epsilon = 1e-8
        self.rf = 0.0

    def _apply_sector_constraints(self, port, sector_constraints, sector_map, returns):
        """
        Gère PROPREMENT la distinction Égalité vs Inégalité pour le solveur.
        """
        if sector_constraints is None or sector_map is None:
            return

        # --- NOUVEAU : SANITY CHECK (Validation de Faisabilité) ---
        total_min = 0.0
        for s, (min_w, max_w) in sector_constraints.items():
            if min_w > max_w + 1e-6:
                raise ValueError(f"Contrainte Impossible sur {s}: Min ({min_w:.1%}) > Max ({max_w:.1%})")
            total_min += min_w
        
        if total_min > 1.0 + 1e-4:
            raise ValueError(f"Contraintes Impossibles: La somme des Min ({total_min:.1%}) dépasse 100%")
        # -----------------------------------------------------------

        # Mapping Assets -> Secteurs
        assets = returns.columns.tolist()
        asset_classes = pd.DataFrame({'Assets': assets})
        asset_classes['Sector'] = asset_classes['Assets'].map(sector_map).fillna('Other')
        asset_classes = asset_classes.set_index('Assets')
        
        n_assets = len(assets)
        epsilon = 1e-5 
        
        # Listes séparées (Inégalités vs Égalités)
        A_ineq, B_ineq = [], []
        A_eq, B_eq = [], []

        for sector, (min_w, max_w) in sector_constraints.items():
            sector_assets = asset_classes[asset_classes['Sector'] == sector].index
            asset_indices = [assets.index(a) for a in sector_assets if a in assets]
            
            if not asset_indices: continue
            
            # --- CAS 1 : C'est une ÉGALITÉ (Min ~= Max) ---
            if abs(max_w - min_w) < epsilon:
                row = np.zeros(n_assets)
                row[asset_indices] = 1.0
                A_eq.append(row)
                B_eq.append(min_w)
                
            # --- CAS 2 : C'est une PLAGE (Range) ---
            else:
                # Min Limit
                if min_w > epsilon:
                    row = np.zeros(n_assets)
                    row[asset_indices] = 1.0
                    A_ineq.append(row)
                    B_ineq.append(min_w)
                # Max Limit (Négatif pour forme A @ w >= B)
                if max_w < 1.0 - epsilon:
                    row = np.zeros(n_assets)
                    row[asset_indices] = -1.0
                    A_ineq.append(row)
                    B_ineq.append(-max_w)

        # Injection sélective dans Riskfolio
        if A_ineq:
            port.ainequality = np.array(A_ineq)
            port.binequality = np.array(B_ineq).reshape(-1, 1)
        else:
            port.ainequality = None; port.binequality = None

        if A_eq: # FIX: Utilisation du canal d'égalité natif
            port.aequality = np.array(A_eq)
            port.bequality = np.array(B_eq).reshape(-1, 1)
        else:
            port.aequality = None; port.bequality = None

    def _apply_turnover_constraint(self, weights, prev_weights, returns):
        """
        Applies turnover constraint via post-processing (linear scaling).
        Used as a fallback if native constraint is ignored or for specific optimizers.
        """
        if self.max_turnover is not None and prev_weights is not None:
            # Align
            current_w = weights.reindex(returns.columns, fill_value=0.0)
            prev_w_aligned = prev_weights.reindex(returns.columns, fill_value=0.0)
            
            # Calculate difference
            diff = current_w - prev_w_aligned
            turnover = diff.abs().sum()
            
            if turnover > self.max_turnover + 1e-6:
                # Scale trade
                scalar = self.max_turnover / turnover
                # Linear interpolation preserves sum=1
                final_weights = prev_w_aligned + diff * scalar
                return final_weights
        
        return weights

    def _apply_sector_fallback(self, weights, sector_constraints, sector_map):
        """
        Algorithme Robuste Stabilisé.
        Priorité absolue au respect des bornes Max/Egalité.
        """
        if sector_constraints is None or sector_map is None: return weights
        w = weights.copy()
        assets = w.index.tolist()
        asset_sector = {a: sector_map.get(a, 'Other') for a in assets}
        
        # 1. Normalisation initiale
        if w.sum() > 0: w = w / w.sum()
        
        # 2. Boucle de Convergence
        for _ in range(100): # Increased iterations for better convergence
            dirty = False
            
            # A. D'abord on booste les MIN (car le clipping Max va réduire après)
            for sector, (min_w, max_w) in sector_constraints.items():
                sec_assets = [a for a in assets if asset_sector[a] == sector]
                sw = w[sec_assets].sum()
                if min_w > 0.0 and sw < min_w - 1e-4:
                    scaler = min_w / sw if sw > 1e-6 else 0.0
                    if scaler > 0:
                        w[sec_assets] *= scaler
                        dirty = True
                    else:
                        # Injection forcée si 0
                        if len(sec_assets) > 0:
                            w[sec_assets] = min_w / len(sec_assets)
                            dirty = True

            # B. Ensuite on RENORMALISE globalement pour éviter l'explosion
            if w.sum() > 0: w = w / w.sum()

            # C. Enfin on CLIPE les MAX (C'est la contrainte la plus dure)
            # On le fait APRÈS la renormalisation pour être sûr de ne pas redépasser
            for sector, (min_w, max_w) in sector_constraints.items():
                sec_assets = [a for a in assets if asset_sector[a] == sector]
                sw = w[sec_assets].sum()
                if max_w < 1.0 and sw > max_w + 1e-4:
                    w[sec_assets] *= (max_w / sw)
                    dirty = True # On marque dirty, mais on ne re-normalise pas tout de suite pour ne pas casser ce clip
            
            if not dirty:
                break
        
        # 3. Normalisation Finale Intelligente (Smart Renormalization)
        # Redistribuer le poids manquant UNIQUEMENT sur les secteurs non contraints
        total = w.sum()
        missing = 1.0 - total
        
        if missing > 1e-6:
            # Identifier les actifs éligibles (ceux dont le secteur n'est pas au Max)
            eligible_assets = []
            for a in assets:
                sec = asset_sector[a]
                sec_max = sector_constraints.get(sec, (0.0, 1.0))[1]
                sec_current = w[[x for x in assets if asset_sector[x] == sec]].sum()
                
                # On ne donne du poids qu'aux secteurs qui ont de la marge
                if sec_current < sec_max - 1e-4:
                    eligible_assets.append(a)
            
            if eligible_assets:
                w_eligible = w[eligible_assets]
                if w_eligible.sum() > 0:
                    # Distribution proportionnelle au poids actuel
                    w[eligible_assets] += missing * (w[eligible_assets] / w_eligible.sum())
                else:
                    # Distribution équitable si poids nuls
                    w[eligible_assets] += missing / len(eligible_assets)
            else:
                # Si aucun actif éligible (tous les secteurs sont au max ?), on normalise brutalement
                if total > 0: w = w / total

        return w

    def get_risk_metrics(self, weights, returns, rm='MV'):
        """ Retourne les contributions au risque (Absolue & Relative) """
        try:
            cov = returns.cov()
            w = weights.values
            port_var = w.T @ cov.values @ w
            port_vol = np.sqrt(port_var)
            mrc = (cov.values @ w) / port_vol
            rc = w * mrc
            rrc = rc / port_vol
            
            return pd.DataFrame({
                'Weights': weights,
                'Risk Contribution': rc,
                'Relative Risk Contribution': rrc
            }, index=weights.index)
        except Exception as e:
            return None

    def _safe_optimize(self, port, **opt_kwargs):
        """ Wrapper de sécurité pour l'appel au solveur """
        try:
            w = port.optimization(**opt_kwargs)
        except Exception as e:
            w = None
        
        # Si échec, tentative de relaxation
        if w is None:
            print("⚠️ Optimisation stricte échouée. Tentative sans contraintes sectorielles...")
            port.ainequality = None
            port.aequality = None
            port.binequality = None
            port.bequality = None
            try:
                w = port.optimization(**opt_kwargs)
            except Exception as e:
                print(f"⚠️ Optimization Error (Relaxed): {e}")
                w = None
                
        if w is None:
            raise ValueError("Optimisation impossible. Vérifiez les contraintes (Somme Min > 100% ?)")
            
        return w

    @abstractmethod
    def optimize(self, returns, prev_weights=None, **kwargs):
        """
        Calculate optimal weights.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns for covariance/mean estimation.
        prev_weights : pd.Series, optional
            Previous weights (required for turnover constraints).
            
        Returns:
        --------
        pd.Series : Optimal weights.
        """
        pass

class MinVarianceOptimizer(BaseOptimizer):
    """
    Minimum Variance Optimizer using Riskfolio-Lib.
    Uses Ledoit-Wolf + Ridge regularization for robustness.
    """
    
    def __init__(self, cov_method='ledoit', long_only=True, max_turnover=None, risk_aversion=None, solver='CLARABEL'):
        """
        Parameters:
        -----------
        cov_method : str
            Method for covariance estimation: 'ledoit' (recommended), 'hist', 'oas', etc.
        long_only : bool
            If True, weights are constrained to [0, 1].
        max_turnover : float, optional
            Maximum allowed turnover (0.0 to 1.0) relative to prev_weights.
            If None, no turnover constraint is applied.
        risk_aversion : float, optional
            Risk aversion parameter 'l' for Utility maximization.
            If None (default), uses 'MinRisk' objective (ignores expected returns).
            If set (e.g. 2.0), uses 'Utility' objective.
        solver : str, optional
            Solver to use (e.g. 'ECOS', 'CLARABEL', 'OSQP'). Default 'CLARABEL'.
            If None, Riskfolio chooses automatically.
        """
        self.cov_method = cov_method
        self.long_only = long_only
        self.max_turnover = max_turnover
        self.risk_aversion = risk_aversion
        self.solver = solver
        self.ridge_epsilon = 1e-8 # Default Ridge for numerical stability
        self.rf = 0.0
        
    def set_ridge(self, epsilon):
        """Set the ridge epsilon for covariance regularization."""
        self.ridge_epsilon = epsilon
        
    def set_rf(self, rf):
        """Set Risk Free Rate."""
        self.rf = rf
        
    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        """
        Calculate Minimum Variance weights.
        """
        # Clean returns
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 1. Initialize Riskfolio Portfolio
        port = rp.Portfolio(returns=returns_clean)
        port.budget = 1.0 # Explicitly set budget to 100%
        
        # 2. Estimate Stats with Ledoit-Wolf (robust)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                port.assets_stats(method_mu='hist', method_cov=self.cov_method)
                
                # [FIX] Custom Mu Support
                if 'custom_mu' in kwargs and kwargs['custom_mu'] is not None:
                    c_mu = kwargs['custom_mu'].reindex(returns.columns).fillna(0)
                    port.mu = c_mu.to_frame().T if isinstance(c_mu, pd.Series) else c_mu
            except Exception as e:
                # Ultimate fallback: use Ledoit
                port.assets_stats(method_mu='hist', method_cov='ledoit')
        
        # Apply Sector Constraints (AFTER stats)
        self._apply_sector_constraints(port, sector_constraints, sector_map, returns)
        
        # 3. Ridge Regularization (always apply for stability)
        if self.ridge_epsilon > 0:
            port.cov = port.cov + np.eye(len(port.cov)) * self.ridge_epsilon
        
        # 4. Optimize
        # Determine objective based on risk_aversion
        if self.risk_aversion is None:
            obj = 'MinRisk'
            l_param = 0 # Ignored for MinRisk
        else:
            obj = 'Utility'
            l_param = self.risk_aversion
        
        # [FIX] Set Short Selling via property, not argument
        port.sht = not self.long_only
        
        # Prepare kwargs
        opt_kwargs = {
            'model': 'Classic',
            'rm': 'MV',
            'obj': obj,
            'rf': self.rf,
            'l': l_param,
            'hist': True
        }
        
        # Set solver explicitly
        port.solver = self.solver
        
        # [FIX] Native Turnover Constraint
        if self.max_turnover is not None and prev_weights is not None:
            # Align prev_weights to current assets
            w0 = prev_weights.reindex(returns.columns, fill_value=0.0)
            port.w0 = w0.to_frame() # Riskfolio expects DataFrame or Series
            port.turnover = self.max_turnover

        w = self._safe_optimize(port, **opt_kwargs)
            
        # 6. Process Result (Clean & Normalize FIRST)
        weights = w.squeeze()
        
        # [FIX] Safety Check for Zero Weights (Check RAW output first)
        if weights.abs().sum() < 1e-6:
            raise ValueError("Optimizer returned zero weights.")

        # Clean small weights (use ABS for L/S support)
        weights = weights[weights.abs() > 1e-6]
        
        # Renormalize
        weights = weights / weights.sum()
        
        # [FALLBACK] Apply Turnover Constraint (Post-Processing)
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)
        
        # [FALLBACK] Apply Sector Constraint (Post-Processing)
        # Because v7.0.1 solvers often ignore ainequality
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)
        
        return weights


class MaxSharpeOptimizer(BaseOptimizer):
    """
    Maximum Sharpe Ratio Optimizer (Tangency Portfolio).
    Maximizes (Return - rf) / Risk.
    """
    
    def __init__(self, cov_method='ledoit', long_only=True, max_turnover=None, solver='CLARABEL'):
        self.cov_method = cov_method
        self.long_only = long_only
        self.max_turnover = max_turnover
        self.solver = solver
        self.ridge_epsilon = 1e-8
        self.rf = 0.0
        
    def set_rf(self, rf):
        self.rf = rf
        
    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        # Clean returns
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        port = rp.Portfolio(returns=returns_clean)
        port.budget = 1.0 # Explicitly set budget to 100%
        
        # Apply Sector Constraints
        self._apply_sector_constraints(port, sector_constraints, sector_map, returns)
        
        # Estimate Stats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                port.assets_stats(method_mu='ewma1', method_cov=self.cov_method)
                
                # [FIX] Custom Mu Support
                if 'custom_mu' in kwargs and kwargs['custom_mu'] is not None:
                    c_mu = kwargs['custom_mu'].reindex(returns.columns).fillna(0)
                    port.mu = c_mu.to_frame().T if isinstance(c_mu, pd.Series) else c_mu
            except:
                port.assets_stats(method_mu='hist', method_cov='ledoit')
                
        # Ridge
        if self.ridge_epsilon > 0:
            port.cov = port.cov + np.eye(len(port.cov)) * self.ridge_epsilon
            
        # Optimize for Max Sharpe
            # Optimize for Max Sharpe
            # [FIX] Set Short Selling via property
            port.sht = not self.long_only
            
            # Set solver
            port.solver = self.solver
            
            # [FIX] Native Turnover Constraint
            if self.max_turnover is not None and prev_weights is not None:
                w0 = prev_weights.reindex(returns.columns, fill_value=0.0)
                port.w0 = w0.to_frame()
                port.turnover = self.max_turnover
            
            # Get RF from kwargs (dynamic) or default to self.rf
            rf = kwargs.get('rf', self.rf)

            opt_kwargs = {
                'model': 'Classic',
                'rm': 'MV',
                'obj': 'Sharpe', # Maximize Sharpe Ratio
                'rf': rf,
                'l': 0,
                'hist': True
            }

            w = self._safe_optimize(port, **opt_kwargs)
            
        # Process Result (Clean & Normalize)
        weights = w.squeeze()
        
        # [FIX] Safety Check
        if weights.abs().sum() < 1e-6:
            raise ValueError("MaxSharpe returned zero weights.")
        
        # Clean small weights (ABS for L/S)
        weights = weights[weights.abs() > 1e-6]
                
        weights = weights / weights.sum()
        
        # [FALLBACK] Apply Turnover Constraint (Post-Processing)
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)
        
        # [FALLBACK] Apply Sector Constraint (Post-Processing)
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)
        
        return weights


class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity Optimizer (Equal Risk Contribution).
    """
    def __init__(self, cov_method='ledoit', max_turnover=None, risk_aversion=None, long_only=True, solver='CLARABEL'):
        super().__init__(cov_method, max_turnover, risk_aversion, long_only, solver=solver)
        
    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, risk_budget=None, **kwargs):
        # 1. Init & Stats (Standard)
        port = rp.Portfolio(returns=returns)
        port.budget = 1.0
        port.sht = not self.long_only
        self._apply_sector_constraints(port, sector_constraints, sector_map, returns)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try: port.assets_stats(method_mu='hist', method_cov=self.cov_method)
            except: port.assets_stats(method_mu='hist', method_cov='ledoit')

        # 2. Gestion du Budget de Risque (NOUVEAU)
        b = None
        if risk_budget is not None:
            assets = returns.columns.tolist()
            # Conversion du dict/series en vecteur aligné
            if isinstance(risk_budget, dict):
                b_vec = [risk_budget.get(a, 0.0) for a in assets]
            elif isinstance(risk_budget, pd.Series):
                b_vec = risk_budget.reindex(assets).fillna(0.0).values
            else:
                b_vec = None
            
            if b_vec is not None:
                b = np.array(b_vec)
                if b.sum() > 0: b = b / b.sum() # Normalisation
                b = b.reshape(-1, 1)

        # 3. Optimisation avec gestion d'erreur (ROBUSTESSE CONSERVÉE)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # On passe 'b' ici
                w = port.rp_optimization(model='Classic', rm='MV', rf=0, b=b, hist=True)
        except Exception:
            w = None
            
        # Relaxation (Si échec)
        if w is None:
            print("⚠️ Risk Parity strict échoué. Tentative sans contraintes sectorielles...")
            port.ainequality = None; port.aequality = None; port.binequality = None; port.bequality = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = port.rp_optimization(model='Classic', rm='MV', rf=0, b=b, hist=True)
            except: w = None

        if w is None: raise ValueError("Risk Parity Optimization failed to converge")
            
        # 4. Post-Process (Identique à avant)
        weights = w.squeeze()
        if weights.abs().sum() < 1e-6: raise ValueError("Risk Parity returned zero weights.")
        weights = weights[weights.abs() > 1e-6]
        weights = weights / weights.sum()
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)
        
        return weights


class MinCVaROptimizer(BaseOptimizer):
    """
    Minimize Conditional Value at Risk (CVaR).
    """
    def __init__(self, cov_method='ledoit', max_turnover=None, long_only=True, solver='CLARABEL'):
        super().__init__(cov_method, max_turnover, risk_aversion=None, long_only=long_only, solver=solver)

    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        port = rp.Portfolio(returns=returns)
        port.budget = 1.0
        port.sht = not self.long_only
        
        # Apply Sector Constraints
        self._apply_sector_constraints(port, sector_constraints, sector_map, returns)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                port.assets_stats(method_mu='hist', method_cov=self.cov_method)
            except:
                port.assets_stats(method_mu='hist', method_cov='ledoit')
                
        # [FIX] Native Turnover Constraint
        if self.max_turnover is not None and prev_weights is not None:
            w0 = prev_weights.reindex(returns.columns, fill_value=0.0)
            port.w0 = w0.to_frame()
            port.turnover = self.max_turnover

        # Set solver explicitly
        port.solver = self.solver

        # Get RF from kwargs (dynamic) or default to 0
        rf = kwargs.get('rf', 0.0)
        # Get Alpha (Confidence Level) from kwargs or default to 0.05 (95%)
        alpha = kwargs.get('alpha', 0.05)
        # Set Alpha (Confidence Level)
        port.alpha = alpha

        opt_kwargs = {
            'model': 'Classic',
            'rm': 'CVaR', # CVaR Risk Measure
            'obj': 'MinRisk',
            'rf': rf,
            'l': 0,
            'hist': True
        }
        
        w = self._safe_optimize(port, **opt_kwargs)
            
        # Process Result (Clean & Normalize)
        weights = w.squeeze()
        if weights.abs().sum() < 1e-6:
            raise ValueError("MinCVaR returned zero weights.")
            
        weights = weights[weights.abs() > 1e-6]
        weights = weights / weights.sum()

        # [FALLBACK] Apply Turnover Constraint (Post-Processing)
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)

        # [FALLBACK] Apply Sector Constraint (Post-Processing)
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)

        return weights



class MinTrackingErrorOptimizer(BaseOptimizer):
    """
    Minimize Tracking Error relative to a Benchmark.
    Uses scipy.optimize for Quadratic Programming.
    """
    def __init__(self, benchmark_weights, cov_method='ledoit', max_turnover=None, long_only=True, solver='CLARABEL'):
        super().__init__(cov_method, max_turnover, risk_aversion=None, long_only=long_only, solver=solver)
        self.benchmark_weights = benchmark_weights
        
    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        # Create Riskfolio Portfolio for Covariance Calculation
        port = rp.Portfolio(returns=returns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                port.assets_stats(method_mu='hist', method_cov=self.cov_method)
            except:
                port.assets_stats(method_mu='hist', method_cov='ledoit')
        
        # Get Covariance Matrix
        cov = port.cov
        
        # Align benchmark weights
        assets = returns.columns.tolist()
        b = self.benchmark_weights.reindex(assets).fillna(0.0)
        if b.sum() > 0: b = b / b.sum()
        
        # Setup QP for Min Tracking Error
        # Min (w-b)' S (w-b)
        # Equivalent to Min 0.5 w'Sw - b'Sw (ignoring constant term)
        
        n = len(assets)
        P = cov.values
        q = -P @ b.values
        
        # Constraints: sum(w) = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds: 0 <= w <= 1 (Long Only)
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        # Initial guess
        x0 = b.values
        
        # Objective
        def objective(w):
            return 0.5 * w @ P @ w + q @ w
            
        try:
            res = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            if res.success:
                weights = pd.Series(res.x, index=assets)
            else:
                print(f'TE Opt failed: {res.message}')
                weights = b
        except Exception as e:
            print(f'TE Opt error: {e}')
            weights = b

        # Apply Turnover Constraint (Post-Processing)
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)
        
        # Apply Sector Constraint (Post-Processing)
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)
        
        return weights


class TargetVolOptimizer(BaseOptimizer):
    """
    Maximize Return subject to Target Volatility.
    """
    def __init__(self, target_vol=0.10, cov_method='ledoit', max_turnover=None, long_only=True, frequency=12, solver='CLARABEL'):
        super().__init__(cov_method, max_turnover, risk_aversion=None, long_only=long_only)
        self.target_vol = target_vol
        self.frequency = frequency
        self.solver = solver

    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        port = rp.Portfolio(returns=returns)
        
        # Apply Sector Constraints
        self._apply_sector_constraints(port, sector_constraints, sector_map, returns)
        
        port.budget = 1.0
        port.sht = not self.long_only
        
        # Set Volatility Constraint
        # Convert Annual Target to Period Target using frequency
        daily_target = self.target_vol / (self.frequency**0.5)
        port.upperdev = daily_target
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                port.assets_stats(method_mu='hist', method_cov=self.cov_method)
            except:
                port.assets_stats(method_mu='hist', method_cov='ledoit')
                
        # Native Turnover Constraint
        if self.max_turnover is not None and prev_weights is not None:
            w0 = prev_weights.reindex(returns.columns, fill_value=0.0)
            port.w0 = w0.to_frame()
            port.turnover = self.max_turnover

        # Set solver
        port.solver = self.solver
        
        # Get RF from kwargs (dynamic) or default to 0
        rf = kwargs.get('rf', 0.0)

        opt_kwargs = {
            'model': 'Classic',
            'rm': 'MV',
            'obj': 'MaxRet', # Maximize Return subject to constraints
            'rf': rf,
            'l': 0,
            'hist': True
        }

        try:
            w = self._safe_optimize(port, **opt_kwargs)
        except ValueError:
            w = None
            
        if w is None:
            # Fallback if MaxRet fails (often due to infeasible constraints)
            # This usually means Target Vol < Min Vol
            print(f"TargetVol ({self.target_vol:.1%}) infeasible. Fallback to MinRisk (MVP).")
            
            # CRITICAL FIX: Remove the volatility constraint for the fallback
            port.upperdev = None
            
            opt_kwargs['obj'] = 'MinRisk'
            # Try MinRisk
            try:
                w = self._safe_optimize(port, **opt_kwargs)
            except ValueError:
                raise ValueError("TargetVol failed (both MaxRet and MinRisk).")
            
        # Process Result (Clean & Normalize)
        weights = w.squeeze()
        if weights.abs().sum() < 1e-6:
            raise ValueError("TargetVol returned zero weights.")
            
        weights = weights[weights.abs() > 1e-6]
        weights = weights / weights.sum()

        # [FALLBACK] Apply Turnover Constraint (Post-Processing)
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)

        # [FALLBACK] Apply Sector Constraint (Post-Processing)
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)

        # Ensure full alignment with returns
        weights = weights.reindex(returns.columns, fill_value=0.0)

        return weights


class MaxRetOptimizer(BaseOptimizer):
    """
    Maximize Expected Return (High Risk).
    """
    def __init__(self, cov_method='ledoit', max_turnover=None, long_only=True, solver='ECOS'):
        super().__init__(cov_method, max_turnover, risk_aversion=None, long_only=long_only)
        self.solver = solver

    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        port = rp.Portfolio(returns=returns)
        
        # Apply Sector Constraints
        self._apply_sector_constraints(port, sector_constraints, sector_map, returns)
        
        port.budget = 1.0
        port.sht = not self.long_only
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                port.assets_stats(method_mu='hist', method_cov=self.cov_method)
            except:
                port.assets_stats(method_mu='hist', method_cov='ledoit')
                
        # Set solver
        port.solver = self.solver
        
        # [FIX] Native Turnover Constraint
        if self.max_turnover is not None and prev_weights is not None:
            w0 = prev_weights.reindex(returns.columns, fill_value=0.0)
            port.w0 = w0.to_frame()
            port.turnover = self.max_turnover

        # Get RF from kwargs (dynamic) or default to 0
        rf = kwargs.get('rf', 0.0)

        opt_kwargs = {
            'model': 'Classic',
            'rm': 'MV',
            'obj': 'MaxRet', # Maximize Return
            'rf': rf,
            'l': 0,
            'hist': True
        }
        
        w = self._safe_optimize(port, **opt_kwargs)
            
        # Process Result (Clean & Normalize)
        weights = w.squeeze()
        if weights.abs().sum() < 1e-6:
            raise ValueError("MaxRet returned zero weights.")
            
        weights = weights[weights.abs() > 1e-6]
        weights = weights / weights.sum()

        # [FALLBACK] Apply Turnover Constraint (Post-Processing)
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)

        # [FALLBACK] Apply Sector Constraint (Post-Processing)
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)

        return weights

class EqualWeightOptimizer(BaseOptimizer):
    """
    Simple Equal Weight Optimizer.
    """
    def optimize(self, returns, prev_weights=None, **kwargs):
        n = len(returns.columns)
        return pd.Series(1.0/n, index=returns.columns)


class ValueWeightedOptimizer:
    """
    Value Weighted (Market Cap Weighted) Optimizer.
    Requires 'market_caps' to be passed to optimize().
    """
    def __init__(self, max_turnover=None, long_only=True):
        self.max_turnover = max_turnover
        self.long_only = long_only # VW is typically Long Only

    def optimize(self, returns, prev_weights=None, **kwargs):
        # 1. Get Market Caps
        if 'market_caps' not in kwargs or kwargs['market_caps'] is None:
            # Fallback to Equal Weight if no caps provided
            n = len(returns.columns)
            return pd.Series(1.0/n, index=returns.columns)
            
        caps = kwargs['market_caps']
        
        # Align
        common = returns.columns.intersection(caps.index)
        if len(common) == 0:
             return pd.Series(1.0/len(returns.columns), index=returns.columns)
             
        caps_aligned = caps[common]
        weights = caps_aligned / caps_aligned.sum()
        
        return weights

class GeneralOptimizer(BaseOptimizer):
    """
    Universal Optimizer for Expert Profile.
    Wraps Riskfolio-Lib with full configuration flexibility.
    NOW BULLETPROOF: Uses _safe_optimize and proper error handling.
    """
    def __init__(self, model='Classic', rm='MV', obj='MinRisk', rf=0.0, l=0.0, hist=True, 
                 cov_method='ledoit', max_turnover=None, long_only=True, solver='CLARABEL'): # Solver CLARABEL par défaut
        super().__init__(cov_method, max_turnover, risk_aversion=l, long_only=long_only, solver=solver)
        self.model = model
        self.rm = rm
        self.obj = obj
        self.rf = rf
        self.l = l
        self.hist = hist
        
    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        # Clean returns
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 1. Initialize Riskfolio Portfolio
        port = rp.Portfolio(returns=returns_clean)
        port.budget = 1.0
        port.sht = not self.long_only
        port.solver = self.solver # Force solver selection
        
        # 2. Estimate Stats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                port.assets_stats(method_mu='hist', method_cov=self.cov_method)
            except:
                port.assets_stats(method_mu='hist', method_cov='ledoit')
        
        # Apply Sector Constraints
        self._apply_sector_constraints(port, sector_constraints, sector_map, returns)
        
        # 3. Optimize using SAFE WRAPPER
        # [FIX] Native Turnover Constraint
        if self.max_turnover is not None and prev_weights is not None:
            w0 = prev_weights.reindex(returns.columns, fill_value=0.0)
            port.w0 = w0.to_frame()
            port.turnover = self.max_turnover
        
        # Dynamic RF from kwargs overrides init
        rf = kwargs.get('rf', self.rf)
        
        opt_kwargs = {
            'model': self.model,
            'rm': self.rm,
            'obj': self.obj,
            'rf': rf,
            'l': self.l,
            'hist': self.hist
        }

        # UTILISATION DU WRAPPER DE SÉCURITÉ
        w = self._safe_optimize(port, **opt_kwargs)
            
        # 4. Process Result
        weights = w.squeeze()
        
        # [FIX] NO SILENT FAIL -> ERROR
        if weights.abs().sum() < 1e-6:
            raise ValueError(f"General Optimization ({self.obj}/{self.rm}) returned zero weights. Constraints impossible?")
            
        weights = weights[weights.abs() > 1e-6]
        weights = weights / weights.sum()
        
        # 5. Fallbacks (Safe post-processing)
        weights = self._apply_turnover_constraint(weights, prev_weights, returns)
        weights = self._apply_sector_fallback(weights, sector_constraints, sector_map)
        
        return weights



class ResampledOptimizer(BaseOptimizer):
    """
    Wrapper class for Michaud Resampling.
    Runs the base optimizer N times with Monte Carlo simulated inputs and averages the weights.
    NOW BULLETPROOF: Handles strict failures from base optimizers.
    """
    def __init__(self, base_optimizer, n_simulations=100, seed=None):
        # On ne passe pas d'arguments au super() car on utilise ceux du base_optimizer
        self.base_optimizer = base_optimizer
        self.n_simulations = n_simulations
        self.seed = seed
        
        # Héritage des propriétés clés pour l'interface
        self.max_turnover = base_optimizer.max_turnover
        self.cov_method = base_optimizer.cov_method
        
    def optimize(self, returns, prev_weights=None, sector_constraints=None, sector_map=None, **kwargs):
        # 0. Bypass si pas de simulation demandée
        if self.n_simulations <= 1:
            return self.base_optimizer.optimize(returns, prev_weights, sector_constraints, sector_map, **kwargs)

        # 1. Calcul des Stats Historiques (Le "Seed" de la simulation)
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        port = rp.Portfolio(returns=returns_clean)
        
        try:
            # On récupère la méthode du fils ou par défaut 'ledoit'
            method_cov = getattr(self.base_optimizer, 'cov_method', 'ledoit')
            port.assets_stats(method_mu='hist', method_cov=method_cov)
            
            mu = port.mu.values.flatten()
            cov = port.cov.values
            # Regularisation PSD pour la simulation Monte Carlo
            cov = cov + np.eye(cov.shape[0]) * 1e-6
        except:
            # Fallback stats simples
            mu = returns_clean.mean().values
            cov = returns_clean.cov().values + np.eye(returns_clean.shape[1]) * 1e-6
            
        n_obs, n_assets = returns_clean.shape
        
        # Seed global
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # 2. Monte Carlo Simulation
        accumulated_weights = pd.DataFrame(0.0, index=returns.columns, columns=['weight'])
        valid_simulations = 0
        failures = 0
        
        for _ in range(self.n_simulations):
            try:
                # A. Simulation des Rendements
                sim_returns_np = np.random.multivariate_normal(mu, cov, size=n_obs)
                sim_returns = pd.DataFrame(sim_returns_np, columns=returns.columns, index=returns.index)
                
                # B. Optimisation sur données simulées
                # IMPORTANT : On ne passe PAS prev_weights ici. 
                # On cherche le portefeuille idéal théorique moyen.
                w = self.base_optimizer.optimize(
                    sim_returns, 
                    prev_weights=None, # Pas de turnover DANS la simulation
                    sector_constraints=sector_constraints, 
                    sector_map=sector_map, 
                    **kwargs # Important : Laisse passer risk_budget, rf, etc.
                )
                
                accumulated_weights['weight'] = accumulated_weights['weight'].add(w, fill_value=0)
                valid_simulations += 1
                
            except (ValueError, Exception):
                failures += 1
                continue
        
        # 3. Vérification de la Santé de la Simulation
        if valid_simulations < self.n_simulations * 0.5:
            print(f"⚠️ WARNING: Michaud instable. {failures}/{self.n_simulations} simulations ont échoué.")
            if valid_simulations == 0:
                print("Falling back to single optimization on original data.")
                return self.base_optimizer.optimize(returns, prev_weights, sector_constraints, sector_map, **kwargs)
                
        # 4. Moyennage
        avg_weights = accumulated_weights['weight'] / valid_simulations
        
        # 5. Post-Processing Final (Le "Polissage")
        # On réapplique le fallback sectoriel sur la moyenne pour nettoyer les erreurs d'arrondi
        avg_weights = self.base_optimizer._apply_sector_fallback(avg_weights, sector_constraints, sector_map)
        
        # On applique le turnover sur le résultat final lissé
        final_weights = self.base_optimizer._apply_turnover_constraint(avg_weights, prev_weights, returns)
        
        return final_weights


