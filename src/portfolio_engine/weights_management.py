from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseWeightUpdater(ABC):
    """
    Abstract base class for weight updating logic.
    Defines how weights evolve from period t to t+1, including rebalancing and drift.
    """
    
    @abstractmethod
    def update_weights(self, **kwargs):
        """
        Calculate the new weights, turnover, and portfolio return for the current period.
        
        Returns:
        --------
        tuple: (final_weights, turnover, gross_return, net_return)
            - final_weights: Weights at the END of the period (after drift).
            - turnover: Turnover incurred at the START of the period.
            - gross_return: Portfolio return BEFORE transaction costs.
            - net_return: Portfolio return AFTER transaction costs.
        """
        pass

class VWWeightUpdater(BaseWeightUpdater):
    """
    Handles weight updates specifically for Value-Weighted (VW) portfolios.
    
    Logic:
    - The 'target' is always the Market Cap weights of the current month (lagged).
    - We ALWAYS trade to these new cap weights at the start of the month.
    - This captures the natural evolution of the portfolio (price changes) plus
      corporate actions (share issuance/buybacks) reflected in the market cap.
    
    New Feature: Partial Rebalancing
    - If max_turnover is set, we limit the trade towards the target.
    """
    
    def update_weights(self, current_cap_weights: pd.Series, prev_weights: pd.Series, returns_month: pd.Series, cost_bps: float = 0.0, max_turnover: float = None):
        """
        Parameters:
        -----------
        current_cap_weights : pd.Series
            Weights based on Market Cap at t-1 (lagged). This is the target for this month.
        prev_weights : pd.Series
            Weights at the end of previous month (used only for turnover calculation).
        returns_month : pd.Series
            Returns for the current month.
        cost_bps : float
            Transaction cost in basis points (e.g., 0.0010 for 10 bps).
        max_turnover : float, optional
            Maximum allowed turnover. If exceeded, we partially rebalance.
        """
        # 1. Determine Target Weights
        target_weights = current_cap_weights
        
        # 2. Calculate Turnover & Apply Constraints
        if prev_weights is None or prev_weights.empty:
            # First period: full allocation
            weights_to_use = target_weights
            turnover = 1.0 # Initial investment is 100% turnover
        else:
            # Align indices
            all_assets = target_weights.index.union(prev_weights.index)
            w_target_aligned = target_weights.reindex(all_assets, fill_value=0)
            w_prev_aligned = prev_weights.reindex(all_assets, fill_value=0)
            
            # Calculate desired trade vector
            trade_vector = w_target_aligned - w_prev_aligned
            desired_turnover = trade_vector.abs().sum()
            
            # Apply Max Turnover Constraint (Partial Rebalancing)
            if max_turnover is not None and desired_turnover > max_turnover:
                # Scale down the trade
                scaling_factor = max_turnover / desired_turnover
                actual_trade = trade_vector * scaling_factor
                weights_to_use = w_prev_aligned + actual_trade
                turnover = max_turnover
            else:
                weights_to_use = w_target_aligned
                turnover = desired_turnover
        
        # 3. Calculate Transaction Costs
        #    Cost is applied to the turnover
        #    Note: Initial investment turnover (1.0) usually incurs cost too.
        transaction_cost = turnover * cost_bps
        
        # 4. Calculate Portfolio Return
        common_assets = weights_to_use.index.intersection(returns_month.index)
        w_active = weights_to_use.reindex(common_assets, fill_value=0)
        r_active = returns_month.reindex(common_assets, fill_value=0)
        
        # Renormalize (crucial for VW if we lose some stocks due to missing returns)
        if w_active.sum() > 0:
            w_active = w_active / w_active.sum()
            
        gross_return = (w_active * r_active).sum()
        net_return = gross_return - transaction_cost
        
        # 5. Evolve weights (Drift) for the record
        #    w_{t+1} = w_t * (1 + r_t) / (1 + R_p)
        final_weights = w_active * (1 + r_active) / (1 + gross_return)
        
        return final_weights, turnover, gross_return, net_return

class OptimizedWeightUpdater(BaseWeightUpdater):
    """
    Handles weight updates for Optimized portfolios (MinVar, EW, Tangency, etc.).
    
    Modes:
    - 'monthly': Forces weights to the new optimized target (high turnover).
    - 'drift': Ignores the new target, lets the previous weights drift with returns (0 turnover).
               (Note: In 'drift' mode, target_weights is only used at t=0).
    """
    
    def update_weights(self, target_weights: pd.Series, prev_weights: pd.Series, mode: str, returns_month: pd.Series, cost_bps: float = 0.0, max_turnover: float = None):
        """
        Parameters:
        -----------
        target_weights : pd.Series
            The new optimal weights calculated for this month.
        prev_weights : pd.Series
            The weights at the end of the previous month.
        mode : str
            'monthly' or 'drift'.
        returns_month : pd.Series
            Returns for the current month.
        cost_bps : float
            Transaction cost in basis points.
        max_turnover : float, optional
            Not used here for 'monthly' mode usually, as the constraint should be in the Optimizer.
            However, we could implement partial rebalancing here too if desired.
            For now, we assume the Optimizer handled the constraint if mode='monthly'.
        """
        # 1. Determine which weights to use based on mode
        if prev_weights is None or prev_weights.empty:
            # First period: we must use the target
            weights_to_use = target_weights
            turnover = 1.0
        else:
            if mode == 'monthly':
                # REBALANCE: We force the new target
                # (Assumption: target_weights already respects max_turnover if it came from MinVarOptimizer)
                weights_to_use = target_weights
                
                # Calculate turnover against drifted previous weights
                all_assets = target_weights.index.union(prev_weights.index)
                w_target_aligned = target_weights.reindex(all_assets, fill_value=0)
                w_prev_aligned = prev_weights.reindex(all_assets, fill_value=0)
                
                turnover = (w_target_aligned - w_prev_aligned).abs().sum()
                
            elif mode == 'drift':
                # DRIFT: We ignore the new optimization (except for the very first month)
                # We essentially "Buy & Hold" the previous portfolio
                weights_to_use = prev_weights
                turnover = 0.0
                
            else:
                raise ValueError(f"Unknown mode for optimized portfolio: {mode}")

        # 2. Calculate Transaction Costs
        transaction_cost = turnover * cost_bps

        # 3. Calculate Portfolio Return for this month
        common_assets = weights_to_use.index.intersection(returns_month.index)
        w_active = weights_to_use.reindex(common_assets, fill_value=0)
        r_active = returns_month.reindex(common_assets, fill_value=0)
        
        # Renormalize if needed
        if w_active.sum() > 0:
            w_active = w_active / w_active.sum()
            
        gross_return = (w_active * r_active).sum()
        net_return = gross_return - transaction_cost
        
        # 4. Evolve weights for next period (Drift)
        final_weights = w_active * (1 + r_active) / (1 + gross_return)
        
        return final_weights, turnover, gross_return, net_return
