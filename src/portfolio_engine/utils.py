import cvxpy as cp
import warnings

def get_optimal_solver():
    """
    Determines the best available solver.
    Priority:
    1. MOSEK (Commercial, Fast, Robust) - Checks if installed AND licensed.
    2. CLARABEL (Open Source, Modern, Robust) - Default fallback.
    
    Returns:
    --------
    str : Solver name ('MOSEK' or 'CLARABEL')
    """
    # Check for MOSEK
    if 'MOSEK' in cp.installed_solvers():
        try:
            # Try a tiny dummy optimization to check license
            # (Just checking installed_solvers() is not enough, it doesn't check license)
            x = cp.Variable(1)
            prob = cp.Problem(cp.Minimize(x), [x >= 0])
            prob.solve(solver='MOSEK', verbose=False)
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                return 'MOSEK'
        except Exception:
            pass # MOSEK failed (likely license issue)
            
    # Fallback
    return 'CLARABEL'
