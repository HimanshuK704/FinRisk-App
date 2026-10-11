
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


MIN_OBSERVATIONS = 30
DEFAULT_SIMULATIONS = 10_000
DEFAULT_SEED = 42


def _clean_returns(returns: pd.Series) -> np.ndarray:
    """Validate and return a finite 1D array of simple returns."""
    values = pd.Series(returns, dtype=float).dropna().to_numpy()

    if values.ndim != 1 or len(values) < MIN_OBSERVATIONS:
        raise ValueError(
            f"At least {MIN_OBSERVATIONS} valid return observations are required."
        )

    if not np.isfinite(values).all():
        raise ValueError("Returns contain non-finite values.")

    if (values <= -1).any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    return values


def historical_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """Historical one-period VaR and CVaR as positive loss fractions."""
    values = _clean_returns(returns)
    _validate_confidence(confidence)

    losses = -values
    var = float(np.quantile(losses, confidence, method="higher"))
    tail = losses[losses >= var]

    return {
        "var": var,
        "cvar": float(tail.mean()),
        "observations": int(len(losses)),
        "method": "historical_simulation",
    }


def gaussian_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """Parametric Gaussian one-period VaR and CVaR."""
    values = _clean_returns(returns)
    _validate_confidence(confidence)

    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Return volatility must be positive and finite.")

    z = float(norm.ppf(confidence))
    var = -mu + z * sigma
    cvar = -mu + sigma * norm.pdf(z) / (1.0 - confidence)

    return {
        "var": float(var),
        "cvar": float(cvar),
        "mean_return": mu,
        "volatility": sigma,
        "method": "gaussian_parametric",
    }


def monte_carlo_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sims: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Reproducible univariate Gaussian Monte Carlo baseline."""
    values = _clean_returns(returns)
    _validate_confidence(confidence)

    if n_sims < 1_000:
        raise ValueError("At least 1,000 simulations are required.")

    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Return volatility must be positive and finite.")

    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(mu, sigma, size=n_sims)
    simulated_losses = -simulated_returns

    var = float(np.quantile(simulated_losses, confidence, method="higher"))
    tail = simulated_losses[simulated_losses >= var]

    return {
        "var": var,
        "cvar": float(tail.mean()),
        "simulations": int(n_sims),
        "seed": int(seed),
        "method": "univariate_gaussian_monte_carlo",
    }


def multivariate_monte_carlo_var_cvar(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    confidence: float = 0.95,
    n_sims: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Gaussian Monte Carlo using the sample covariance between assets."""
    _validate_confidence(confidence)

    if n_sims < 1_000:
        raise ValueError("At least 1,000 simulations are required.")

    if asset_returns.empty or asset_returns.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    if set(asset_returns.columns) != set(weights):
        raise ValueError("Weights must match the return columns exactly.")

    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Weights must be finite and non-negative.")

    if not np.isclose(sum(weights.values()), 1.0, atol=1e-6):
        raise ValueError("Weights must sum to 1.")

    clean = asset_returns.astype(float).dropna()
    if len(clean) < MIN_OBSERVATIONS:
        raise ValueError("Insufficient aligned asset return observations.")

    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Asset returns contain non-finite values.")

    w = np.array([weights[col] for col in clean.columns], dtype=float)
    mean_vector = clean.mean().to_numpy()
    covariance = clean.cov().to_numpy()

    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() < -1e-8:
        raise ValueError("Estimated covariance matrix is not positive semidefinite.")

    rng = np.random.default_rng(seed)
    simulated_asset_returns = rng.multivariate_normal(
        mean_vector,
        covariance,
        size=n_sims,
        check_valid="raise",
    )

    simulated_portfolio_returns = simulated_asset_returns @ w
    simulated_losses = -simulated_portfolio_returns

    var = float(np.quantile(simulated_losses, confidence, method="higher"))
    tail = simulated_losses[simulated_losses >= var]

    return {
        "var": var,
        "cvar": float(tail.mean()),
        "simulations": int(n_sims),
        "seed": int(seed),
        "method": "multivariate_gaussian_monte_carlo",
        "assets": list(clean.columns),
    }


def _validate_confidence(confidence: float) -> None:
    if not np.isfinite(confidence) or not 0.5 < confidence < 0.999:
        raise ValueError("Confidence must be greater than 0.5 and less than 0.999.")
