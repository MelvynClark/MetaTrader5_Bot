"""
hmm_engine.py — Regime Detection Engine (Highest Priority)
===========================================================
Gaussian HMM with exactly 3 components:
  - Bull Trend   (highest mean return)
  - Bear Trend   (lowest mean return)
  - Neutral/Sideways (near-zero return, contracting vol)

The Subordination Principle: HMM state is the absolute baseline.
All strategy execution is strictly subordinate to the current regime.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "Bull Trend", 1: "Bear Trend", 2: "Neutral"}
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class RegimeDetector:
    """
    3-state Gaussian HMM for market regime detection.

    Features:
        1. Log Returns
        2. Normalized Range  = (High - Low) / Close
        3. Volume Volatility = rolling std of tick_volume (window=20)
    """

    N_COMPONENTS = 3  # Hard-coded — never change this
    COVARIANCE_TYPE = "full"
    N_ITER = 200
    RANDOM_STATE = 42
    VOL_WINDOW = 20

    def __init__(self):
        from hmmlearn.hmm import GaussianHMM

        self.model = GaussianHMM(
            n_components=self.N_COMPONENTS,
            covariance_type=self.COVARIANCE_TYPE,
            n_iter=self.N_ITER,
            random_state=self.RANDOM_STATE,
            verbose=False,
        )
        self.scaler = StandardScaler()
        self.state_map: dict[int, str] = {}
        self.fitted = False

    # ── Feature Engineering ──────────────────
    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the 3-feature matrix from OHLCV data.
        Expects columns: Open, High, Low, Close, Volume.
        """
        features = pd.DataFrame(index=df.index)

        # 1. Log Returns
        features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # 2. Normalized Range (intrabar volatility proxy)
        features["norm_range"] = (df["High"] - df["Low"]) / df["Close"]

        # 3. Volume Volatility (rolling std of tick volume)
        features["vol_volatility"] = df["Volume"].rolling(
            window=RegimeDetector.VOL_WINDOW, min_periods=5
        ).std()

        features.dropna(inplace=True)
        return features

    # ── Training ─────────────────────────────
    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """
        Train the HMM on historical OHLCV data.
        Automatically maps hidden states to human-readable regimes.
        """
        features = self._build_features(df)
        X = features.values
        X_scaled = self.scaler.fit_transform(X)

        logger.info("Fitting HMM on %d samples, %d features", *X_scaled.shape)
        self.model.fit(X_scaled)

        # Decode states for the training set
        states = self.model.predict(X_scaled)

        # Map states by mean log return
        self.state_map = self._auto_map_states(features, states)
        self.fitted = True
        logger.info("HMM fitted. State map: %s", self.state_map)
        return self

    def _auto_map_states(
        self, features: pd.DataFrame, states: np.ndarray
    ) -> dict[int, str]:
        """
        Automatically assign labels based on statistical profile:
          - Bull  = state with highest mean log return
          - Bear  = state with lowest mean log return
          - Neutral = remaining state
        """
        mean_returns = {}
        mean_vol = {}
        for s in range(self.N_COMPONENTS):
            mask = states == s
            mean_returns[s] = features.loc[mask, "log_return"].mean() if mask.any() else 0.0
            mean_vol[s] = features.loc[mask, "norm_range"].mean() if mask.any() else 0.0

        sorted_by_return = sorted(mean_returns, key=mean_returns.get)
        bear_state = sorted_by_return[0]
        bull_state = sorted_by_return[-1]
        neutral_state = sorted_by_return[1]

        mapping = {
            bull_state: "Bull Trend",
            bear_state: "Bear Trend",
            neutral_state: "Neutral",
        }

        for s, label in mapping.items():
            logger.info(
                "  State %d → %s  (mean_ret=%.6f, mean_range=%.6f)",
                s, label, mean_returns[s], mean_vol[s],
            )
        return mapping

    # ── Prediction ───────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each bar. Returns Series of regime labels
        aligned to df.index (after dropping NaN feature rows).
        """
        if not self.fitted:
            raise RuntimeError("RegimeDetector not fitted — call .fit() first")

        features = self._build_features(df)
        X_scaled = self.scaler.transform(features.values)
        raw_states = self.model.predict(X_scaled)
        labels = pd.Series(
            [self.state_map.get(s, "Unknown") for s in raw_states],
            index=features.index,
            name="regime",
        )
        return labels

    def predict_latest(self, df: pd.DataFrame) -> str:
        """Return the regime label for the most recent bar."""
        labels = self.predict(df)
        return labels.iloc[-1] if len(labels) > 0 else "Unknown"

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return per-bar posterior probabilities for each regime.
        Columns: Bull Trend, Bear Trend, Neutral.
        """
        if not self.fitted:
            raise RuntimeError("RegimeDetector not fitted — call .fit() first")

        features = self._build_features(df)
        X_scaled = self.scaler.transform(features.values)
        log_proba = self.model.score_samples(X_scaled)[1]
        proba = np.exp(log_proba)

        col_names = [self.state_map.get(i, f"State_{i}") for i in range(self.N_COMPONENTS)]
        return pd.DataFrame(proba, index=features.index, columns=col_names)

    # ── Persistence ──────────────────────────
    def save(self, name: str = "hmm_default"):
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "state_map": self.state_map,
                "fitted": self.fitted,
            }, f)
        logger.info("HMM saved -> %s", path)

    def load(self, name: str = "hmm_default") -> "RegimeDetector":
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved HMM at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.state_map = data["state_map"]
        self.fitted = data["fitted"]
        logger.info("HMM loaded <- %s | map: %s", path, self.state_map)
        return self

    # ── Utility ──────────────────────────────
    def regime_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary table: regime, count, pct, mean_return, mean_range.
        """
        labels = self.predict(df)
        features = self._build_features(df)
        features["regime"] = labels

        summary = features.groupby("regime").agg(
            count=("log_return", "count"),
            mean_return=("log_return", "mean"),
            std_return=("log_return", "std"),
            mean_range=("norm_range", "mean"),
            mean_vol_vol=("vol_volatility", "mean"),
        )
        summary["pct"] = (summary["count"] / summary["count"].sum() * 100).round(1)
        return summary.sort_values("mean_return", ascending=False)
