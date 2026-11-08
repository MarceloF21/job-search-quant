import pandas as pd
import pytest

from quant_trading_platform.strategies import (
    MACDStrategy,
    MeanReversionStrategy,
    MLStrategy,
    MomentumStrategy,
)

RULE_STRATEGIES = [MomentumStrategy(), MeanReversionStrategy(), MACDStrategy()]


@pytest.mark.parametrize("strategy", RULE_STRATEGIES, ids=lambda s: s.name)
def test_targets_are_binary(strategy, synthetic_prices):
    targets = strategy.target_positions(synthetic_prices["SIM_A"])
    assert set(targets.unique()) <= {0, 1}
    assert targets.index.equals(synthetic_prices["SIM_A"].index)


@pytest.mark.parametrize(
    "strategy",
    [*RULE_STRATEGIES, MLStrategy(min_train=150, retrain_every=40, n_estimators=20)],
    ids=lambda s: s.name,
)
def test_no_lookahead(strategy, synthetic_prices):
    """Targets up to day k must not change when later prices are added."""
    full = synthetic_prices["SIM_B"]
    full_targets = strategy.target_positions(full)
    for cut in (320, 455, 610):
        partial = strategy.target_positions(full.iloc[:cut])
        pd.testing.assert_series_equal(partial, full_targets.iloc[:cut])


def test_ml_predicts_only_after_training_window(synthetic_prices):
    model = MLStrategy(min_train=150, retrain_every=40, n_estimators=20)
    probs = model.probabilities(synthetic_prices["SIM_C"])
    first_prediction = probs.first_valid_index()
    warmup_and_training = probs.index.get_loc(first_prediction)
    assert warmup_and_training >= 150
    assert probs.dropna().between(0, 1).all()
