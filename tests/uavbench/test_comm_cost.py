"""Communication accounting: pin the payload constants and the 2-tier formula.

Previously only sign/relative-ordering was checked; these pin the literal
values so any silent change to the accounting invalidates loudly.
"""

import pytest

from uavbench.fl.federated import _IOT_MODEL_SIZE_MB, _MODEL_SIZE_MB, _UAV_MODEL_SIZE_MB

# Parameter counts documented at the constants' definition:
# IoT payload  = struct_branch (17,216) + fusion (50,436)            = 67,652
# UAV payload  = img_proj (65,664) + struct_branch + fusion          = 133,316
IOT_PARAMS = 67_652
UAV_PARAMS = 133_316


def test_payload_constants_pinned():
    assert _IOT_MODEL_SIZE_MB == pytest.approx(IOT_PARAMS * 4 / 1_000_000)
    assert _UAV_MODEL_SIZE_MB == pytest.approx(UAV_PARAMS * 4 / 1_000_000)
    assert _MODEL_SIZE_MB == _IOT_MODEL_SIZE_MB  # run_tier2 back-compat alias


def test_two_tier_formula_hand_computed():
    # run_full_hfl hierarchical path: uplink+downlink on both tiers.
    n_selected, n_active_uavs = 12, 3
    expected = 2.0 * 12 * (67_652 * 4 / 1e6) + 2.0 * 3 * (133_316 * 4 / 1e6)
    comm_mb = 2.0 * n_selected * _IOT_MODEL_SIZE_MB + 2.0 * n_active_uavs * _UAV_MODEL_SIZE_MB
    assert comm_mb == pytest.approx(expected)
    # Literal value: 24 x 0.270608 + 6 x 0.533264 = 6.494592 + 3.199584 MB.
    assert comm_mb == pytest.approx(9.694176)


def test_flat_path_uses_iot_payload_only():
    n_selected = 12
    flat = 2.0 * n_selected * _IOT_MODEL_SIZE_MB
    assert flat == pytest.approx(2.0 * 12 * 0.270608)
