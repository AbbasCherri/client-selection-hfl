"""Tests for CachedFusionModel, fedavg, reputation_fedavg, and mixed aggregation (model.py)."""

import numpy as np
import pytest
import torch

from uavbench.fl.model import (
    CachedFusionModel,
    clone_model,
    fedavg,
    mixed_fedavg,
    mixed_reputation_fedavg,
    reputation_fedavg,
)


# ── CachedFusionModel ─────────────────────────────────────────────────────────

class TestCachedFusionModel:
    def _model(self):
        m = CachedFusionModel()
        m.eval()
        return m

    def test_forward_output_shape(self):
        m = self._model()
        img = torch.randn(4, 512)
        struct = torch.randn(4, 9)
        with torch.no_grad():
            out = m(img, struct)
        assert out.shape == (4, 4)

    def test_forward_single_sample(self):
        m = self._model()
        img = torch.randn(1, 512)
        struct = torch.randn(1, 9)
        with torch.no_grad():
            out = m(img, struct)
        assert out.shape == (1, 4)

    def test_img_proj_is_trainable_by_default(self):
        """img_proj starts trainable; FL harness calls freeze_img_proj() after construction."""
        m = self._model()
        for p in m.img_proj.parameters():
            assert p.requires_grad, "img_proj must be trainable on fresh model"

    def test_struct_branch_is_trainable(self):
        m = self._model()
        for p in m.struct_branch.parameters():
            assert p.requires_grad

    def test_fusion_is_trainable(self):
        m = self._model()
        for p in m.fusion.parameters():
            assert p.requires_grad

    def test_trainable_state_dict_excludes_img_proj(self):
        m = self._model()
        sd = m.trainable_state_dict()
        for k in sd:
            assert not k.startswith("img_proj."), f"frozen key leaked: {k}"

    def test_trainable_state_dict_includes_struct_and_fusion(self):
        m = self._model()
        sd = m.trainable_state_dict()
        struct_keys = [k for k in sd if k.startswith("struct_branch.")]
        fusion_keys = [k for k in sd if k.startswith("fusion.")]
        assert len(struct_keys) > 0
        assert len(fusion_keys) > 0

    def test_trainable_param_count(self):
        m = self._model()
        sd = m.trainable_state_dict()
        n_params = sum(v.numel() for v in sd.values())
        # struct_branch: 17,216  |  fusion: 50,436
        assert n_params == 67_652, f"Expected 67,652 trainable params, got {n_params}"

    def test_load_trainable_state_dict_modifies_weights(self):
        m1 = CachedFusionModel()
        m2 = CachedFusionModel()
        # Verify they differ after independent init
        sd1 = m1.trainable_state_dict()
        sd2 = m2.trainable_state_dict()
        # Load m1's weights into m2
        m2.load_trainable_state_dict(sd1)
        sd2_after = m2.trainable_state_dict()
        for k in sd1:
            assert torch.allclose(sd1[k], sd2_after[k]), f"Key {k} not loaded correctly"

    def test_load_trainable_state_dict_leaves_img_proj_unchanged(self):
        m = CachedFusionModel()
        img_proj_w_before = m.img_proj.proj[0].weight.clone()
        # Load fresh trainable dict (will not touch img_proj)
        m.load_trainable_state_dict(m.trainable_state_dict())
        img_proj_w_after = m.img_proj.proj[0].weight
        assert torch.equal(img_proj_w_before, img_proj_w_after)

    def test_clone_model_is_independent(self):
        m1 = CachedFusionModel()
        m2 = clone_model(m1)
        # Modifying m1's trainable weights should not affect m2
        with torch.no_grad():
            m1.struct_branch.mlp[0].weight.fill_(0.0)
        assert not torch.all(m2.struct_branch.mlp[0].weight == 0.0)


# ── freeze / unfreeze ────────────────────────────────────────────────────────

class TestFreezeUnfreeze:
    def test_freeze_img_proj_disables_grad(self):
        m = CachedFusionModel()
        m.freeze_img_proj()
        for p in m.img_proj.parameters():
            assert not p.requires_grad

    def test_unfreeze_img_proj_enables_grad(self):
        m = CachedFusionModel()
        m.freeze_img_proj()
        m.unfreeze_img_proj()
        for p in m.img_proj.parameters():
            assert p.requires_grad

    def test_freeze_does_not_affect_struct_branch(self):
        m = CachedFusionModel()
        m.freeze_img_proj()
        for p in m.struct_branch.parameters():
            assert p.requires_grad

    def test_freeze_does_not_affect_fusion(self):
        m = CachedFusionModel()
        m.freeze_img_proj()
        for p in m.fusion.parameters():
            assert p.requires_grad

    def test_unfreeze_does_not_affect_struct_branch(self):
        m = CachedFusionModel()
        m.unfreeze_img_proj()
        for p in m.struct_branch.parameters():
            assert p.requires_grad

    def test_toggle_roundtrip(self):
        m = CachedFusionModel()
        m.freeze_img_proj()
        m.unfreeze_img_proj()
        m.freeze_img_proj()
        for p in m.img_proj.parameters():
            assert not p.requires_grad


# ── full_trainable_state_dict ─────────────────────────────────────────────────

class TestFullTrainableStateDict:
    def test_includes_img_proj_keys(self):
        m = CachedFusionModel()
        sd = m.full_trainable_state_dict()
        img_keys = [k for k in sd if k.startswith("img_proj.")]
        assert len(img_keys) > 0

    def test_includes_struct_branch_keys(self):
        m = CachedFusionModel()
        sd = m.full_trainable_state_dict()
        struct_keys = [k for k in sd if k.startswith("struct_branch.")]
        assert len(struct_keys) > 0

    def test_includes_fusion_keys(self):
        m = CachedFusionModel()
        sd = m.full_trainable_state_dict()
        fusion_keys = [k for k in sd if k.startswith("fusion.")]
        assert len(fusion_keys) > 0

    def test_total_param_count(self):
        m = CachedFusionModel()
        sd = m.full_trainable_state_dict()
        n_params = sum(v.numel() for v in sd.values())
        # img_proj: 65,664  |  struct_branch: 17,216  |  fusion: 50,436
        assert n_params == 133_316, f"Expected 133,316 params, got {n_params}"

    def test_load_full_state_dict_updates_all_modules(self):
        m1 = CachedFusionModel()
        m2 = CachedFusionModel()
        m1.unfreeze_img_proj()
        sd1 = m1.full_trainable_state_dict()
        m2.load_full_trainable_state_dict(sd1)
        sd2 = m2.full_trainable_state_dict()
        for k in sd1:
            assert torch.allclose(sd1[k], sd2[k]), f"Key {k} not loaded correctly"

    def test_load_full_without_img_proj_leaves_it_unchanged(self):
        """flat_fl server aggregation omits img_proj — those weights must stay."""
        m = CachedFusionModel()
        before = {k: v.clone() for k, v in m.img_proj.state_dict().items()}
        partial = m.trainable_state_dict()  # only struct_branch + fusion
        m.load_full_trainable_state_dict(partial)
        after = m.img_proj.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k]), f"img_proj.{k} changed unexpectedly"

    def test_full_sd_is_clone_not_reference(self):
        m = CachedFusionModel()
        sd = m.full_trainable_state_dict()
        first_key = next(iter(sd))
        sd[first_key].fill_(999.0)
        # Original model param must be unchanged
        with torch.no_grad():
            actual = list(m.img_proj.parameters())[0]
            assert not torch.all(actual == 999.0)


# ── mixed_fedavg ──────────────────────────────────────────────────────────────

class TestMixedFedAvg:
    def _uav_sd(self, val_img: float, val_struct: float) -> dict:
        return {
            "img_proj.w": torch.full((2,), val_img),
            "struct_branch.w": torch.full((2,), val_struct),
            "fusion.w": torch.full((2,), val_struct),
        }

    def _iot_sd(self, val: float) -> dict:
        return {
            "struct_branch.w": torch.full((2,), val),
            "fusion.w": torch.full((2,), val),
        }

    def test_img_proj_from_uav_only(self):
        uav_sd = self._uav_sd(7.0, 1.0)
        iot_sd = self._iot_sd(0.0)
        result = mixed_fedavg((uav_sd, 10), [(iot_sd, 10)])
        assert torch.allclose(result["img_proj.w"], torch.full((2,), 7.0))

    def test_struct_fusion_is_weighted_average(self):
        uav_sd = self._uav_sd(0.0, 0.0)
        iot_sd = self._iot_sd(10.0)
        # UAV n=0, IoT n=10 → struct should be 10.0
        result = mixed_fedavg((uav_sd, 0), [(iot_sd, 10)])
        assert torch.allclose(result["struct_branch.w"], torch.full((2,), 10.0))

    def test_uav_and_iot_equal_weight(self):
        uav_sd = self._uav_sd(5.0, 0.0)
        iot_sd = self._iot_sd(10.0)
        # UAV n=1, IoT n=1 → struct = 0.5*0 + 0.5*10 = 5.0
        result = mixed_fedavg((uav_sd, 1), [(iot_sd, 1)])
        assert torch.allclose(result["struct_branch.w"], torch.full((2,), 5.0))

    def test_zero_total_returns_uav_sd(self):
        uav_sd = self._uav_sd(3.0, 3.0)
        result = mixed_fedavg((uav_sd, 0), [])
        assert torch.allclose(result["img_proj.w"], torch.full((2,), 3.0))

    def test_multiple_iot_clients(self):
        uav_sd = self._uav_sd(0.0, 6.0)
        iot1 = self._iot_sd(0.0)
        iot2 = self._iot_sd(12.0)
        # UAV n=1, IoT1 n=1, IoT2 n=1 → total=3
        # struct: (1/3)*6 + (1/3)*0 + (1/3)*12 = 6.0
        result = mixed_fedavg((uav_sd, 1), [(iot1, 1), (iot2, 1)])
        assert torch.allclose(result["struct_branch.w"], torch.full((2,), 6.0), atol=1e-5)


# ── mixed_reputation_fedavg ───────────────────────────────────────────────────

class TestMixedReputationFedAvg:
    def _uav_sd(self, val_img: float, val_struct: float) -> dict:
        return {
            "img_proj.w": torch.full((2,), val_img),
            "struct_branch.w": torch.full((2,), val_struct),
            "fusion.w": torch.full((2,), val_struct),
        }

    def _iot_sd(self, val: float) -> dict:
        return {
            "struct_branch.w": torch.full((2,), val),
            "fusion.w": torch.full((2,), val),
        }

    def test_img_proj_always_from_uav(self):
        uav_sd = self._uav_sd(9.0, 0.0)
        iot_sd = self._iot_sd(0.0)
        result = mixed_reputation_fedavg((uav_sd, 10, 1.0), [(iot_sd, 10, 0.5)])
        assert torch.allclose(result["img_proj.w"], torch.full((2,), 9.0))

    def test_high_iot_reputation_dominates_struct(self):
        uav_sd = self._uav_sd(0.0, 0.0)
        iot_sd = self._iot_sd(10.0)
        # UAV rep=0.0 means w_uav=0; IoT rep=1.0, n=10 → struct should be 10.0
        result = mixed_reputation_fedavg((uav_sd, 10, 0.0), [(iot_sd, 10, 1.0)])
        assert torch.allclose(result["struct_branch.w"], torch.full((2,), 10.0), atol=1e-5)

    def test_zero_all_weights_falls_back_to_mixed_fedavg(self):
        uav_sd = self._uav_sd(5.0, 5.0)
        iot_sd = self._iot_sd(5.0)
        # All reps 0, all n=0 → fallback; result should still have img_proj from UAV
        result = mixed_reputation_fedavg((uav_sd, 0, 0.0), [(iot_sd, 0, 0.0)])
        assert "img_proj.w" in result

    def test_negative_reputation_clipped_to_zero(self):
        uav_sd = self._uav_sd(0.0, 0.0)
        iot_sd = self._iot_sd(10.0)
        # UAV rep=-1.0 → clipped to 0; IoT gets full weight
        result = mixed_reputation_fedavg((uav_sd, 1, -1.0), [(iot_sd, 1, 1.0)])
        # struct should be close to 10.0 (uav struct ignored due to rep=0)
        assert float(result["struct_branch.w"][0]) > 5.0

    def test_result_has_all_expected_keys(self):
        uav_sd = self._uav_sd(1.0, 1.0)
        iot_sd = self._iot_sd(1.0)
        result = mixed_reputation_fedavg((uav_sd, 1, 1.0), [(iot_sd, 1, 1.0)])
        assert "img_proj.w" in result
        assert "struct_branch.w" in result
        assert "fusion.w" in result


# ── fedavg ────────────────────────────────────────────────────────────────────

class TestFedAvg:
    def _sd(self, val: float) -> dict:
        return {"w": torch.full((4,), val)}

    def test_single_update_returns_copy(self):
        sd = self._sd(3.0)
        result = fedavg([(sd, 10)])
        assert torch.allclose(result["w"], torch.full((4,), 3.0))

    def test_single_update_is_a_copy_not_reference(self):
        sd = self._sd(3.0)
        result = fedavg([(sd, 10)])
        # Mutate result; original must not change
        result["w"].fill_(99.0)
        assert not torch.allclose(sd["w"], torch.full((4,), 99.0))

    def test_uniform_weights_equal_mean(self):
        sd1, sd2 = self._sd(2.0), self._sd(4.0)
        result = fedavg([(sd1, 1), (sd2, 1)])
        assert torch.allclose(result["w"], torch.full((4,), 3.0))

    def test_weighted_average(self):
        sd1, sd2 = self._sd(0.0), self._sd(10.0)
        result = fedavg([(sd1, 3), (sd2, 1)])  # weight 0.75 and 0.25
        expected = 0.75 * 0.0 + 0.25 * 10.0   # = 2.5
        assert torch.allclose(result["w"], torch.full((4,), expected))

    def test_zero_total_returns_first_copy(self):
        sd = self._sd(7.0)
        result = fedavg([(sd, 0)])
        assert torch.allclose(result["w"], torch.full((4,), 7.0))

    def test_zero_total_returns_clone_not_reference(self):
        sd = self._sd(7.0)
        result = fedavg([(sd, 0)])
        result["w"].fill_(0.0)
        assert torch.allclose(sd["w"], torch.full((4,), 7.0))

    def test_three_clients(self):
        sd1 = {"a": torch.tensor([1.0, 2.0])}
        sd2 = {"a": torch.tensor([3.0, 4.0])}
        sd3 = {"a": torch.tensor([5.0, 6.0])}
        result = fedavg([(sd1, 1), (sd2, 1), (sd3, 1)])
        assert torch.allclose(result["a"], torch.tensor([3.0, 4.0]))


# ── reputation_fedavg ─────────────────────────────────────────────────────────

class TestReputationFedAvg:
    def _sd(self, val: float) -> dict:
        return {"w": torch.full((4,), val)}

    def test_equal_reputations_equal_sample_counts_is_mean(self):
        sd1, sd2 = self._sd(0.0), self._sd(4.0)
        result = reputation_fedavg([(sd1, 1, 0.5), (sd2, 1, 0.5)])
        assert torch.allclose(result["w"], torch.full((4,), 2.0))

    def test_higher_reputation_gets_more_weight(self):
        sd_low, sd_high = self._sd(0.0), self._sd(10.0)
        result = reputation_fedavg([(sd_low, 1, 0.1), (sd_high, 1, 0.9)])
        # Expected: 0.1*0.0 + 0.9*10.0) / (0.1 + 0.9) = 9.0
        assert float(result["w"][0]) > 5.0

    def test_zero_reputations_falls_back_to_fedavg(self):
        sd1, sd2 = self._sd(2.0), self._sd(8.0)
        result = reputation_fedavg([(sd1, 1, 0.0), (sd2, 1, 0.0)])
        assert torch.allclose(result["w"], torch.full((4,), 5.0))

    def test_negative_reputation_clipped_to_zero(self):
        sd1, sd2 = self._sd(0.0), self._sd(10.0)
        # Negative reputation should count as 0 weight
        result = reputation_fedavg([(sd1, 1, -5.0), (sd2, 1, 1.0)])
        assert torch.allclose(result["w"], torch.full((4,), 10.0))

    def test_result_is_not_reference_to_input(self):
        sd = self._sd(5.0)
        result = reputation_fedavg([(sd, 1, 1.0)])
        result["w"].fill_(0.0)
        assert torch.allclose(sd["w"], torch.full((4,), 5.0))
