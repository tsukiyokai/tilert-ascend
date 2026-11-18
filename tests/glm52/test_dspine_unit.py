import torch


def test_calibrated_l78_zero_init_and_folding_parity():
    from tilert.models.glm_5_2_ascend.dspine import (
        CalibratedL78Config,
        CalibratedL78Model,
        TpCalibratedL78Weights,
    )

    torch.manual_seed(41)
    cfg = CalibratedL78Config(
        hidden_size=128,
        q_lora_rank=64,
        num_local_heads=8,
        qk_nope_dim=24,
        kv_lora_rank=512,
        v_head_dim=256,
        num_experts=32,
        eh_local_rows=32,
        rot_local_cols=16,
        max_steps=4,
        rot_rank=8,
        eh_rank=8,
        kv_rank=4,
        router_rank=4,
        head_rank=8,
    )
    model = CalibratedL78Model(cfg)
    rot_col = torch.randn(128, 16, dtype=torch.bfloat16)
    w_eh = torch.randn(32, 128, dtype=torch.bfloat16)
    w_uk0 = torch.randn(8, 24, 512, dtype=torch.bfloat16)
    w_uv0 = torch.randn(8, 512, 256, dtype=torch.bfloat16)
    r_w0 = torch.randn(128, 32, dtype=torch.bfloat16)

    rw0, rb0, _, _, weh0 = model.fold_step_prolog(0, rot_col, w_eh)
    uk0, uv0, rt0 = model.fold_layer_weights(w_uk0, w_uv0, r_w0)
    assert torch.equal(rw0, rot_col)
    assert torch.equal(rb0, torch.zeros_like(rb0))
    assert torch.equal(weh0, w_eh)
    assert torch.equal(uk0, w_uk0)
    assert torch.equal(uv0, w_uv0)
    assert torch.equal(rt0, r_w0)
    assert torch.allclose(model.head_matrix_f32(), torch.eye(128))

    rec = torch.randn(3, 128)
    loss = (
        model.fold_step_prolog(2, rot_col, w_eh)[0].float().sum()
        + model.fold_layer_weights(w_uk0, w_uv0, r_w0)[0].float().sum()
        + model.apply_head_proj(rec).sum()
    )
    loss.backward()
    assert model.rot_u.grad.abs().max() > 0
    assert model.uk_u.grad.abs().max() > 0
    assert model.head_p.grad.abs().max() > 0

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.08 * torch.randn_like(p))
        rw, rb, _, _, _ = model.fold_step_prolog(1, rot_col, w_eh)
        rc_f = rot_col.float()
        unfolded = (
            (
                rec * model.rot_scale[1]
                + (16.0 / cfg.rot_rank) * ((rec @ model.rot_u[1]) @ model.rot_v[1])
                + model.rot_bias[1]
            )
            @ rc_f
        )
        folded = rec @ rw.float() + rb.float()
        assert torch.allclose(unfolded, folded, atol=5e-2, rtol=2e-2)

        shards = []
        for r in range(8):
            shards.append({
                "tp_rank": r,
                "rot_w_step": torch.stack([rw] * 4),
                "rot_b_step": torch.stack([rb] * 4),
                "w_eh_step": torch.stack([weh0] * 4),
                "cal_w_uk_t": uk0,
                "cal_w_uv": uv0,
            })
        blob = {
            "max_d": 4,
            "enorm_step": model.enorm_step.to(torch.bfloat16),
            "hnorm_step": model.hnorm_step.to(torch.bfloat16),
            "logit_norm_step": model.logit_norm_step.to(torch.bfloat16),
            "cal_input_gamma": model.cal_input_gamma.to(torch.bfloat16),
            "cal_q_a_norm": model.cal_q_a_norm.to(torch.bfloat16),
            "cal_post_gamma": model.cal_post_gamma.to(torch.bfloat16),
            "cal_router_w": rt0,
            "m_head_f32": model.head_matrix_f32(),
            "shards": shards,
        }
        tp_wt = TpCalibratedL78Weights(blob, 0, torch.device("cpu"))
        assert tp_wt.cal_w_uv_packed.shape == (16, 512, 128)
        assert tp_wt.cal_router_w_packed.shape == (128, 32)
        ref_rt_packed = rt0.reshape(128, 16, 2).permute(1, 0, 2).contiguous().view(128, 32)
        assert torch.equal(tp_wt.cal_router_w_packed, ref_rt_packed)
        assert tp_wt.step_idx(1) == 0 and tp_wt.step_idx(9) == 3

        head_sh = torch.randn(20, 128, dtype=torch.bfloat16)
        folded_h = tp_wt.fold_head_shard(head_sh, valid_vocab=16, inv_temp=1.1)
        ref_logits = (model.apply_head_proj(rec) * 1.1) @ head_sh[:16].float().t()
        got_logits = rec @ folded_h[:16].t()
        assert torch.allclose(ref_logits, got_logits, atol=1e-4, rtol=1e-4)
        assert float(folded_h[16:].abs().max()) == 0.0
        q8_t, s8 = tp_wt.quantize_head_q8_nz(head_sh, valid_vocab=16)
        assert q8_t.shape == (128, 20) and q8_t.dtype == torch.int8 and s8.shape == (20,)


def test_horizon_controller_hysteresis_and_beam_specs():
    from tilert.models.glm_5_2_ascend.dspine import HorizonController, build_beam_specs

    specs7, depths7 = build_beam_specs([4, 4, 4, 4, 4, 2, 2])
    specs10, depths10 = build_beam_specs([4, 4, 4, 4, 2, 2, 2, 2, 1, 1])
    assert len(specs7) == 7 and max(depths7) == 8 and len(depths7) <= 256
    assert len(specs10) == 10 and max(depths10) == 11 and len(depths10) <= 256

    ctrl = HorizonController(init_ema=6.8, alpha=0.20, up_thresh=7.30, down_thresh=7.10)
    assert not ctrl.select_high()
    for acc in [4, 5, 4, 3, 5, 4]:
        assert not ctrl.update(acc)

    ctrl.reset()
    assert not ctrl.update(8)
    assert not ctrl.update(8)
    assert ctrl.update(8)
    assert ctrl.select_high()
    assert ctrl.update(7)
    assert not ctrl.update(5)


def test_calibrated_l78_v3_folding_and_p2_controls():
    from tilert.models.glm_5_2_ascend.dspine import (
        TpCalibratedL78Weights,
        ZeroLagHorizonController,
        build_beam_specs,
        select_optimal_w_act,
    )

    specs11, depths11 = build_beam_specs([4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1])
    assert len(specs11) == 11 and max(depths11) == 12 and len(depths11) <= 256

    zctrl = ZeroLagHorizonController(
        num_levels=3,
        init_level=1,
        init_ema=6.4,
        alpha=0.28,
        mid_thresh=5.5,
        up_thresh=7.2,
        down_thresh=4.9,
        drop_acc=3,
        jump_acc=9,
        streak_acc=7,
    )
    assert zctrl.select_level() == 1 and zctrl.select_high()
    assert zctrl.update(2) == 0 and not zctrl.select_high()
    assert zctrl.update(9) == 2 and zctrl.select_level() == 2
    zctrl.reset()
    assert zctrl.select_level() == 1
    assert zctrl.update(7) == 1
    assert zctrl.update(8) == 2
    assert zctrl.update(3) in (1, 2)
    assert zctrl.update(3) == 0

    low_conf = [0.0] + [-9.0] * 15
    high_conf = [0.0] + [-0.02] * 15
    assert select_optimal_w_act(low_conf, width=16, min_w_act=8, t_base_ms=47.0, c_w_ms=0.72) == 8
    assert select_optimal_w_act(high_conf, width=16, min_w_act=8, t_base_ms=47.0, c_w_ms=0.72) == 16
    assert select_optimal_w_act(low_conf, width=16, min_w_act=16, t_base_ms=47.0, c_w_ms=0.72) == 16

    torch.manual_seed(73)
    h = 6144
    shards = []
    for r in range(8):
        shards.append({
            "tp_rank": r,
            "rot_w_step": torch.randn(11, h, 768, dtype=torch.bfloat16),
            "rot_b_step": torch.randn(11, 768, dtype=torch.bfloat16),
            "w_eh_step": torch.randn(11, 1536, h, dtype=torch.bfloat16),
            "cal_w_uq_qr": torch.randn(2048, 2048, dtype=torch.bfloat16),
            "cal_w_uk_t": torch.randn(8, 192, 512, dtype=torch.bfloat16),
            "cal_w_uv": torch.randn(8, 512, 256, dtype=torch.bfloat16),
            "cal_w_o": torch.randn(2048, h, dtype=torch.bfloat16),
            "cal_shared_gu": torch.randn(h, 512, dtype=torch.bfloat16),
            "cal_shared_down": torch.randn(256, h, dtype=torch.bfloat16),
            "cal_routed_w13_mul": torch.full((32, 4096), 1.25, dtype=torch.float32),
            "cal_routed_w2_mul": torch.full((32, h), 0.80, dtype=torch.float32),
        })
    blob = {
        "version": 3,
        "max_d": 11,
        "tree_calib": {"beta_margin": 0.08, "beta_depth": 0.04},
        "enorm_step": torch.ones(11, h, dtype=torch.bfloat16),
        "hnorm_step": torch.ones(11, h, dtype=torch.bfloat16),
        "logit_norm_seed0": torch.full((h,), 1.05, dtype=torch.bfloat16),
        "logit_norm_step": torch.ones(11, h, dtype=torch.bfloat16),
        "cal_input_gamma": torch.ones(h, dtype=torch.bfloat16),
        "cal_q_a_norm": torch.ones(2048, dtype=torch.bfloat16),
        "cal_post_gamma": torch.ones(h, dtype=torch.bfloat16),
        "cal_router_w": torch.randn(h, 256, dtype=torch.bfloat16),
        "m_head_f32": torch.eye(h, dtype=torch.float32),
        "shards": shards,
    }
    wt = TpCalibratedL78Weights(blob, 0, torch.device("cpu"))
    assert wt.version == 3 and wt.max_d == 11
    assert abs(wt.beta_margin - 0.08) < 1e-6 and abs(wt.beta_depth - 0.04) < 1e-6
    assert wt.logit_norm_seed0 is not None and wt.logit_norm_seed0.shape == (h,)
    assert wt.cal_uq_qr_q8_nz is not None and wt.cal_uq_qr_scale.shape == (1, 2048)
    assert wt.cal_w_o_packed is not None and wt.cal_w_o_scale.shape == (h,)
    assert wt.cal_shared_packed is not None and len(wt.cal_shared_packed) == 4
    assert wt.step_idx(11) == 10 and wt.step_idx(12) == 10

    orig_routed = [
        torch.zeros((32, h, 512), dtype=torch.int32),
        torch.full((32, 4096), 2.0, dtype=torch.float32),
        torch.full((32, 4096), 4.0, dtype=torch.float32),
        torch.zeros((32, 2048, h // 8), dtype=torch.int32),
        torch.full((32, h), 5.0, dtype=torch.float32),
        torch.full((32, h), 10.0, dtype=torch.float32),
    ]
    folded_routed = wt.fold_routed_weights(orig_routed)
    assert folded_routed[0] is orig_routed[0] and folded_routed[3] is orig_routed[3]
    assert torch.allclose(folded_routed[1], torch.full((32, 4096), 2.5))
    assert torch.allclose(folded_routed[2], torch.full((32, 4096), 5.0))
    assert torch.allclose(folded_routed[4], torch.full((32, h), 4.0))
    assert torch.allclose(folded_routed[5], torch.full((32, h), 8.0))

    hw = torch.randn(32, h, dtype=torch.bfloat16)
    folded_hw = wt.fold_head_shard(hw, valid_vocab=24, inv_temp=0.95)
    assert folded_hw.shape == (32, h) and torch.all(folded_hw[24:] == 0)
    q8_t, s8 = wt.quantize_head_q8_nz(hw, valid_vocab=24, inv_temp=0.95)
    assert q8_t.shape == (h, 32) and q8_t.dtype == torch.int8 and s8.shape == (32,)
