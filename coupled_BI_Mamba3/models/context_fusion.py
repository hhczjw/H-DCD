"""
ContextFusion: 对话上下文特征级拼接 (Phase 5) + 双向增强 (Phase 17)
=================================================================

CAGMamba 对齐: context→main 和 main→context 双向 CHM 增强。
两种增强后的特征拼接送入最终预测。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def encode_with_context(
    model,
    text: torch.Tensor,
    audio: torch.Tensor,
    video: torch.Tensor,
    context_text: torch.Tensor,
    context_audio: torch.Tensor,
    context_video: torch.Tensor,
    cu_seqlens=None,
    return_ism_cls: bool = False,
    bidirectional: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    编码 context + main, 构建对话级序列 [context, main] → CoupledMamba3Fork.

    ★ Phase 17: 双向上下文 (CAGMamba 对齐)
      - 正向: context→main (main 在 context 条件下被增强)
      - 反向: main→context (context 在 main 条件下被增强)
      - 两种增强结果拼接返回

    流程:
      1) 分别编码 main 和 context 话语
      2) 池化为 (B, d_model) 特征向量
      3) stack([context, main], dim=1) → (B, 2, d_model)
      4) CoupledMamba3Fork 对话级跨模态融合
      5) 正向取 t=1 (main), 反向取 t=1 (context)

    Args:
        model: MSAClassifier 实例
        bidirectional: 是否启用双向上下文 (Phase 17)

    Returns:
        如果 bidirectional=False: out_l, out_a, out_v (正向的 main 输出)
        如果 bidirectional=True:  out_l, out_a, out_v,
                                    ctx_l, ctx_a, ctx_v (双向增强)
        如果 return_ism_cls=True, 额外返回 (c_t, c_a, c_v)
    """
    # 1) 编码 main 话语
    if return_ism_cls:
        out_l, out_a, out_v, c_t, c_a, c_v = model._encode(
            text, audio, video, cu_seqlens, return_ism_cls=True,
        )
    else:
        out_l, out_a, out_v = model._encode(
            text, audio, video, cu_seqlens,
        )

    # 2) 编码 context 话语
    ctx_l, ctx_a, ctx_v = model._encode(
        context_text, context_audio, context_video, cu_seqlens,
    )

    # 3) 池化单话语 → (B, d_model)
    m_l = model._pool(out_l)
    m_a = model._pool(out_a)
    m_v = model._pool(out_v)
    c_l = model._pool(ctx_l)
    c_a = model._pool(ctx_a)
    c_v = model._pool(ctx_v)

    # 4) 构建正向序列: stack([context, main], dim=1) → (B, 2, d_model)
    fwd_l = torch.stack([c_l, m_l], dim=1)
    fwd_a = torch.stack([c_a, m_a], dim=1)
    fwd_v = torch.stack([c_v, m_v], dim=1)

    # 5) CoupledMamba3Fork 对话级跨模态融合 (正向: context→main)
    for layer in model.layers:
        fwd_a, fwd_v, fwd_l = layer(fwd_a, fwd_v, fwd_l, cu_seqlens=cu_seqlens)

    # 6) 取 t=1 (main) 位置的输出 — 正向 context→main 增强
    out_l = fwd_l[:, 1, :]  # (B, d_model)
    out_a = fwd_a[:, 1, :]
    out_v = fwd_v[:, 1, :]

    # 7) ★ Phase 17: 反向 main→context
    if bidirectional:
        # 构建反向序列: stack([main, context], dim=1)
        rev_l = torch.stack([m_l, c_l], dim=1)
        rev_a = torch.stack([m_a, c_a], dim=1)
        rev_v = torch.stack([m_v, c_v], dim=1)

        for layer in model.layers:
            rev_a, rev_v, rev_l = layer(rev_a, rev_v, rev_l, cu_seqlens=cu_seqlens)

        # 取 t=1 (context) 位置的输出 — main→context 增强
        rev_l = rev_l[:, 1, :]  # context 被 main 增强
        rev_a = rev_a[:, 1, :]
        rev_v = rev_v[:, 1, :]

        if return_ism_cls:
            return out_l, out_a, out_v, rev_l, rev_a, rev_v, c_t, c_a, c_v
        return out_l, out_a, out_v, rev_l, rev_a, rev_v

    if return_ism_cls:
        return out_l, out_a, out_v, c_t, c_a, c_v
    return out_l, out_a, out_v
