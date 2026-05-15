# Coupled-BI-Mamba3 吸收 CAGMamba 优点的完整升级方案

> **目标**: 保留 Coupled-BI-Mamba3 的核心创新（Mamba-3 SSM、Cross-Modal QKV 注入），
> 吸收 CAGMamba 的关键优势（预训练编码器、对话上下文、门控融合、多任务训练、精细超参），
> 两者结合达到最优性能。

---

## 目录

1. [改造前整体架构](#1-改造前整体架构)
2. [改造后整体架构](#2-改造后整体架构)
3. [改造点详细分析与代码实现](#3-改造点详细分析与代码实现)
4. [配置文件更新](#4-配置文件更新)
5. [环境配置与依赖安装](#5-环境配置与依赖安装)
6. [验证路线](#6-验证路线)

---

## 1. 改造前整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    改造前: Coupled-BI-Mamba3                        │
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │  BERT-base  │   │ COVAREP     │   │ FACET       │               │
│  │  text_bert  │   │ audio_feat  │   │ vision_feat │               │
│  │  (768 dim)  │   │ (5~74 dim)  │   │ (20~35 dim) │               │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
│         ↓                 ↓                 ↓                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ proj_text    │ │ proj_audio   │ │ proj_video   │                │
│  │ Linear(→128) │ │ Linear(→128) │ │ Linear(→128) │                │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                │
│         ↓                 ↓                 ↓                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ ISMEncoder×3 │ │ ISMEncoder×3 │ │ ISMEncoder×3 │                │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                │
│         └────────┬──────┴────────┬──────┘                           │
│                  ↓               ↓                                  │
│        ┌─────────────────────────────┐                              │
│        │  CoupledMamba3Fork × 2      │  ◀── 三模态 QKV 跨模态融合  │
│        │  (v_self_ratio=0, 无门控)   │  ◀── B/V 完全来自源模态     │
│        │  (帧级序列 L=50)             │                              │
│        └──────────────┬──────────────┘                              │
│                       ↓                                             │
│                  Mean Pool → Concat → Linear Head                   │
│                  (无上下文, 无sub_loss, dropout=0.1)                │
└─────────────────────────────────────────────────────────────────────┘

关键特征:
  - 文本: BERT-base-uncased (12层, 768维), 离线 .pkl 特征
  - 音频: COVAREP 手工特征 (5~74维), Linear 直接映射, 信息瓶颈严重
  - 视频: FACET 手工特征 (20~35维), 潜在噪声源
  - 跨模态: Q 来自目标模态, B/V 全部由源模态加权提供 (v_self_ratio=0)
  - 上下文: ❌ 无需对话上下文信息
  - ISM: depth=3 (约 1.76M 参数), 小数据集上易过拟合
  - 训练: 2 组参数 (BERT vs Other), dropout=0.1, 无模态级辅助损失
```

---

## 2. 改造后整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│         改造后: Coupled-BI-Mamba3 + CAGMamba 优势吸收                     │
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│   │RoBERTa-base  │    │Data2Vec-Audio│    │FACET/CLIP    │               │
│   │(768/1024 dim)│    │(768 dim)     │    │(20~768 dim)  │               │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│          ↓                   ↓                   ↓                       │
│   ┌──────────────┐  ┌──────────────┐   ┌──────────────┐                 │
│   │ proj_text    │  │ proj_audio   │   │ proj_video   │                 │
│   │ Linear(→128) │  │ Linear(→128) │   │ Linear(→128) │                 │
│   └──────┬───────┘  └──────┬───────┘   └──────┬───────┘                 │
│          ↓                   ↓                   ↓                       │
│   ┌──────────────┐  ┌──────────────┐   ┌──────────────┐                 │
│   │ ISMEncoder×1 │  │ ISMEncoder×1 │   │ ISMEncoder×1 │ ◀── depth↓      │
│   └──────┬───────┘  └──────┬───────┘   └──────┬───────┘                 │
│          └─────────┬───────┴────────┬──────────┘                         │
│                    ↓                ↓                                    │
│          ┌────────────────────────────────┐                              │
│          │  上下文感知编码 (Context Encoder) │                              │
│          │  ┌──────────────────────────┐ │                              │
│          │  │ ISM → Pool → ctx_vec     │ │ ◀── Context utterance        │
│          │  │ ISM → Pool → main_vec    │ │ ◀── Main utterance           │
│          │  │ Stack → (B, 2, d_model)  │ │                              │
│          │  └──────────────────────────┘ │                              │
│          └──────────────┬─────────────────┘                              │
│                         ↓                                                │
│         ┌────────────────────────────────────┐                          │
│         │ CoupledMamba3Fork (L=2) × 2        │  ◀── 上下文 + 跨模态融合 │
│         │ V 通道: λ·V_self + (1-λ)·V_cross   │  ◀── v_self_ratio=0.3   │
│         │ SSM: h₀(context) → h₁(main)        │  ◀── 情感转移建模       │
│         └──────────────┬─────────────────────┘                          │
│                        ↓ 取 t=1 (main)                                   │
│                   ┌──────────┐                                          │
│                   │ Mean Pool│                                          │
│                   └────┬─────┘                                          │
│                        ↓                                                │
│              ┌─────────────────────┐                                    │
│              │ Concat → LayerNorm  │                                    │
│              └──────────┬──────────┘                                    │
│                         ↓                                               │
│              ┌──────────────────────┐                                   │
│              │    分类头 (Head)      │  → L_fused                        │
│              │ sub_fc_T  ← ISM CLS │  → L_T (sub_loss_lambda=0.3)     │
│              │ sub_fc_A  ← ISM CLS │  → L_A                            │
│              │ sub_fc_V  ← ISM CLS │  → L_V                            │
│              └──────────────────────┘                                   │
│                                                                          │
│   Total Loss = L_fused + λ·(L_T + L_A + L_V)  (λ=0.3)                  │
└──────────────────────────────────────────────────────────────────────────┘

关键改进:
  - 文本: RoBERTa-base (12层, 768维, BPE 词表), 可选在线编码 + fine-tune
  - 音频: Data2Vec-Audio-base (12层 Transformer, 768维), 可选在线编码
  - 视频: 保持手工特征, 但通过 v_self_ratio 降低噪声污染 (也可完全移除)
  - 上下文: ✅ 对话上下文 (video_id 分组, context→main 序列建模)
  - V 通道: v_self_ratio=0.3 保留目标模态自身 V 锚点, 抗噪声
  - ISM: depth=1 (参数减少 67%), 降低过拟合
  - 训练: 4 组参数, dropout=0.25, sub_loss_lambda=0.3, 多任务监督
```

---

## 3. 改造点详细分析与代码实现

### 3.0 设计原则: 避免原方案的架构错误

| 原方案问题 | 本方案修复 |
|-----------|-----------|
| Phase 6 gate 残差重复相加 (u_tgt 被加两次) | Gate 输出纯残差增量, 外层 CoupledMamba3Fork 统一做 `y + u_tgt` |
| Phase 5 CoupledMamba3Fork 被调用三次 (ctx+main+stack) | 每句话只走到 ISM 为止, 仅堆叠后的序列走 CoupledMamba3Fork |
| Phase 5 np.roll 不尊重对话边界 | 使用 video_id 分组 + 索引偏移 |
| Phase 5 `__init_mosi` 不存在 | 直接修改通用 MMDataset.__init__ |
| Phase 5 sub_loss TODO (互斥) | _encode 增加 with_fusion 参数, 上下文路径天然支持 return_ism_cls |

---

### 3.1 文本编码器升级 (BERT → RoBERTa)

#### 改造分析

| 对比 | BERT-base | RoBERTa-base |
|------|-----------|-------------|
| 层数 | 12 | 12 |
| 隐藏维度 | 768 | 768 |
| 词表 | 30K (WordPiece) | 50K (BPE) |
| Masking | Static | Dynamic |
| token_type_ids | 需要 | 不需要 |

RoBERTa-base 与 BERT-base 维度相同 (768), 替换后 `feature_dims[0]` 不变。
RoBERTa-large (1024 维) 需要调整 `proj_text` 维度。

#### 代码: models/classifier.py — TextPretrainedEncoder

在 `classifier.py` 中新增通用文本编码器, 替换原有的 `BertTextEncoder`:

```python
class TextPretrainedEncoder(nn.Module):
    """
    通用文本预训练编码器, 支持 BERT/RoBERTa.
    与原有 BertTextEncoder 的差异:
        1. 支持 RoBERTa (无 token_type_ids)
        2. 自动检测模型类型
        3. 支持在线编码 (原始文本) 和离线特征 (三通道 pkl)
        4. 自动获取输出维度 (768/1024)
    """

    def __init__(self, pretrained: str = "roberta-base", finetune: bool = True,
                 strict: bool = True):
        super().__init__()
        self.pretrained_name = pretrained
        self.use_hf = False

        # 自动检测模型类型
        pn = pretrained.lower()
        if "roberta" in pn:
            self.model_type = "roberta"
        elif "deberta" in pn:
            self.model_type = "deberta"
        else:
            self.model_type = "bert"

        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
            self.transformer = AutoModel.from_pretrained(pretrained)
            self.out_dim = self.transformer.config.hidden_size
            self.use_hf = True
        except Exception as e:
            if strict:
                raise RuntimeError(f"加载 {pretrained} 失败: {e}")
            print(f"[WARN] 回退到 Embedding: {e}")
            self.transformer = nn.Embedding(50265, 768, padding_idx=1)
            self.tokenizer = None
            self.out_dim = 768

        if self.use_hf and not finetune:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def forward(self, text_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_input:
                - 离线三通道模式: (B, 3, L)  [ids, mask, segment]
                - 在线模式: (B, L) 仅 input_ids
        Returns:
            hidden:         (B, L, out_dim)
            attention_mask: (B, L) long
        """
        if text_input.dim() == 3 and text_input.size(1) == 3:
            # 兼容旧的三通道 pkl 格式
            input_ids = text_input[:, 0].long()
            attention_mask = text_input[:, 1].long()
            token_type_ids = text_input[:, 2].long()
        elif text_input.dim() == 2:
            input_ids = text_input.long()
            attention_mask = (input_ids != (self.tokenizer.pad_token_id
                              if self.tokenizer else 0)).long()
            token_type_ids = None
        else:
            input_ids = text_input.squeeze(1).long()
            attention_mask = (input_ids != 0).long()
            token_type_ids = None

        if not self.use_hf:
            return self.transformer(input_ids), attention_mask

        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        # RoBERTa/DeBERTa 不需要 token_type_ids
        if self.model_type == "bert" and token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids

        out = self.transformer(**kw)
        return out.last_hidden_state, attention_mask
```

在 `MSAClassifier.__init__` 中替换:

```python
if use_bert:
    self.text_encoder = TextPretrainedEncoder(
        pretrained=bert_pretrained,   # 默认 "roberta-base"
        finetune=bert_finetune,
    )
    text_feat_dim = self.text_encoder.out_dim  # 自动获取 (768 或 1024)
else:
    self.text_encoder = None
    text_feat_dim = text_input_dim
```

---

### 3.2 音频编码器升级 (COVAREP → Data2Vec-Audio)

#### 改造分析

| 对比 | COVAREP | Data2Vec-Audio-base |
|------|---------|-------------------|
| 特征维度 | 5~74 | 768 |
| 类型 | 手工特征 | 自监督预训练 |
| 信息量 | 低级声学特征 | 高级语义表征 |
| Fine-tune | N/A | 可端到端训练 |

提供两种方案:
- **方案 A (在线编码)**: 需要原始 `.wav` 文件, 模型内部调用 Data2Vec, 可 fine-tune
- **方案 B (离线特征)**: 预提取 .pkl 文件, 不改变数据加载流程

#### 代码: models/classifier.py — AudioPretrainedEncoder

```python
class AudioPretrainedEncoder(nn.Module):
    """
    预训练音频编码器 (Data2Vec / HuBERT / WavLM).

    使用方式:
        1. 在线编码: forward 接收原始波形 (B, T_wav)
        2. 离线编码: forward 接收 (B, L, 768) 预提取特征
    """

    def __init__(self, pretrained: str = "facebook/data2vec-audio-base-960h",
                 finetune: bool = True, strict: bool = True):
        super().__init__()
        self.pretrained_name = pretrained
        self.use_hf = False

        try:
            from transformers import AutoModel, AutoFeatureExtractor
            self.processor = AutoFeatureExtractor.from_pretrained(pretrained)
            self.transformer = AutoModel.from_pretrained(pretrained)
            self.out_dim = self.transformer.config.hidden_size
            self.use_hf = True
        except Exception as e:
            if strict:
                raise RuntimeError(f"加载 {pretrained} 失败: {e}")
            print(f"[WARN] 音频编码器加载失败, fallback: {e}")
            self.transformer = None
            self.processor = None
            self.out_dim = 768

        if self.use_hf and not finetune:
            for p in self.transformer.parameters():
                p.requires_grad = False

    def forward(self, audio_input: torch.Tensor,
                audio_lengths: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            audio_input:
                - 在线模式: (B, T_wav) 原始波形
                - 离线模式: (B, L, D) 预提取特征
            audio_lengths: 在线模式下可选的有效长度 (用于 mask)
        Returns:
            hidden: (B, L', out_dim) 特征序列
        """
        if not self.use_hf:
            return audio_input

        # 检测是否已在特征空间 (离线模式)
        if audio_input.dim() == 3:
            return audio_input  # (B, L, D) 直接返回

        # 在线编码: 原始波形 → Data2Vec
        with torch.inference_mode() if not self.training else torch.enable_grad():
            inputs = self.processor(
                audio_input.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16000 * 30,  # 30 秒
            ).to(audio_input.device)
            outputs = self.transformer(
                inputs["input_values"].squeeze(1),
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=False,
            )
        return outputs.last_hidden_state  # (B, T, 768)
```

在 `MSAClassifier.__init__` 中添加:

```python
self.use_audio_encoder = bool(getattr(args, "use_audio_encoder", False))
if self.use_audio_encoder:
    self.audio_encoder = AudioPretrainedEncoder(
        pretrained=getattr(args, "audio_pretrained", "facebook/data2vec-audio-base-960h"),
        finetune=getattr(args, "audio_finetune", True),
    )
    audio_feat_dim = self.audio_encoder.out_dim
else:
    self.audio_encoder = None
    audio_feat_dim = audio_input_dim
```

---

### 3.3 对话上下文引入 (核心改进)

#### 改造分析

CAGMamba 的核心优势之一是按 `video_id` 分组加载 context utterance，然后在 Mamba CHM 中建模 `context → main` 的情感转移。

**设计要点**:
1. 数据加载时按 `video_id` 分组, 取前一条同 `video_id` 的话语作为 context
2. Context 和 Main 各自走完 ISM 后 pool 为向量, 堆叠为 `[ctx_vec, main_vec]` (L=2)
3. CoupledMamba3Fork 在 L=2 序列上做跨模态融合 + 上下文 SSM 状态传递
4. 取 t=1 (main 位置) 的输出做分类

**数据加载修改**: 使用 `id` 字段推断 `video_id` (MOSI/MOSEI 的 id 格式: `videoId_segmentNum`)

#### 代码: dataset/data_loader.py — 上下文加载

```python
def _infer_video_id(sample_id: str) -> str:
    """从 sample id 推断 video_id (格式: 'videoId_segmentNum' 或 'videoId###seg')"""
    # MOSI/MOSEI 格式: "videoId_segmentNum"
    for sep in ["_", "###", "#"]:
        parts = sample_id.split(sep)
        if len(parts) >= 2 and parts[0]:
            return parts[0]
    return sample_id  # fallback: 整个 id 作为 video_id


class MMDataset(Dataset):
    def __init__(self, args: Any, mode: str = "train"):
        # ... 原有初始化代码保持不变 ...

        # ===== 新增: 对话上下文加载 =====
        self.use_context = bool(getattr(args, "use_context", False))
        if self.use_context:
            self._load_context()

    def _load_context(self):
        """按 video_id 分组加载 context utterance. """
        # 1) 从 sample ids 推断 video_id
        video_ids = []
        for sid in self.ids:
            vid = _infer_video_id(sid)
            video_ids.append(vid)

        # 2) 为每个样本找到同 video_id 的前一条 utterance 的索引
        self.context_indices = []
        for i in range(len(self.ids)):
            ctx_idx = self._find_prev_in_same_video(i, video_ids)
            self.context_indices.append(ctx_idx)

        n_found = sum(1 for c in self.context_indices if c != -1)
        n_total = len(self.context_indices)
        self.logger.info(
            f"[{self.mode}] Context loaded: {n_found}/{n_total} "
            f"have previous utterance in same video"
        )

    def _find_prev_in_same_video(self, idx: int, video_ids: List[str]) -> int:
        """从 idx 开始向前搜索同 video_id 的最近话语. """
        if idx <= 0:
            return -1
        cur_vid = video_ids[idx]
        # 向前搜索最多 10 条
        for offset in range(1, min(idx + 1, 11)):
            candidate = idx - offset
            if video_ids[candidate] == cur_vid:
                return candidate
        return -1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "text": torch.from_numpy(self.text[idx]),
            "audio": torch.from_numpy(self.audio[idx].copy()),
            "vision": torch.from_numpy(self.vision[idx].copy()),
            "id": self.ids[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }

        # ===== 新增: 上下文特征 =====
        if self.use_context:
            cidx = self.context_indices[idx]
            if cidx >= 0:
                sample["context_text"] = torch.from_numpy(self.text[cidx])
                sample["context_audio"] = torch.from_numpy(self.audio[cidx].copy())
                sample["context_vision"] = torch.from_numpy(self.vision[cidx].copy())
            else:
                # 无语境时用自身复制 (退化为无上下文效果)
                sample["context_text"] = torch.from_numpy(self.text[idx])
                sample["context_audio"] = torch.from_numpy(self.audio[idx].copy())
                sample["context_vision"] = torch.from_numpy(self.vision[idx].copy())

        labels = {}
        for tag, arr in self.labels.items():
            if self.dataset_name in REGRESSION_DATASETS:
                labels[tag] = torch.tensor(arr[idx], dtype=torch.float32)
            else:
                labels[tag] = torch.tensor(arr[idx], dtype=torch.long)
        sample["labels"] = labels
        sample["label"] = labels["M"]
        return sample
```

`_collate_fn` 中需要处理新增的 context 字段 (context 字段也是 Tensor, 会自动 stack):

```python
def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "labels":
            sub_keys = vals[0].keys()
            out["labels"] = {sk: torch.stack([v[sk] for v in vals], dim=0) for sk in sub_keys}
        elif k == "id":
            out["ids"] = vals
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
```

#### 代码: models/classifier.py — MSAClassifier 上下文改造

核心改造: `_encode` 增加 `with_fusion` 参数, 控制是否走 CoupledMamba3Fork。
新增 `forward` 中的上下文路径。

```python
class MSAClassifier(nn.Module):

    def _encode(self, text, audio, video, cu_seqlens=None,
                return_ism_cls: bool = False, with_fusion: bool = True):
        """编码到融合后的三模态表征.

        Args:
            with_fusion: True=走完整 ISM+CoupledMamba3Fork; False=只走到 ISM 为止
        """
        # 0) 文本嵌入
        if self.use_bert and self.text_encoder is not None:
            text, _ = self.text_encoder(text)

        # 1) 投影到 d_model
        xt = self.proj_text(text)
        xa = self.proj_audio(audio)
        xv = self.proj_video(video)

        # 2) 序列对齐
        Lt = xt.size(1)
        if xa.size(1) != Lt:
            xa = F.adaptive_avg_pool1d(xa.transpose(1, 2), Lt).transpose(1, 2)
        if xv.size(1) != Lt:
            xv = F.adaptive_avg_pool1d(xv.transpose(1, 2), Lt).transpose(1, 2)

        # 3) ISM (各模态独立)
        ism_cls_t = ism_cls_a = ism_cls_v = None
        if self.ism_depth > 0:
            if return_ism_cls:
                xt, ism_cls_t = self.ism_text(xt, return_cls=True)
                xa, ism_cls_a = self.ism_audio(xa, return_cls=True)
                xv, ism_cls_v = self.ism_video(xv, return_cls=True)
            else:
                xt = self.ism_text(xt)
                xa = self.ism_audio(xa)
                xv = self.ism_video(xv)

        # 如果不需要融合, 就在这里返回 (供上下文路径使用)
        if not with_fusion:
            if return_ism_cls:
                return xt, xa, xv, ism_cls_t, ism_cls_a, ism_cls_v
            return xt, xa, xv

        # 4) CoupledMamba3Fork (跨模态融合)
        out_a, out_v, out_l = xa, xv, xt
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        if return_ism_cls:
            return out_l, out_a, out_v, ism_cls_t, ism_cls_a, ism_cls_v
        return out_l, out_a, out_v

    def forward(self, text, audio, video, cu_seqlens=None,
                audio_lengths=None, vision_lengths=None,
                context_text=None, context_audio=None, context_video=None):
        """
        新增: context_text/context_audio/context_video 为可选的上下文特征.
        提供时走上下文感知路径, 否则走原始单话语路径.
        """
        has_context = all(x is not None for x in
                          [context_text, context_audio, context_video])

        if has_context:
            return self._forward_with_context(
                text, audio, video,
                context_text, context_audio, context_video,
                cu_seqlens,
            )

        # ===== 原始路径 (无上下文) =====
        if self.use_sub_loss:
            out_l, out_a, out_v, c_t, c_a, c_v = self._encode(
                text, audio, video, cu_seqlens, return_ism_cls=True,
            )
        else:
            out_l, out_a, out_v = self._encode(text, audio, video, cu_seqlens)
            c_t = c_a = c_v = None

        pl = self._pool(out_l)
        pa = self._pool(out_a)
        pv = self._pool(out_v)
        fused = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        if not self.use_sub_loss:
            return logits
        out = {"logits": logits}
        out["sub_T"] = self.sub_fc_T(c_t) if c_t is not None else None
        out["sub_A"] = self.sub_fc_A(c_a) if c_a is not None else None
        out["sub_V"] = self.sub_fc_V(c_v) if c_v is not None else None
        return out

    def _forward_with_context(self, text, audio, video,
                               context_text, context_audio, context_video,
                               cu_seqlens=None):
        """
        上下文感知前向路径:

        1. Context utterance: ISM → Pool → ctx_vec
        2. Main utterance:   ISM → Pool → main_vec  (含 ISM CLS 用于 sub_loss)
        3. Stack [ctx, main] → (B, 2, d_model) → CoupledMamba3Fork
        4. 取 t=1 (main 位置) 输出 → 分类头 + sub_loss
        """
        # ---- Step 1: 编码 Context (仅 ISM, 无融合) ----
        ctx_l, ctx_a, ctx_v = self._encode(
            context_text, context_audio, context_video,
            cu_seqlens, with_fusion=False,
        )
        ctx_l = self._pool(ctx_l)  # (B, d_model)
        ctx_a = self._pool(ctx_a)
        ctx_v = self._pool(ctx_v)

        # ---- Step 2: 编码 Main (ISM + ISM CLS 用于 sub_loss) ----
        if self.use_sub_loss:
            main_l, main_a, main_v, c_t, c_a, c_v = self._encode(
                text, audio, video, cu_seqlens,
                return_ism_cls=True, with_fusion=False,
            )
        else:
            main_l, main_a, main_v = self._encode(
                text, audio, video, cu_seqlens, with_fusion=False,
            )
            c_t = c_a = c_v = None

        main_l = self._pool(main_l)  # (B, d_model)
        main_a = self._pool(main_a)
        main_v = self._pool(main_v)

        # ---- Step 3: Stack [context, main] → (B, 2, d_model) ----
        out_l = torch.stack([ctx_l, main_l], dim=1)  # (B, 2, d_model)
        out_a = torch.stack([ctx_a, main_a], dim=1)
        out_v = torch.stack([ctx_v, main_v], dim=1)

        # ---- Step 4: CoupledMamba3Fork 跨模态融合 + 上下文传递 ----
        for layer in self.layers:
            out_a, out_v, out_l = layer(out_a, out_v, out_l, cu_seqlens=cu_seqlens)

        # ---- Step 5: 取 t=1 (main) 位置的输出 ----
        pl = out_l[:, 1, :]  # (B, d_model)
        pa = out_a[:, 1, :]
        pv = out_v[:, 1, :]

        # ---- 分类头 ----
        fused = self.fusion_norm(torch.cat([pl, pa, pv], dim=-1))
        logits = self.head(fused)

        if not self.use_sub_loss:
            return logits

        out = {"logits": logits}
        out["sub_T"] = self.sub_fc_T(c_t) if c_t is not None else None
        out["sub_A"] = self.sub_fc_A(c_a) if c_a is not None else None
        out["sub_V"] = self.sub_fc_V(c_v) if c_v is not None else None
        return out
```

**关键设计说明**:

- Context 和 Main **各自只走到 ISM**, 不走 CoupledMamba3Fork
- 只有 `[ctx_stack, main_vec]` 的 2-长度序列才走 CoupledMamba3Fork
- sub_loss 的 ISM CLS 从 Main 的 ISM 输出中获取, 与上下文路径天然兼容
- 无语境时退化到原始路径, 完全向后兼容

---

### 3.4 CoupledMamba3Fork 门控融合优化

#### 改造分析

原版 `v_self_ratio=0` 时, V/x 完全来自源模态, 目标模态自身 V 被丢弃。
本方案实现:
1. `v_self_ratio=0.3`: V/x = 0.3 × V_self + 0.7 × V_cross (已存在于代码中)
2. **显式门控 (新增)**: 仿照 CAGMamba, 使用可学习门控控制跨模态注入强度

#### 关键修复: 门控残差不可重复相加

原代码 (`coupled_mamba3_fork.py:440`):
```python
outs[tgt] = self.layer_norms[tgt](y + u_tgt)  # 外层已有残差
```

原方案代码在 cell 内部再做 `out = u_tgt + gate * out`, 导致 `u_tgt` 加了两次。
**正确做法**: Cell 内部只输出残差增量, 不包含 u_tgt。

#### 代码: models/coupled_mamba3_fork.py — CrossMamba3Cell

```python
class CrossMamba3Cell(nn.Module):
    def __init__(self, ..., use_gate: bool = False, v_self_ratio: float = 0.0):
        super().__init__()
        # ... 原有初始化 ...
        self.v_self_ratio = float(v_self_ratio)
        self.use_gate = use_gate

        # ★ 门控网络: 输出纯标量 gate ∈ (0,1)
        if use_gate:
            self.gate_proj = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, 1),  # 输出标量 gate (broadcast 到所有位置)
            )
            # 初始化 gate_proj 偏向 1.0 (初始时接近纯跨模态)
            nn.init.constant_(self.gate_proj[-1].weight, 0.0)
            nn.init.constant_(self.gate_proj[-1].bias, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, u_tgt, u_src0, u_src1, w_src, src_keys, cu_seqlens=None):
        """前向传播.

        Returns:
            out: (B, L, d_model)
            ★ 注意: 返回的是 **残差增量**, 不包含 u_tgt.
                外层 CoupledMamba3Fork 统一做 y + u_tgt.
        """
        batch, seqlen, _ = u_tgt.shape
        s0_key, s1_key = src_keys

        # ---------------- 1) tgt 出 z + x_default + 控制信号 ----------------
        proj_t = self.in_proj_tgt(u_tgt)
        z, x_default, dd_dt, dd_A, trap, angles = torch.split(
            proj_t,
            [self.d_inner, self.d_inner,
             self.nheads, self.nheads, self.nheads,
             self.num_rope_angles],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        if self.v_self_ratio > 0.0:
            x_default = rearrange(x_default, "b l (h p) -> b l h p", p=self.headdim)

        # ---------------- 2) tgt 出 C (Q) ----------------
        C = self.c_proj_tgt(u_tgt)
        C = rearrange(C, "b l (r g n) -> b l r g n",
                      r=self.mimo_rank, g=self.num_bc_heads)

        # ---------------- 3) src 加权出 B (K) 和 V (x) ----------------
        w0 = w_src[..., 0:1]
        w1 = w_src[..., 1:2]
        B0 = self.b_projs[s0_key](u_src0)
        B1 = self.b_projs[s1_key](u_src1)
        B = w0 * B0 + w1 * B1
        B = rearrange(B, "b l (r g n) -> b l r g n",
                      r=self.mimo_rank, g=self.num_bc_heads)

        V0 = self.v_projs[s0_key](u_src0)
        V1 = self.v_projs[s1_key](u_src1)
        x = w0 * V0 + w1 * V1
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        if self.v_self_ratio > 0.0:
            x = self.v_self_ratio * x_default + (1.0 - self.v_self_ratio) * x

        # ---------------- 4-6) SSM kernel (同原代码) ----------------
        # (照抄原 CrossMamba3Cell.forward 的 4-6 步, 此处省略)
        # ...
        # y = self.out_proj(...)  → (B, L, d_model)

        # ---------------- 7) 显式门控 (★ 修复残差重复) ----------------
        if self.use_gate:
            gate_input = torch.cat([u_tgt, y], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, L, 1)
            # ★ 只输出 gate * y, 不加 u_tgt!
            # 外层 CoupledMamba3Fork 负责做 y + u_tgt
            out = gate * y
        else:
            out = y  # 无门控时直接输出 SSM 结果, 外层做 y + u_tgt

        return out
```

对应的 `CoupledMamba3Fork.forward` **无需修改**，因为外层已经正确做了残差:

```python
# coupled_mamba3_fork.py:440 — 这行代码已经正确!
outs[tgt] = self.layer_norms[tgt](y + u_tgt)
```

---

### 3.5 多任务训练启用 (sub_loss)

#### 改造分析

CAGMamba 的三路损失 (M+T+A) 强制单模态编码器独立预测情感，提供深监督。
Coupled-BI-Mamba3 已有 sub_loss 代码（ISM CLS + sub_fc_T/A/V），但仅对 SIMS 启用。

#### 代码: models/classifier.py — sub_loss 默认启用

```python
class MSAClassifier(nn.Module):
    def __init__(self, ...):
        # ... 原有代码 ...

        # ★ 修改: sub_loss 对所有回归任务启用 (不再局限于 multi_task)
        self.use_sub_loss = (
            ism_depth > 0
            and task_type == "regression"
            and float(getattr(args, "sub_loss_lambda", 0.0)) > 0.0
            # ★ 去掉 "and self.multi_task" 的限制
        )
        if self.use_sub_loss:
            self.sub_fc_T = nn.Linear(d_model, 1, **factory_kwargs)
            self.sub_fc_A = nn.Linear(d_model, 1, **factory_kwargs)
            self.sub_fc_V = nn.Linear(d_model, 1, **factory_kwargs)
        else:
            self.sub_fc_T = self.sub_fc_A = self.sub_fc_V = None
```

Trainer 中已经支持 `sub_loss_lambda` (trainer.py:105)，传入 CLI 参数即可启用:

```python
# trainer.py 中已存在的逻辑:
self.sub_loss_lambda = float(getattr(args, "sub_loss_lambda", 0.0))
```

---

### 3.6 精细参数分组 (4组)

#### 改造分析

CAGMamba 使用 5 组参数 (BERT decay/no-decay, Audio base/large, Mamba 各子组)，
对不同组件使用差异化学习率。本方案实现 4 组:

| 分组 | 包含参数 | LR | WD |
|------|---------|----|----|
| bert | text_encoder.* | bert_lr (1e-5) | 1e-5 |
| mamba_core | A_log, .D | main_lr × 0.5 | 1e-6 |
| mamba_dt | dt_proj, dt_bias | main_lr × 0.3 | 1e-6 |
| other | 其余参数 | main_lr (5e-4) | 1e-5 |

#### 代码: trainer.py

```python
def _build_optimizer(self):
    """★ 替换原有的 2 组参数分组为 4 组 (对齐 CAGMamba)."""
    bert_lr = float(getattr(self.args, "bert_learning_rate", 2e-5))
    main_lr = float(self.args.learning_rate)
    wd = float(self.args.weight_decay)

    bert_params = []       # BERT/RoBERTa 编码器
    mamba_core_params = [] # A_log, D (状态空间核心)
    mamba_dt_params = []   # dt_proj, dt_bias (离散化控制)
    other_params = []      # 其他所有参数

    for n, p in self.model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_encoder" in n:
            bert_params.append(p)
        elif any(k in n for k in ("A_log", ".D")):
            mamba_core_params.append(p)
        elif any(k in n for k in ("dt_proj", "dt_bias")):
            mamba_dt_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if bert_params:
        param_groups.append({
            "params": bert_params, "lr": bert_lr,
            "weight_decay": wd, "name": "bert",
        })
    if mamba_core_params:
        param_groups.append({
            "params": mamba_core_params,
            "lr": main_lr * 0.5,       # 状态参数保守更新
            "weight_decay": wd * 0.1,  # 极小正则化
            "name": "mamba_core",
        })
    if mamba_dt_params:
        param_groups.append({
            "params": mamba_dt_params,
            "lr": main_lr * 0.3,       # dt 对学习率敏感
            "weight_decay": wd * 0.1,
            "name": "mamba_dt",
        })
    param_groups.append({
        "params": other_params, "lr": main_lr,
        "weight_decay": wd, "name": "other",
    })

    self.optimizer = torch.optim.AdamW(param_groups)
```

在 `__init__` 中调用 `_build_optimizer()` 替换原有的优化器构建。

> **关于数据编码器 (Data2Vec) 参数分组**: 如果启用了在线音频编码器,
> 可以增加 `audio_encoder` 分组 (学习率 5e-6 或 2e-6),
> 与 BERT 组类似处理。

---

### 3.7 ISM 深度 + Dropout 超参调整

#### 改造分析

| 参数 | 改造前 | 改造后 | 原因 |
|------|--------|--------|------|
| ism_depth | 3 | 1 | 预训练编码器已编码序列上下文, ISM 部分冗余 |
| dropout | 0.1 | 0.25 | MOSI/MOSEI 样本少, 强正则化防过拟合 |
| main_lr (MOSI) | 1e-3 | 5e-4 | Mamba dt 对学习率敏感, 低 lr 更稳定 |
| sub_loss_lambda | 0 (关闭) | 0.3 | 深监督, 对齐 CAGMamba |
| loss | SmoothL1 | 保留 (CAGMamba 用 L1, 保留现有选择) | |

在 `config.json` 中调整默认值, 同时保留 CLI 参数覆盖:

```json
{
    "model": {
        "dropout": 0.25,
        "ism_depth": 1
    },
    "datasets": {
        "MOSI": {
            "learning_rate": 0.0005,
            "sub_loss_lambda": 0.3
        },
        "MOSEI": {
            "learning_rate": 0.0001,
            "sub_loss_lambda": 0.3
        },
        "SIMS": {
            "learning_rate": 0.0001,
            "sub_loss_lambda": 0.3
        }
    }
}
```

---

### 3.8 视频模态消融 (可选)

视频特征是 MOSI/MOSEI 潜在噪声源 (20/35 维手工 FACET 特征)。
提供 CLI 选项控制视频是否参与:

```python
# 在 MSAClassifier.__init__ 中添加:
self.video_dropout_rate = float(getattr(args, "video_dropout_rate", 0.0))
```

或者更激进: 在训练中完全移除视频模态:

```bash
# 通过修改 feature_dims 或设置 dummy 视频:
python train.py --dataset MOSI --video_dim 0 --v_self_ratio 0.3 ...
```

---

## 4. 配置文件更新

### config.json 修改

```json
{
    "common": {
        "use_bert": true,
        "bert_pretrained": "roberta-base",
        "bert_finetune": true,
        "bert_learning_rate": 1e-5,
        "need_data_aligned": false,
        "need_normalized": true,
        "early_stop": 8,
        "num_workers": 4,
        "pin_memory": true,
        "save_checkpoints": true,
        "checkpoints_dir": "checkpoints",
        "logs_dir": "logs",
        "results_dir": "results",
        "use_context": true
    },
    "model": {
        "name": "CoupledBIMamba3_Context",
        "d_model": 128,
        "num_layers": 2,
        "d_state": 64,
        "expand": 2,
        "headdim": 32,
        "ngroups": 1,
        "rope_fraction": 0.5,
        "is_mimo": false,
        "mimo_rank": 4,
        "chunk_size": 64,
        "is_outproj_norm": false,
        "dropout": 0.25,
        "v_self_ratio": 0.3,
        "pool_type": "mean",
        "ism_depth": 1,
        "ism_d_state": 32,
        "ism_mixer_type": "bimamba",
        "ism_bimamba3_headdim": 64,
        "ism_bimamba3_ngroups": 1,
        "ism_bimamba3_rope_fraction": 0.5,
        "ism_bimamba3_chunk_size": 64,
        "ism_bimamba3_is_mimo": false,
        "ism_bimamba3_fusion": "add_divide2",
        "ism_bimamba3_share_mimo": true
    },
    "datasets": {
        "MOSI": {
            "task_type": "regression",
            "num_classes": 1,
            "featurePath": "CMU-MOSI/Processed/unaligned_50.pkl",
            "feature_dims": [768, 5, 20],
            "seq_lens": [50, 50, 50],
            "train_samples": 1284,
            "batch_size": 32,
            "learning_rate": 0.0005,
            "weight_decay": 1e-5,
            "epochs": 120,
            "sub_loss_lambda": 0.3,
            "KeyEval": "Acc2"
        },
        "MOSEI": {
            "task_type": "regression",
            "num_classes": 1,
            "featurePath": "CMU-MOSEI/Processed/unaligned_50.pkl",
            "feature_dims": [768, 74, 35],
            "seq_lens": [50, 500, 375],
            "train_samples": 16326,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 1e-4,
            "epochs": 30,
            "sub_loss_lambda": 0.3,
            "KeyEval": "Loss"
        },
        "SIMS": {
            "task_type": "regression",
            "num_classes": 1,
            "multi_task": true,
            "task_weights": {"M": 1.0, "T": 0.4, "A": 0.4, "V": 0.4},
            "featurePath": "SIMS/Processed/unaligned_39.pkl",
            "feature_dims": [768, 33, 709],
            "seq_lens": [39, 400, 55],
            "train_samples": 1368,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 1e-4,
            "epochs": 50,
            "sub_loss_lambda": 0.3,
            "KeyEval": "Loss"
        },
        "IEMOCAP": {
            "task_type": "classification",
            "num_classes": 4,
            "class_names": ["happy", "sad", "neutral", "angry"],
            "featurePath": "IEMOCAP/Processed/iemocap_unaligned.pkl",
            "feature_dims": [768, 100, 342],
            "seq_lens": [50, 1500, 500],
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 1e-4,
            "epochs": 50,
            "sub_loss_lambda": 0.3,
            "KeyEval": "F1"
        },
        "MELD": {
            "task_type": "classification",
            "num_classes": 7,
            "class_names": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
            "featurePath": "MELD/Processed/meld_unaligned.pkl",
            "feature_dims": [768, 300, 342],
            "seq_lens": [50, 1500, 500],
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 1e-4,
            "epochs": 30,
            "sub_loss_lambda": 0.3,
            "KeyEval": "F1"
        }
    }
}
```

---

## 5. 环境配置与依赖安装

### 5.1 新增依赖

```bash
# RoBERTa / Data2Vec (通过 transformers 已有)
# 无需额外安装, 只需要下载模型权重

# 下载 RoBERTa-base (首次运行会自动下载, 或手动):
huggingface-cli download roberta-base --local-dir ./pretrained/roberta-base

# 下载 Data2Vec-Audio-base:
huggingface-cli download facebook/data2vec-audio-base-960h --local-dir ./pretrained/data2vec-audio-base
```

### 5.2 修改后的 `_path_setup.py`

如果用镜像加速 HuggingFace 下载:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 5.3 run.py 新增 CLI 参数

```python
parser.add_argument("--v_self_ratio", type=float, default=0.3,
                    help="V 通道自信息保留比例 [0,1]")
parser.add_argument("--use_context", action="store_true", default=False,
                    help="启用对话上下文")
parser.add_argument("--sub_loss_lambda", type=float, default=0.0,
                    help="模态级辅助损失权重")
parser.add_argument("--use_gate", action="store_true", default=False,
                    help="启用显式门控融合")
parser.add_argument("--use_audio_encoder", action="store_true", default=False,
                    help="启用 Data2Vec 音频在线编码")
parser.add_argument("--audio_pretrained", type=str,
                    default="facebook/data2vec-audio-base-960h",
                    help="预训练音频模型名称")
parser.add_argument("--audio_finetune", action="store_true", default=True,
                    help="fine-tune 音频编码器")
```

---

## 6. 验证路线

### 6.1 渐进式验证 (5 步)

每步做一次 MOSI 实验, 逐步叠加, 记录每一步的指标变化:

| Step | 改动 | 命令 | 预期收益 |
|------|------|------|---------|
| 0 | **Baseline** (原代码) | `python train.py --dataset MOSI --seed 42` | 对照 |
| 1 | v_self_ratio + dropout + sub_loss + ism_depth | `python train.py --dataset MOSI --seed 42 --v_self_ratio 0.3 --dropout 0.25 --sub_loss_lambda 0.3 --ism_depth 1` | ★★ |
| 2 | + RoBERTa-base 文本 (改 config) | `python train.py --dataset MOSI --seed 42 --bert_pretrained roberta-base --v_self_ratio 0.3 --dropout 0.25 --sub_loss_lambda 0.3 --ism_depth 1` | ★★★★ |
| 3 | + 对话上下文 | `python train.py --dataset MOSI --seed 42 --use_context --bert_pretrained roberta-base --v_self_ratio 0.3 --dropout 0.25 --sub_loss_lambda 0.3 --ism_depth 1` | ★★★★★ |
| 4 | + 门控融合 | `python train.py --dataset MOSI --seed 42 --use_context --use_gate --bert_pretrained roberta-base --v_self_ratio 0.3 --dropout 0.25 --sub_loss_lambda 0.3 --ism_depth 1` | ★★★ |
| 5 | + 4 组参数 + Data2Vec (可选) | 改 trainer.py 和配置文件后运行 | ★★ |

### 6.2 预期收益

| Step | 改进 | MAE ↓ | Acc2 ↑ | Acc7 ↑ |
|------|------|-------|--------|--------|
| 1 | 参数调优 | 0.03~0.05 | 1~3% | 1~2% |
| 2 | + RoBERTa | 0.05~0.10 | 2~5% | 2~4% |
| 3 | + 上下文 | 0.05~0.15 | 3~8% | 3~6% |
| 4 | + 门控 | 0.01~0.03 | 1~2% | 1~2% |
| **总计** | | **0.14~0.33** | **7~18%** | **7~14%** |

### 6.3 全量训练命令

```bash
# MOSI — 全量最推荐配置
python train.py --dataset MOSI --seed 42 \
    --use_context \
    --use_gate \
    --bert_pretrained roberta-base \
    --v_self_ratio 0.3 \
    --dropout 0.25 \
    --sub_loss_lambda 0.3 \
    --ism_depth 1 \
    --lr 5e-4 \
    --bert_lr 1e-5 \
    --weight_decay 1e-5 \
    --epochs 120 \
    --exp_tag full_upgrade_mosi

# MOSEI
python train.py --dataset MOSEI --seed 42 \
    --use_context \
    --use_gate \
    --bert_pretrained roberta-base \
    --v_self_ratio 0.3 \
    --dropout 0.2 \
    --sub_loss_lambda 0.3 \
    --ism_depth 1 \
    --lr 1e-4 \
    --bert_lr 5e-6 \
    --weight_decay 1e-4 \
    --epochs 30 \
    --exp_tag full_upgrade_mosei
```

---

## 附录: 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `models/classifier.py` | 修改 | 新增 TextPretrainedEncoder; _encode 增加 with_fusion 参数; forward 增加上下文路径 _forward_with_context; sub_loss 默认启用; 支持 use_gate / v_self_ratio |
| `models/coupled_mamba3_fork.py` | 修改 | CrossMamba3Cell 增加 use_gate 选项; 门控输出纯残差增量 (修复重复相加 bug) |
| `dataset/data_loader.py` | 修改 | MMDataset 增加 _load_context / _infer_video_id / __getitem__ 上下文字段 |
| `trainer.py` | 修改 | 4 组参数分组替换原有的 2 组; 新增 _build_optimizer 方法 |
| `configs/config.json` | 修改 | 更新 RoBERTa/dropout/v_self_ratio/ism_depth/sub_loss_lambda/lr 默认值 |
| `run.py` | 修改 | 新增 CLI 参数 (v_self_ratio / use_context / sub_loss_lambda / use_gate 等) |

---

> **文档版本**: v2.0
> **关联文档**: [性能差距分析.md](./性能差距分析.md) (诊断) → 本文档 (改造方案)
> **设计原则**: 保留 Coupled-BI-Mamba3 的 Mamba-3 SSM + QKV 注入核心创新,
> 吸收 CAGMamba 的上下文/门控/多任务/精细超参优点, 两者互补而非替代。
