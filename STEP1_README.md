# MULTIFLOW Step 1: 自适应稀疏度分配

## 📋 这一步做什么？

**目标**：自动发现 LLaVA 模型中不同模态（视觉/文本/融合）的冗余度差异，为每层分配合适的剪枝比例。

**核心思想**：
- 全局目标：整体剪掉 50% 参数
- 局部自适应：视觉层可能只剪 40%，融合层剪 55%，文本层剪 45%
- 结果：总体达到 50%，但尊重各模态特性

---

## 🚀 如何运行

### 方法 1：使用脚本（推荐）

```bash
chmod +x run_step1.sh
./run_step1.sh
```

### 方法 2：直接运行 Python

```bash
# 测试 50% 稀疏度
python step1_compute_sparsity_distribution.py --target_sparsity 0.5

# 测试 70% 稀疏度（更激进）
python step1_compute_sparsity_distribution.py --target_sparsity 0.7

# 测试 30% 稀疏度（保守）
python step1_compute_sparsity_distribution.py --target_sparsity 0.3
```

### 参数说明

```bash
--model_path         # LLaVA 模型路径（默认：/gpfs/volcano/models/llava-v1.5-7b）
--target_sparsity    # 目标全局稀疏度 0-1（默认：0.5，即 50%）
--device            # 运行设备（默认：cuda）
```

---

## 📊 预期输出示例

### 1. 模态分组统计

```
Modality-wise parameter distribution:
--------------------------------------------------------------------------------
VISION  :  96 layers,   88,080,384 params ( 2.09%)
FUSION  : 194 layers, 4,123,254,784 params (97.91%)
TEXT    :   0 layers,           0 params ( 0.00%)
TOTAL   : 290 layers, 4,211,335,168 params (100.00%)
--------------------------------------------------------------------------------
```

**解读**：
- LLaVA 的大部分参数在 FUSION 层（LLM decoder + MLP connector）
- VISION 层相对较少（CLIP ViT 编码器）
- TEXT 层为 0（因为 embed_tokens 和 lm_head 通常不剪枝）

---

### 2. 各模态剪枝结果

#### Vision 模态（示例）

```
================================================================================
Processing VISION modality
================================================================================
Total parameters in vision: 88,080,384
Target sparsity: 50.0%
Pruning threshold: 0.012345
Will prune 44,040,192 / 88,080,384 parameters

Per-layer sparsity distribution in VISION:
--------------------------------------------------------------------------------
Layer Name                                                              Sparsity
--------------------------------------------------------------------------------
model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj      48.23%
model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj      49.87%
model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj      51.12%
model.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj    50.45%
model.vision_tower.vision_model.encoder.layers.0.mlp.fc1               49.67%
...
--------------------------------------------------------------------------------
Modality summary:
  Layers: 96
  Total params: 88,080,384
  Pruned params: 44,040,192
  Actual sparsity: 50.00%
  Per-layer sparsity range: [48.00%, 52.00%]
  Per-layer sparsity mean ± std: 50.00% ± 0.85%
```

#### Fusion 模态（示例）

```
================================================================================
Processing FUSION modality
================================================================================
Total parameters in fusion: 4,123,254,784
Target sparsity: 50.0%
Pruning threshold: 0.008976
Will prune 2,061,627,392 / 4,123,254,784 parameters

Per-layer sparsity distribution in FUSION:
--------------------------------------------------------------------------------
Layer Name                                                              Sparsity
--------------------------------------------------------------------------------
model.mm_projector.0                                                    52.34%
model.mm_projector.2                                                    48.91%
model.model.layers.0.self_attn.q_proj                                   49.23%
model.model.layers.0.self_attn.k_proj                                   50.78%
model.model.layers.0.self_attn.v_proj                                   51.45%
...
--------------------------------------------------------------------------------
Modality summary:
  Layers: 194
  Total params: 4,123,254,784
  Pruned params: 2,061,627,392
  Actual sparsity: 50.00%
  Per-layer sparsity range: [46.50%, 53.50%]
  Per-layer sparsity mean ± std: 50.00% ± 1.23%
```

---

### 3. 全局总结

```
================================================================================
GLOBAL SUMMARY
================================================================================

Target global sparsity: 50.00%
Actual global sparsity: 50.00%

Modality-wise breakdown:
--------------------------------------------------------------------------------
Modality        Params       Pruned   Sparsity   % of Total
--------------------------------------------------------------------------------
FUSION    4,123,254,784 2,061,627,392     50.00%       97.91%
VISION       88,080,384    44,040,192     50.00%        2.09%
--------------------------------------------------------------------------------
TOTAL     4,211,335,168 2,105,667,584     50.00%      100.00%
================================================================================
```

---

### 4. 关键洞察（🔍 KEY INSIGHT）

```
========================= 🔍 KEY INSIGHT =========================

MULTIFLOW's adaptive sparsity distribution:
  • FUSION : 50.00% (+0.00% vs target) → Equal redundancy
  • VISION : 50.00% (+0.00% vs target) → Equal redundancy

This shows which modalities have more/less redundancy!
In Step 2, we'll use activation statistics for more accurate scoring.
================================================================================
```

**注意**：
- 在 Step 1 中，我们只用权重幅度 `|W|` 作为得分
- 各模态可能显示相似的稀疏度（因为只是简单的幅度排序）
- **真正的差异会在 Step 2 中显现**（使用激活统计后）

---

## 🎯 输出文件

运行后会生成 JSON 文件：`sparsity_distribution_0.5.json`

```json
{
  "target_sparsity": 0.5,
  "modality_stats": {
    "vision": {
      "num_layers": 96,
      "total_params": 88080384,
      "pruned_params": 44040192,
      "actual_sparsity": 0.5,
      "min_layer_sparsity": 0.48,
      "max_layer_sparsity": 0.52,
      "mean_layer_sparsity": 0.50,
      "std_layer_sparsity": 0.0085
    },
    "fusion": {
      "num_layers": 194,
      "total_params": 4123254784,
      "pruned_params": 2061627392,
      "actual_sparsity": 0.5,
      ...
    }
  },
  "layer_distribution": {
    "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj": 0.4823,
    "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj": 0.4987,
    ...
  }
}
```

这个文件将在 Step 2 和 Step 3 中使用！

---

## 🔍 如何解读结果

### 场景 1：模态稀疏度差异大

```
VISION : 40.00% (-10.00% vs target) → LESS redundant  ← 视觉层重要！
FUSION : 55.00% (+5.00% vs target)  → MORE redundant
```

**含义**：
- 视觉层冗余少，应该少剪一点（40%）
- 融合层冗余多，可以多剪一点（55%）
- 这是**好现象**，说明模型不同部分确实有不同的冗余度

### 场景 2：模态稀疏度接近

```
VISION : 50.00% (+0.00% vs target) → Equal redundancy
FUSION : 50.00% (+0.00% vs target) → Equal redundancy
```

**含义**：
- 在 Step 1（只用权重幅度）下，可能看到这种情况
- 不用担心！Step 2 会用激活统计，差异会更明显
- 或者说明模型各部分确实均衡

---

## ⚠️ 常见问题

### Q1: 为什么 TEXT 模态参数为 0？

A: 因为 `embed_tokens` 和 `lm_head` 通常不应该被剪枝（它们是词汇表映射）。如果你想剪枝这些层，可以修改 `detect_modality_llava` 函数。

### Q2: 为什么各模态稀疏度都是 50.00%？

A: 在 Step 1 中，我们对每个模态**独立**应用 50% 稀疏度。这是为了探索各模态的"自然稀疏度分布"。实际的差异会在层间分布的方差中体现（看 `std_layer_sparsity`）。

### Q3: 层间稀疏度的 std 是什么意思？

A:
- **std 小**（如 0.5%）：该模态内各层的冗余度很均匀
- **std 大**（如 5%）：该模态内有的层很冗余，有的层不冗余

---

## 📌 下一步

完成 Step 1 后，你会得到：

✅ **每层的目标稀疏度** (`layer_distribution`)
✅ **各模态的统计信息** (`modality_stats`)

**下一步（Step 2）**：
- 使用真实数据的激活统计
- 计算更准确的重要性得分（不只是权重幅度）
- 用 Step 1 的分配比例进行最终剪枝

---

## 💡 代码说明

### 核心函数

1. **`detect_modality_llava(param_name)`**
   - 根据参数名判断属于哪个模态
   - 你可以修改这个函数来调整模态分组

2. **`compute_multimodal_distribution(model, target_sparsity)`**
   - 主函数：计算自适应稀疏度分配
   - 返回：每层的目标稀疏度 + 统计信息

### 输出控制

- **简洁输出**：只显示前 5 层和后 5 层
- **详细输出**：修改代码中的 `if` 条件可以显示所有层
- **保存结果**：自动保存到 JSON 文件

---

## 🎓 理论背景

这一步实现了 MULTIFLOW 论文的 **Equation 7**：

> 通过对各模态分别应用目标稀疏度，自动发现层级稀疏度分布 $\{s_l\}_{l=1}^L$

在实际应用中：
- **Wanda**: 所有层统一 50% 稀疏度
- **MULTIFLOW**: 视觉层 40%，融合层 55%，文本层 45%（自动适应）

**优势**：保护重要模态，更激进地剪枝冗余模态。
