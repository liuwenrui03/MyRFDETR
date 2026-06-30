# RF-DETR 迁移协议（Agent Handoff Protocol）

> 目的：帮助后续 Agent 快速理解本轮改造的**意图、边界、已验证保障、潜在遗漏**，并在最短路径上完成复验与收尾。

---

## 1. 迁移目标与范围

本轮改造目标：

1. 支持 DINO 家族多版本、多骨干：
   - DINOv2：`tiny / small / base`（含 registers + windowed 变体）
   - DINOv3 ViT：`tiny / small`
   - DINOv3 ConvNext：`tiny / small`
2. 保留“检测 + 分割”可用配置，并删除推理代价过高配置。
3. 保持预训练加载链路不被破坏，尤其是 `rf-detr-seg` 预训练资产映射。
4. 清理部分冗余注释/演示代码。

---

## 2. 关键改动清单（按模块）

### 2.1 Backbone 扩展

- 新增：`src/rfdetr/models/backbone/dinov3_vit.py`
  - 新增 DINOv3 ViT 封装，支持 tiny/small。
  - 预训练模型名已写死到实现中（HF 名称）。

- 新增：`src/rfdetr/models/backbone/dinov3_convnext.py`
  - 新增 DINOv3 ConvNext 封装，支持 tiny/small。
  - 输出 stage 通道根据 config 计算。

- 修改：`src/rfdetr/models/backbone/backbone.py`
  - `Backbone` 增加三路分发：`dinov2` / `dinov3_vit` / `dinov3_convnext`。
  - DINOv2 保留原解析逻辑（registers/windowed）。
  - `get_named_param_lr_pairs` 对非 DINOv2 路径使用更保守的统一策略，避免错误套用 DINOv2 layer decay。

### 2.2 DINOv2 tiny 补齐

- 修改：`src/rfdetr/models/backbone/dinov2.py`
  - `size_to_config`、`size_to_config_with_registers` 新增 tiny 映射。
  - 清理了部分冗余注释与 `__main__` 演示块。

- 新增：
  - `src/rfdetr/models/backbone/dinov2_configs/dinov2_tiny.json`
  - `src/rfdetr/models/backbone/dinov2_configs/dinov2_with_registers_tiny.json`

### 2.3 配置系统扩展

- 修改：`src/rfdetr/config.py`
  - `EncoderName` 扩展到：
    - `dinov2_windowed_tiny/small/base`
    - `dinov2_registers_windowed_tiny/small`
    - `dinov3_vit_tiny/small`
    - `dinov3_convnext_tiny/small`
  - `RFDETRNanoConfig` / `RFDETRSegNanoConfig` 切换到 `dinov2_windowed_tiny`。
  - 新增配置类：
    - `RFDETRDinoV3ViTSmallConfig`
    - `RFDETRDinoV3ConvNextTinyConfig`
    - `RFDETRSegDinoV3ConvNextTinyConfig`

### 2.4 变体导出

- 修改：`src/rfdetr/variants.py`
  - 新增变体类：
    - `RFDETRDinoV3ViTSmall`
    - `RFDETRDinoV3ConvNextTiny`
    - `RFDETRSegDinoV3ConvNextTiny`

- 修改：`src/rfdetr/__init__.py`
  - 对外导出新增变体。

### 2.5 输入通道适配兼容

- 修改：`src/rfdetr/inference.py`
  - 抽象 `_resolve_patch_projection()`，不再写死 `encoder.encoder.embeddings...` 路径。
  - 对 ViT 仍支持 `num_channels != 3` 的 patch projection 改写。
  - 对无 patch projection 的 backbone（ConvNext）主动报错，避免静默错误。

### 2.6 配置文件策略调整

- 新增：
  - `configs/rfdetr_dinov3_convnext_tiny.yaml`
  - `configs/rfdetr_seg_dinov3_convnext_tiny.yaml`

- 删除（高推理代价）：
  - `configs/rfdetr_base.yaml`
  - `configs/rfdetr_large.yaml`
  - `configs/rfdetr_seg_large.yaml`
  - `configs/rfdetr_seg_xlarge.yaml`
  - `configs/rfdetr_seg_2xlarge.yaml`

- 修改：`tests/cli/test_configs.py`
  - 同步检测/分割配置集合与 class_path 断言。

---

## 3. 本轮“已核查保障”（Done Guarantees）

已完成的保障类型：

1. **语法级保障**
   - 关键变更文件通过 `py_compile` / `ast.parse`。

2. **结构级保障**
   - 关键新类存在性（config/variant）已通过 AST 扫描确认。
   - 新增配置文件存在，测试侧清单已同步。

3. **预训练路径保障（静态）**
   - DINOv3 ViT/ConvNext 的 HF 模型名已写入实现并核对。
   - `rfdetr-seg` 的预训练资产映射（`nano/small/medium`）仍在 `model_weights` 中保留。

---

## 4. 可能遗漏与风险清单（Must Recheck）

> 这些是后续 Agent 需要优先复验的点。

1. **运行时加载未做端到端验证**
   - 当前环境缺少 `torch`，未实际跑 `from_pretrained` 下载 + forward。
   - 风险：HF 名称在不同 transformers 版本上可能存在别名差异。

2. **ConvNext 输出层索引与 projector 假设**
   - 当前使用 `out_feature_indexes=[2,3,4]` 映射 stage2/3/4。
   - 风险：某些 transformers 版本 `hidden_sizes` 与 stage 对齐方式变化。

3. **非 DINOv2 的 LR decay 策略**
   - 当前为保守策略（统一 lr/wd 处理）。
   - 风险：训练收敛速度和最佳超参可能需单独调优。

4. **num_channels != 3 与 ConvNext**
   - 当前显式报错而非自动适配。
   - 风险：如果业务需要多通道 ConvNext 输入，需补充专用适配逻辑。

5. **删除配置的连锁引用**
   - 已同步 `tests/cli/test_configs.py`。
   - 仍需确认 docs/脚本/CI 中无硬编码引用被删配置名。

---

## 5. 后续 Agent 接手执行顺序（推荐）

### Step A: 环境就绪

确保可运行环境具备：
- `torch`
- `transformers`（建议与项目约束一致）

### Step B: 最小加载验证（必须）

建议至少覆盖以下 smoke case：

1. DINOv2 tiny 检测配置构建并前向
2. DINOv3 ViT small 检测配置构建并前向
3. DINOv3 ConvNext tiny 检测配置构建并前向
4. DINOv3 ConvNext tiny 分割配置构建并前向
5. `rf-detr-seg-{nano,small,medium}.pt` 逐个执行预训练加载（至少到 `load_state_dict` 成功）

### Step C: 回归测试

优先运行：
- `tests/cli/test_configs.py`
- 与 `backbone`、`weights`、`inference` 相关测试子集

再视时间跑更广泛测试与 lint。

---

## 6. 失败回滚策略

若 DINOv3 路径在目标环境不稳定，可按最小破坏回滚：

1. 回滚 `backbone/backbone.py` 的 dino3 分发分支
2. 保留 DINOv2 tiny 支持（低风险、高收益）
3. 暂时下线 dino3 对应 config 与 variant 导出
4. 保留配置删减策略（只要测试通过）

---

## 7. 验收标准（Definition of Done）

满足以下条件才算迁移完成：

1. DINOv2 tiny、DINOv3 ViT、DINOv3 ConvNext 均可构建并完成一次前向。
2. `rfdetr-seg` 预训练（至少 nano/small/medium）可成功加载。
3. `tests/cli/test_configs.py` 通过。
4. 无新增语法/导入错误；关键 lint 通过。
5. 删除的高代价配置无残留引用导致失败。

---

## 8. 给下一个 Agent 的一句话总结

这次改造已经把**架构入口和配置类型系统**扩展到 dino 全家桶（v2 tiny + v3 vit/convnext），并完成了配置精简；你下一步的核心工作不是再改代码，而是做**真实运行环境下的预训练加载与前向闭环验证**，并根据实测结果微调默认超参/命名兼容。
