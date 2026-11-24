# CAD-BEBC：连续自回归扩散块外双向因果架构（统一系统规范）

> 状态：已实现于统一入口 `src/cad_bebc.mora`，与 `src/latent_codec.mora`、`src/diffusion/pipeline.mora`（完全可微误差门控权重涌现）及冻结 Mini-LLaMA 外部专家组成单一整合系统。
> 架构标识：`cad-bebc-unified-v2`

---

## 1. 一体式架构定位（一套系统、多条通路）

**CAD-BEBC**（Continuous Autoregressive Diffusion with Block-External Bidirectional Causality）是一套在**连续潜空间**上运行的块扩散系统，不再拆分为相互独立的离散语言模型与扩散模型：

- **统一潜空间编解码**（`src/latent_codec.mora`）：使用分词器与 Mini-LLaMA 词嵌入 / 输出头，将离散文本前缀与待生成后缀映射至连续空间 $z \in \mathbb{R}^{B \times (K B_s) \times D}$（默认演示 $K=4, B_s=4, D=32$），去噪后再解码回离散 token。维度不匹配时文本解码路径会自动跳过。
- **单一统一入口**（`src/cad_bebc.mora`）：输入统一经过潜空间编码与无路由器竞争门控，经由三条通路完成推理与演进：
  1. **外部冻结专家快速通道**：直接调用预训练冻结的 Mini-LLaMA（`mode="llama"`）或 HTTP 服务，当 $\pi_{\text{int}} < 0.05$ 时走纯外部路径（0 扩散步）；
  2. **内化短通路**：成熟度直接写入权重 $(U, V, w_g, b_g)$ 自身，门控突破 `theta_skip` 时真正跳过中间主干步（NFE $8 \to 2$）；
  3. **连续块扩散主干**：多步连续流匹配 ODE 去噪（深度思考），并驱动内化权重缓慢生长或高误差回退。

---

## 2. 三层注意力与因果结构

注意力结构偏置由可微矩阵 $W_{\text{blk}} \in [-\text{init\_iso}, 0]^{K \times K}$ 与前缀因果掩码构成：

$$B_{\text{struct}} = P_{\text{blk}}\, (W_{\text{blk}} \odot M_{\text{off}})\, P_{\text{blk}}^\top + M_{\text{prefix}}$$

1. **块内双向**（$b_i = b_j$）：偏置恒为 0，当前块内部 token 充分双向交互。
2. **前缀因果**（前缀块内部严格因果顺序）：由 $M_{\text{prefix}}$ 施加强屏蔽，保持自回归前缀顺序。
3. **块外双向因果**（其余 $b_i \neq b_j$）：由 $W_{\text{blk}}[b_i, b_j]$ 控制，通过误差门控反向传播缓慢生长或回退，允许前块影响后块，也允许后块反向影响前序块。

---

## 3. 误差门控的纯权重涌现机制（完全可微）

不维护显式巩固度 $c$ 或误差滑动均值，成熟度完全写入权重 $(U, V, w_g, b_g, W_{\text{blk}}, w_{\text{comp}}, b_{\text{comp}})$ 自身。所有参数通过统一的 `value_and_grad(unified_diff_loss)` 获得梯度，再用软门缩放步长。

### 3.1 驱动信号与静默死区门

- 影子对齐误差 $\tilde{\mathcal{E}}_k$（短通路对齐）
- 一致性误差 $\mathcal{E}_{\text{base}}$

$$m_{\text{low}} = \sigma\!\left(\frac{\epsilon_{\text{low}} - \mathcal{E}_{\text{base}}}{T_e}\right), \quad m_{\text{fail}} = \sigma\!\left(\frac{\mathcal{E}_{\text{base}} - \epsilon_{\text{high}}}{T_e}\right)$$

当前主更新路径以 $m_{\text{low}}$ 作为正向步长缩放（`step_ok`），以 $m_{\text{fail}}$ 作为回退缩放。

### 3.2 统一可微损失与不对称更新（慢生长 / 快回退）

`unified_diff_loss` 同时包含：
- 混合重构损失（内部输出与外部专家的连续混合）
- 短通路对齐损失
- 可微门控监督（对齐较好时积极推高门控）

在 `emerge_step` 中应用：

- **跨步低秩映射**（初始 $V=0$，冷启动恒等）：
  $$U, V \leftarrow (1 - \lambda_{\text{decay}} m_{\text{fail}}) \cdot \{U,V\} - \eta_{\text{lr}} m_{\text{low}} \nabla \mathcal{L}$$
- **跳过门神经元**（$g = \sigma(w_g^\top \bar{z} + b_g)$，静息 $b_{\text{rest}}=-5.5$）：
  $$w_g, b_g \leftarrow \text{可微监督梯度} \times m_{\text{low}} + \text{回退到 } b_{\text{rest}} \times m_{\text{fail}}$$
- **块外因果 $W_{\text{blk}}$** 与 **竞争门**：通过同一反向传播梯度，乘以 $m_{\text{low}} / m_{\text{fail}}$ 后更新，并约束在 $[-\text{init\_iso}, 0]$。

---

## 4. 外部冻结专家与竞争

- 默认 `mode="llama"`：加载冻结 Mini-LLaMA（`models/mini_llama_frozen.safetensors`）；缺大 checkpoint 时自动回退到随机初始化。
- 推理时：竞争门 $\pi_{\text{int}}$ 连续混合内部输出与外部专家；仅当 $\pi_{\text{int}} < 0.05$ 时完全跳过内部计算。
- 无显式路由器，竞争完全由权重与误差信号驱动。短通路跳步决策由门控 $g_k$ 独立控制，与 $\pi_{\text{int}}$ 无直接硬耦合。

---

## 5. 定位与边界

- 单一 **CAD-BEBC** 统一系统。
- `config/diffusion.json`（微型）用于 CPU 秒级验证完整生命周期：冷启动 → 误差门控权重涌现 → 真跳步 → OOD 回退。
- 工业级目标配置见 `config/diffusion.industrial.json`（d=256, 8 层, 16 块）。
- 当前为原型验证阶段，大规模能力评估待进一步实现。文本解码路径在维度不匹配时会自动跳过。
