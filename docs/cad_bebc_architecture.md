# CAD-BEBC：连续自回归扩散块外双向因果架构（统一系统规范）

> 状态：已实现于统一入口 `src/cad_bebc.mora`，与 `src/latent_codec.mora`、`src/diffusion/pipeline.mora`（误差门控死区 + 权重涌现）及冻结 Mini-LLaMA 外部专家组成单一整合系统。
> 架构标识：`cad-bebc-unified-v2`

---

## 1. 一体式架构定位（一套系统、多条通路）

**CAD-BEBC**（Continuous Autoregressive Diffusion with Block-External Bidirectional Causality）是一套在**连续潜空间**上运行的块扩散系统，不再拆分为相互独立的离散语言模型与扩散模型：

- **统一潜空间编解码**（`src/latent_codec.mora`）：使用分词器与 Mini-LLaMA 词嵌入 / 输出头，将离散文本前缀与待生成后缀映射至连续空间 $z \in \mathbb{R}^{B \times (K B_s) \times D}$（默认 $K=4, B_s=4, D=32$），去噪后再解码回离散 token。
- **单一统一入口**（`src/cad_bebc.mora`）：输入统一经过潜空间编码与无路由器竞争门控，经由三条通路完成推理与演进：
  1. **外部冻结专家快速通道**：直接调用预训练冻结的 Mini-LLaMA（`mode="llama"`）或 HTTP 服务，0 扩散步快速响应；
  2. **内化短通路**：成熟度直接写入权重 $(U, V, w_g, b_g)$ 自身，门控突破阈值时真正跳过中间主干步（NFE $8 \to 2$）；
  3. **连续块扩散主干**：多步连续流匹配 ODE 去噪（深度思考），并驱动内化权重缓慢生长或高误差回退。

---

## 2. 三层注意力与因果结构

注意力结构偏置由可微矩阵 $W_{\text{blk}} \in [-\text{init\_iso}, 0]^{K \times K}$ 与前缀因果掩码构成：

$$B_{\text{struct}} = P_{\text{blk}}\, (W_{\text{blk}} \odot M_{\text{off}})\, P_{\text{blk}}^\top + M_{\text{prefix}}$$

1. **块内双向**（$b_i = b_j$）：偏置恒为 0，当前块内部 token 充分双向交互。
2. **前缀因果**（前缀块内部严格因果顺序）：由 $M_{\text{prefix}}$ 施加强屏蔽，保持自回归前缀顺序。
3. **块外双向因果**（其余 $b_i \neq b_j$）：由 $W_{\text{blk}}[b_i, b_j]$ 控制，通过误差门控反向传播缓慢生长或回退，允许前块影响后块，也允许后块反向影响前序块。

---

## 3. 误差门控的纯权重涌现机制

不维护显式巩固度 $c$ 或误差滑动均值，成熟度完全写入权重 $(U, V, w_g, b_g, W_{\text{blk}}, w_{\text{comp}}, b_{\text{comp}})$ 自身。

### 3.1 驱动信号与静默死区门

- 影子对齐误差 $\tilde{\mathcal{E}}_k$
- 一致性误差 $\mathcal{E}_{\text{base}}$

$$m_{\text{low}} = \sigma\!\left(\frac{\epsilon_{\text{low}} - \mathcal{E}_{\text{base}}}{T_e}\right), \quad m_{\text{ok},k} = m_{\text{low}} \cdot \sigma\!\left(\frac{\epsilon_{\text{sc}} - \tilde{\mathcal{E}}_k}{T_s}\right), \quad m_{\text{fail}} = \sigma\!\left(\frac{\mathcal{E}_{\text{base}} - \epsilon_{\text{high}}}{T_e}\right)$$

死区使生长与回退在中间区间近似静默。

### 3.2 不对称权重更新（慢生长 / 快回退）

在 `emerge_step` 中实现：

- **跨步低秩映射**（初始 $V=0$，冷启动恒等）：
  $$U, V \leftarrow (1 - \lambda_{\text{decay}} m_{\text{fail}}) \cdot \{U,V\} - \eta_{\text{lr}} m_{\text{ok}} \nabla \tilde{\mathcal{E}}$$
- **跳过门神经元**（$g = \sigma(w_g^\top \bar{z} + b_g)$，静息 $b_{\text{rest}}=-4.5$）：
  $$w_g, b_g \leftarrow \text{误差门控 BCE + 回退到 } b_{\text{rest}}$$
- **块外因果 $W_{\text{blk}}$** 与 **竞争门**：通过反向传播梯度，乘以 $m_{\text{low}} / m_{\text{fail}}$ 后更新，并约束在 $[-\text{init\_iso}, 0]$。

---

## 4. 外部冻结专家与竞争

- 默认 `mode="llama"`：加载冻结 Mini-LLaMA（`models/mini_llama_frozen.safetensors`）。
- 推理时：竞争门 $\pi_{\text{int}}$ 决定走外部专家（$\pi_{\text{int}} < 0.2$）、混合或内化路径。
- 无显式路由器，竞争完全由权重与误差信号驱动。

---

## 5. 定位与边界

- 单一 **CAD-BEBC** 统一系统。
- `config/diffusion.json`（微型）用于 CPU 秒级验证完整生命周期：冷启动 → 误差门控权重涌现 → 真跳步 → OOD 回退。
- 扩展配置见 `config/diffusion_scale.json`。
- 当前为原型验证阶段，大规模能力评估待进一步实现。
