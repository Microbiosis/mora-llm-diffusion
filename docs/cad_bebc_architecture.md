# CAD-BEBC：连续自回归扩散块外双向因果架构（统一系统规范）

> 状态：已实现于统一入口 `src/cad_bebc.mora`，与 `src/latent_codec.mora`、`src/diffusion/pipeline.mora`（完全可微误差门控权重涌现）、`src/diffusion/schedule.mora`（VP 调度与 DDIM 采样器）及冻结 Mini-LLaMA 外部专家组成单一整合系统。
> 架构标识：`cad-bebc-unified-v2`
>
> 训练入口：`src/train_cad_bebc.mora`（真实语料）。本文件描述的离散化与门控语义与
> `docs/mechanism_fixes_2026-11-24.md` 中的实测记录一致。

---

## 1. 一体式架构定位（一套系统、多条通路）

**CAD-BEBC**（Continuous Autoregressive Diffusion with Block-External Bidirectional Causality）是一套在**连续潜空间**上运行的块扩散系统，不再拆分为相互独立的离散语言模型与扩散模型：

- **统一潜空间编解码**（`src/latent_codec.mora`）：使用分词器与 Mini-LLaMA 词嵌入 / 输出头，将离散文本前缀与待生成后缀映射至连续空间 $z \in \mathbb{R}^{B \times (K B_s) \times D}$，去噪后再解码回离散 token。
- **单一统一入口**（`src/cad_bebc.mora`）：输入统一经过潜空间编码与无路由器竞争门控，经由三条通路完成推理与演进：
  1. **外部冻结专家快速通道**：直接调用预训练冻结的 Mini-LLaMA（`mode="llama"`）或 HTTP 服务，当 $\pi_{\text{int}} < 0.05$ 时走纯外部路径（0 扩散步）；
  2. **内化短通路**：成熟度直接写入权重 $(U, V, w_g, b_g)$ 自身，门控突破 `theta_skip` 时，该段的 DDIM 更新被学到的跨步低秩映射替换，**骨干网络在该段不被求值**（这正是 NFE 节省的来源）；
  3. **连续块扩散主干**：多步连续流匹配 ODE 去噪（深度思考），并驱动内化权重缓慢生长或高误差回退。

---

## 2. 噪声过程与采样器（VP + DDIM）

前向过程为标准**方差保持（VP）**过程：

$$z_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad
\bar\alpha_t = \prod_{i \le t}(1-\beta_i),\ \beta_i \text{ 线性于 } [\beta_{\text{start}},\beta_{\text{end}}]$$

观测 $x_{\text{obs}}$ **不以条件方式进入噪声项**，而是通过**独立投影加性注入**骨干：

$$h = (z_t @ W_{\text{in}}) + (x_{\text{obs}} @ W_{\text{cond}}) + p_{\text{pos}} + \tau W_{\text{time}}$$

> 采用加性注入而非拼接：`Tensor::cat` 在 Mora 中**无 autograd 支持**——实测在 `value_and_grad` 内 loss 正确（100.0）但穿过拼接的梯度**精确为 0.0**，会静默冻结整个骨干。

采样器为 **DDIM**（$\eta = 0$，$x_0$ 参数化），实现于 `src/diffusion/schedule.mora`：

$$\epsilon_\theta = \frac{z_t - \sqrt{\bar\alpha_t}\,\hat x_0}{\sqrt{1-\bar\alpha_t}},\qquad
z_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s-\sigma_s^2}\,\epsilon_\theta + \sigma_s z_t$$

采样起点为**把观测投影到 VP 轨迹**（img2img 式条件采样）：

$$z_{t_0} = \sqrt{\bar\alpha_{t_0}}\,x_{\text{obs}} + \sqrt{1-\bar\alpha_{t_0}}\,\epsilon$$

起点层级由 `infer_start_step` 决定，默认取完整调度（8）。跳步段不消耗骨干求值，因此 **NFE 低于调度步数**。

> **层级一致性**：采样循环走到第 1 级即停止，最终的读出调用本身就是 $1\!\to\!0$ 这一步，因此满路径 NFE **恰为 `infer_start_step`**（8），读出时骨干被告知的 $\tau$ 与状态所在层级一致。短通路只替换 **2 级跨段**（与训练时的影子跨段 $8\!\to\!6$、$6\!\to\!4$、$4\!\to\!2$ 一致），最后一段 $2\!\to\!1$ 恒为真实骨干步，故成熟后满路径 8 次求值、短路径 2 次（详见 `docs/mechanism_fixes_2026-11-25.md`）。

---

## 3. 三层注意力与因果结构

注意力结构偏置由可微矩阵 $W_{\text{blk}} \in [-\text{init\_iso}, 0]^{K \times K}$ 与前缀因果掩码构成：

$$B_{\text{struct}} = P_{\text{blk}}\, (W_{\text{blk}} \odot M_{\text{off}})\, P_{\text{blk}}^T + M_{\text{prefix}}$$

1. **块内双向**（$b_i = b_j$）：偏置恒为 0，当前块内部 token 充分双向交互。
2. **前缀因果**（前缀块内部严格因果顺序）：由 $M_{\text{prefix}}$ 施加强屏蔽，保持自回归前缀顺序。
3. **块外双向因果**（其余 $b_i \neq b_j$）：由 $W_{\text{blk}}[b_i, b_j]$ 控制，通过误差门控反向传播缓慢生长或回退，允许前块影响后块，也允许后块反向影响前序块。

> $W_{\text{blk}}$ 的初值取 $-\text{init\_iso}\cdot 0.5$，**严格位于 clamp 区间内部**。此前初值恰在边界上，使任何"增强耦合"的梯度都被 clip 掉，耦合指标 $\text{mean}(\exp W_{\text{blk}})$ 全程不动。

---

## 4. 误差门控的纯权重涌现机制（完全可微）

不维护显式巩固度 $c$ 或误差滑动均值，成熟度完全写入权重 $(U, V, w_g, b_g, W_{\text{blk}}, w_{\text{comp}}, b_{\text{comp}})$ 自身。所有参数通过统一的 `value_and_grad(unified_diff_loss)` 获得梯度，再用软门缩放步长。

### 4.1 驱动信号与静默死区门

- 影子对齐误差 $\tilde{\mathcal{E}}_k$（短通路对齐）
- 一致性误差 $\mathcal{E}_{\text{base}}$

$$m_{\text{low}} = \sigma\!\left(\frac{\epsilon_{\text{low}} - \mathcal{E}_{\text{base}}}{T_e}\right), \quad m_{\text{fail}} = \sigma\!\left(\frac{\mathcal{E}_{\text{base}} - \epsilon_{\text{high}}}{T_e}\right)$$

当前主更新路径以 $m_{\text{low}}$ 作为正向步长缩放（`step_ok`），以 $m_{\text{fail}}$ 作为回退缩放。

### 4.2 统一可微损失与不对称更新（慢生长 / 快回退）

`unified_diff_loss` 同时包含：
- 混合重构损失（内部输出与外部专家的连续混合）
- 短通路对齐损失（权重 `align_weight`，可配）
- 可微门控监督（**在激活前 logit 空间**，权重 `gate_sup_weight`，可配）

门控监督在 logit 空间而非概率空间：$b_{\text{rest}} = -5.5$ 时 $\sigma(-5.5) \approx 0.004$，$\mathrm{d}g/\mathrm{d}(\text{logit}) \approx 0.004$，概率空间的门梯度比 logit 梯度弱约 **250 倍**。监督目标取**执行阈值余量** $\text{logit}(\theta_{\text{skip}}) + \text{gate\_margin}$（而非 $\text{logit}(0.999)\approx +7$），使成熟斜坡从 $b_{\text{rest}}$ 可达。

在 `emerge_step` 中应用：

- **跨步低秩映射**（$U, V$）：
  $$U, V \leftarrow (1 - \lambda_{\text{decay}} m_{\text{fail}}) \cdot \{U,V\} - \eta_{\text{lr}} m_{\text{low}} \nabla \mathcal{L}$$
  > $V$ 的初值取小随机而非精确零：精确零时 $\mathrm{d}\mathcal{L}/\mathrm{d}U \equiv 0$（实测 $\lvert g_u\rvert = 0.0$），$U$ 永远学不到，$V$ 只能靠 $\lvert U\rvert = 0.01$ 的微弱因子推动。
- **跳过门神经元**（$g = \sigma(w_g^T \bar{z} + b_g)$，静息 $b_{\text{rest}}=-5.5$）：
  $$w_g, b_g \leftarrow \text{可微监督梯度} \times m_{\text{low}} + \text{回退到 } b_{\text{rest}} \times m_{\text{fail}}$$
- **块外因果 $W_{\text{blk}}$** 与 **竞争门**：通过同一反向传播梯度，乘以 $m_{\text{low}} / m_{\text{fail}}$ 后更新，并约束在 $[-\text{init\_iso}, 0]$。

### 4.3 竞争门：按误差优势定向

竞争门**不**用无条件正则（原 `lambda_ext_bias * pi_int.sum()` 每步无条件把 $\pi_{\text{int}}$ 往下压，实测内部路径明显更准时 $g_{b_{\text{comp}}}$ 仍为负 $-0.0057$），而是监督到**误差更低的那一支**：

$$\text{advantage} = e_{\text{ext}} - e_{\text{int}},\qquad
\text{target\_logit} = \text{logit}\!\left(\sigma(\text{comp\_adv\_scale}\cdot\text{sign}(\text{advantage}))\right)$$

两条分支必须在**同一输入**上比较（此前内部路径从中间 ODE 态出发、外部专家吃原始观测，"谁更准"没有意义，实测每个真实语料窗口上 advantage 都是负的）。

---

## 5. 外部冻结专家与竞争

- 默认 `mode="llama"`：加载冻结 Mini-LLaMA（`models/mini_llama_frozen.safetensors`）；缺 checkpoint 时自动回退到随机初始化。
- 推理时：竞争门 $\pi_{\text{int}}$ 连续混合内部输出与外部专家；仅当 $\pi_{\text{int}} < 0.05$ 时完全跳过内部计算。
- 无显式路由器，竞争完全由权重与误差信号驱动。短通路跳步决策由门控 $g_k$ 独立控制，与 $\pi_{\text{int}}$ 无直接硬耦合。

---

## 6. 定位与边界

- 单一 **CAD-BEBC** 统一系统。
- `config/diffusion.json`（微型）用于 CPU 秒级验证完整生命周期：冷启动 → 误差门控权重涌现 → 真跳步 → OOD 回退。
- 工业级目标配置见 `config/diffusion.industrial.json`（d=256, 8 层, 16 块）。
- 当前为原型验证阶段，大规模能力评估待进一步实现。
- **已验证的结论与未验证的部分见 `docs/mechanism_fixes_2026-11-24.md`（第一/二轮）与 `docs/mechanism_fixes_2026-11-25.md`（第三轮）**，其中明确区分了"短通路的计算效率收益（已实测：2 次求值优于 8 次求值）"与"多步采样优于单步（当前不成立，原因已量化）"。
