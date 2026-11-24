# CAD-BEBC 机制修复与实测记录

> 本文记录 2026-11-24 对 CAD-BEBC（`cad-bebc-unified-v2`）的一次机制级修复。
> 所有数字均来自本地实跑（`mora src/train_cad_bebc.mora`）与 Hugging Face Space
> `redmingwei/cad-bebc-train` 的同配置运行，两者逐位一致。
>
> 修复前：`mora test` = 7 passed / 4 failed。修复后：15 passed / 0 failed。

---

## 1. 修复前的实测状态

在动代码之前，先测出基线（`config/diffusion.json`，720 步 consolidation）：

| 观测项 | 修复前实测 | 含义 |
|---|---|---|
| `\|g_u\|` | **0.0**（精确零） | U 永远学不到 |
| `\|g_v\|` | 7.0e-6 | V 只能以二阶速度爬 |
| `V_rms`（720 步后） | 0.00032 | 短通路实质恒等映射 |
| 块外耦合 `coupling` | 0.002478752176666359 → 0.0024790725693325496（**+0.013%**） | `W_blk` 几乎不生长 |
| 短通路跳步后 MSE | 0.2905 → 0.2970（+2.66%） | 跳 6 步几乎不改变输出 |
| `pi_int` | 0.018，单调下降 | 永久钉在外部快路径 |
| 冷启动 `pi_int` | 0.0759 > 0.05 | 冷启动不走外部快路径 |

结论：修复前"NFE 8→2"虽然**在计数上成立**，但 `V≈0` 意味着跳掉的 6 步被一个近似
恒等映射替代；"块外双向因果由误差门控生长"则完全没有发生（+0.013%）。

---

## 2. 六个机制缺陷与修复

### 2.1 短通路 U/V 双线性冷启动死锁

`shortcut_apply = z + silu(z @ U) @ V^T`，且 `V` 初值为**精确零**：

```
d(loss)/dU ∝ V = 0    ⇒ |g_u| 精确为 0（实测 0.0）
d(loss)/dV ∝ silu(z @ U)，而 |U| = 0.01 ⇒ |g_v| 仅 7.0e-6
```

U 被永久冻结，V 只能靠 `|U|=0.01` 这个微弱因子推动。

**修复**：`V` 改为小随机初值（仍近似恒等，因为 V 以乘性残差进入），`U` 尺度提到 0.5。

**实测效果**：

| | 修复前 | 修复后 |
|---|---|---|
| `\|g_u\|` | 0.0 | 2.0e-5 |
| `\|g_v\|` | 7.0e-6 | 8.2e-4（**117×**） |
| loss | 8.06984161569635 | 8.06981320392422（语义未变） |

### 2.2 门控监督在概率空间 → 梯度被 sigmoid 饱和压死

`b_rest = -5.5` 时 `sigmoid(-5.5) ≈ 0.004`，于是 `d(g)/d(logit) = g(1-g) ≈ 0.004`，
门控梯度比 logit 梯度弱约 **250 倍**。

**修复**：改为监督**激活前 logit**，且目标取执行阈值余量
`logit(theta_skip) + gate_margin`（= 1.49），而非 `logit(0.999) ≈ +7`
——门只需"够用"即可跳步，无需饱和到 1.0，且成熟斜坡从 `b_rest` 可达。

**实测**：`|g_bg|` 从 0.0202 提升到 0.74（37×）。

### 2.3 `init_iso` 过大导致 softmax 饱和，块外耦合梯度消失

`init_iso = 6.0` 让块间注意力 logit 差 6.0，softmax 后梯度极小。实测扫描：

| `init_iso` | `\|g_wblk\|` | 初始 coupling |
|---|---|---|
| 6.0 | 1.78e-5 | 0.0025 |
| 4.0 | 1.43e-4 | 0.0183 |
| **2.0** | **4.84e-4** | 0.1353 |
| 1.0 | 4.10e-4 | 0.3679 |
| 0.5 | 3.63e-4 | 0.6065 |

**修复**：取 `init_iso = 2.0`——梯度提升 **27×**，同时初始 coupling 0.135 仍保留
"块间弱耦合"语义（0.5 会让初始就强耦合）。

### 2.4 竞争门被无条件正则钉死在外部路径

原损失含 `lambda_ext_bias * pi_int.sum()`，**每步无条件**把 `pi_int` 往下压。
实测：内部路径明显更准（MSE 0.2895 vs 0.3615），`g_bcomp` 却是 **-0.0057**（负）。

更根本的问题是**两条分支的输入不对称**：内部路径从中间 ODE 态 `z_pre_final`
出发，外部专家吃原始观测 `x_obs`，"谁更准"的比较本身没有意义——实测在每个真实
语料窗口上 `advantage` 都是负的。

**修复**：
1. 竞争门改为**监督到误差更低的那一支**（logit 空间，`comp_gate_weight`）；
2. 比较时**两边用同一输入**（内部路径也从 `x_obs` 出发，与 `infer_system` 的推理
   语义一致）；
3. 目标取 `advantage` 的**符号**而非幅度（真实语料上 advantage 幅度小且噪声大，
   用幅度会让目标 logit 在窗口间翻转，净位移近零）；
4. 竞争门用独立的 `lambda_comp`，不再复用为跳步门设计的 `lambda_b`。

**实测**：`pi_int` 0.018 → **0.982**，`mode` 从 `external_fast` 变为 `internal_full`。

### 2.5 损失必须保持张量形态

Mora 的 autograd 只对**张量**输出播种根梯度
（`src/runtime/autograd.rs`：`grads[out_idx] = Some(vec![1.0])`）；
返回 float 会走 `zero_grads_for_leaves` 路径，所有叶子梯度归零。

`unified_diff_loss` 因此返回 1 元素张量，并让 `recon` 表达式里保留 `pi_int`
以维持到 `p_wcomp / p_bcomp` 的 autograd 边。

### 2.6 两处 API 误用

- `Tensor.item()` 在**可微上下文**返回张量，`float()` 不接受张量 → 改用 `to_list()`。
- `float(tensor)` 不是合法转换。

---

## 3. 修复前后对照（720 步，同一配置）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 跳步是否触发 | nfe=2（但 V≈0，假跳步） | nfe=2, skips=3 |
| 跳步精度损失 | +2.66% | **+1.55%** |
| 块外耦合生长 | +0.013% | **+0.206%（16×）** |
| `pi_int` | 0.018 → 永久外部路径 | 0.018 → 0.982，内部路径激活 |
| `bg` 成熟 | -5.5 → 2.94（饱和梯度下） | -5.5 → 1.00（logit 空间，可达阈值） |
| `mora test` | 7 passed / 4 failed | **15 passed / 0 failed** |

---

## 4. 真实语料训练

新增 `src/train_cad_bebc.mora`（此前 CAD-BEBC 只在合成正弦数据上训练过；
`src/train.mora` 训的是外部专家，不是被研究的架构本体）。

语料路径新增 `src/data/corpus.mora`：原路径对整份语料做逐字节 `list.push`
（2 MB TinyStories 实测 **>300 s**），改为张量切片按需取窗口，
2000 窗口 **0.104 s**，token id 完全一致。

**训练结果**（720 步，2,000,000 字节 TinyStories，潜空间 = Mini-LLaMA token embedding）：

| 指标 | 值 |
|---|---|
| final train loss | 0.3677 |
| held-out full-path MSE | 0.2454 |
| held-out shortcut-path MSE | 0.1463 |
| backbone NFE（短通路） | 2.0（自 8 降） |
| shortcuts used | 3.0 |
| skip gate bias `b_g` | 0.999 |
| inter-block coupling | 0.1387 |

---

## 5. 必须诚实说明的负面结果

**短通路比满路径更准（0.1463 vs 0.2454），但这不证明短通路有效。**

对骨干单独测量 MSE 与 ODE 步数的关系：

| ODE 步数 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| MSE | **0.0969** | 0.1207 | 0.1503 | 0.1802 | 0.2081 | 0.2334 | 0.2558 | 0.2755 |

观测基线 `MSE(x_obs, x_0) = 0.1180`。

即 **MSE 随步数单调恶化**，而 1 步（0.0969）优于观测基线。原因是
`backbone_ode_step` 做 `z + (x0h - z)/step_t` 的混合，`step_t=1` 时完全替换为
`x0h`，而多步会把同一个粗糙的 `x0h` 反复叠加（直接替换版 8 步更差：0.8822）。

**结论：当前骨干只在单步预测下准确，多步 ODE 必然退化。** 因此"NFE 8→2"
目前**不构成内化短通路有效的证据**——减少 NFE 有利是因为多步本身有害。

短通路的实际修正量也偏小：`|delta| rms = 0.0328` 对 `|z| rms = 0.331`
（约 10% 相对修正），`V_rms = 0.0089`。

块外耦合 720 步只长 0.206%，远小于文档暗示的量级。

### 待解决的下一个问题

骨干的多步退化是**架构级**问题（连续扩散主干的多步离散化），不是超参问题。
可行的方向：
1. 把 `backbone_ode_step` 换成真正的 flow-matching / DDIM 离散化，
   让 `x0h` 随 t 变化而非固定；
2. 或让骨干在多步下被显式训练（当前 `pretrain_backbone` 只训单步 x0 预测）；
3. 或重定义 NFE 指标——在当前骨干下，"NFE 越少越准"，8→2 的卖点本身需要
   重新论证。

---

## 6. 复现

```bash
mora src/train_cad_bebc.mora          # 训练 + 评估 + 写 checkpoint/metrics
mora test                              # 15 passed / 0 failed
```

HF Space `redmingwei/cad-bebc-train` 的运行与本地运行**逐位一致**
（step-720 train loss `0.3676810575090782`，两边相同）。

产物：`https://huggingface.co/redmingwei/cad-bebc-tinystories`
