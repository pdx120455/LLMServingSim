# LLMServingSim 支持 GLM-5.1 (FP8) 计划

> 目标：在 H20 8 卡服务器上 profile GLM-5.1 FP8，产出可被 LLMServingSim simulator 消费的性能数据，最终把支持回馈给上游项目。
>
> 文档维护：`@pdx_ll`，最新更新 2026-06-03。

---

## 1. 目标与定位

### 终态
- **profiler 侧**：`profiler/perf/H20/zai-org/GLM-5.1/fp8/tp{1,8}/{dense,per_sequence,attention,moe,skew,skew_fit}.csv` + `meta.yaml`（只 profile tp1/tp8：tp2/tp4 对 GLM-5.1 FP8 物理不可行，见决策日志 2026-06-27）
- **simulator 侧**：`python -m serving --cluster-config <H20_8x_glm5.1.json> --workload <xxx.jsonl>` 能跑完整模拟，cycle 数误差 < 20%（对比真实 vLLM bench）
- **上游**：PR 合入 LLMServingSim main，含新 yaml + 必要的 simulator 改动 + 测试用例

### 非目标
- 不做 GLM-4 / GLM-4.5 / 其他 GLM 变体
- 不在 5060Ti 这种小卡上验证（已确认 OOM）
- 不在第一版支持 MTP speculative decoding（layer 78 的 `num_nextn_predict_layers=1`）

---

## 2. 模型架构关键事实

```
architectures: ["GlmMoeDsaForCausalLM"]      → vllm/model_executor/models/deepseek_v2.py:1744
model_type: "glm_moe_dsa"                    → 需要 profiler/models/glm_moe_dsa.yaml (不存在)
父类: DeepseekV2ForCausalLM                  → 复用 DeepSeek-V3.2 全套基础设施
量化: FP8 e4m3 + block_size [128,128]        → 静态权重量化，动态激活量化
```

### 结构要点
| 组件 | GLM-5.1 实现 | 影响 |
|---|---|---|
| Attention | **MLA** (Multi-head Latent Attention) | 没有 `qkv_proj`，分裂成 `q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj` |
| KV cache | 压缩 latent `kv_lora_rank=512` | 实际占用 ~10× 小于 GQA 估算 |
| Q 投影 | LoRA-style `q_lora_rank=2048` | TP 切分要包含 `q_lora_rank` |
| Position | split rope/nope: `qk_nope_head_dim=192` + `qk_rope_head_dim=64` | 影响 sequence 中 rotary_emb 的语义 |
| Sparse Attn | **Lightning Indexer** + `index_topk=2048` | 主 attention 只看 top-2048，cost 模型完全不同 |
| MoE | `n_routed_experts=256, n_shared_experts=1, top_k=8, noaux_tc` | shared expert 每 token 必激活；routing 算法跟 force_moe_routing 不兼容 |
| Layer 混合 | `first_k_dense_replace=3` → 前 3 层 dense MLP，后 75 层 MoE | profiler 不能假设单层代表全模型 |
| MTP head | layer 78 `num_nextn_predict_layers=1` | 第一版不支持，simulator 忽略 |

---

## 3. 任务清单（优先级 P0 > P1 > P2 > P3）

### Phase 1 - profiler 跑通（P0，2-4 天）

| # | 任务 | 文件 | 优先级 | 预计 | 谁 |
|---|---|---|---|---|---|
| 1.1 | 新增 `profiler/models/glm_moe_dsa.yaml`，映射 14 个核心 class | 新文件 | P0 | 0.5d | ✅ 2026-06-03 完成（13 dense + 2 per_seq + 1 attention + 1 moe，pydantic 验证通过） |
| 1.2 | ~~`SHARD_FIELDS` 加入 `q_lora_rank, kv_lora_rank`~~ | `profiler/core/config.py:39` | ~~P0~~ | - | ❌ 2026-06-03 撤销：源码确认 LoRA 段不沿 TP 切，无需新增（见决策日志） |
| 1.3 | profiler 能在 H20 上以 `--skip-skew --skip-moe` 启动并写出 dense/per_sequence/attention CSV(dense 轮**必带 `--skip-moe`**,否则 moe shots 在无 FusedMoE 的单层 dense 模型上 RuntimeError 中断整个 sweep;flag 2026-06-12 加入,见决策日志) | 运行验证 | P0 | 0.5d | 人 + Claude |
| 1.4 | MoE category 支持 shared expert：在 fire() 中标注 "shared 必激活" | `profiler/core/hooks/moe_hook.py`, `categories.py::ExpertCategory` | P0 | 1d | Claude。**前置已解**：hook 已重写适配 vLLM 0.21 MoERunner（R11，commit `7bdde56`，5060Ti 验证通过）；H20 上先跑 `python -m profiler.core.hooks.verify_moe_hook` 复验 |
| 1.5 | 处理 `first_k_dense_replace`：profile 两种 layer（dense MLP 层 + MoE 层），写到不同 CSV 段 | `profiler/core/runner.py`, `categories.py` | P1 | 1d | Claude |
| 1.6 | Lightning Indexer 作为独立 layer 出现在 dense.csv（或新建 indexer.csv） | `profiler/models/glm_moe_dsa.yaml`, `categories.py` | P1 | 0.5d | Claude |
| 1.7 | FP8 variant 自动命名 `fp8-bf16kv` / `fp8` —— 当前 `effective_variant` 应该已支持 | 验证 `profiler/core/config.py::ProfileArgs.effective_variant` | P2 | 0.1d | Claude |

### Phase 2 - simulator 能消费（P0，2-3 周）

| # | 任务 | 文件 | 优先级 | 状态 |
|---|---|---|---|---|
| 2.1 | `memory_model.py::calculate_sizes` 支持 MLA：KV 用 `kv_lora_rank + qk_rope_head_dim`、TP 不切、不乘 2 | `serving/core/memory_model.py` | P0 | ✅ 2026-06-03 完成（is_mla 用 `kv_lora_rank` 检测；mla_kv_elems=576；rotary/attention/o_proj 走 MLA 公式；新增 fused_qkv_a_proj/mla_lora_layernorm/q_b_proj/kv_b_proj/indexer 条目） |
| 2.2 | 按 `first_k_dense_replace` 逐层区分 dense / MoE（trace_generator **和** calculate_sizes 都要） | `serving/core/trace_generator.py` + `memory_model.py` | P0 | ✅ 2026-06-04 完成。新增 `_is_moe_layer(config, layer_num)`（`layer_num >= first_k_dense_replace` 且按 `moe_layer_freq` 取模）+ `_layer_segments()`（把 [0,num_layers) 切成 dense 段 + MoE 段，`first_k=0`/`freq<=1` 退回单段，零回归）。`_emit_post_attn_layers` 改逐层判断；block-copy 主循环 + interleaved 中段都按段建块复制（dense 段 ×first_k + MoE 段 ×(num_layers-first_k)）。`memory_model`：`is_moe` 补 `n_routed_experts`、新增 `first_k_dense`、`get_weight` 按段累加（heaviest-rank 上界，pp=1 精确）、`_get_weight_per_block(is_moe_block=)`。dry-run 验证：nl=2 fk=1 → 1 dense+1 moe；nl=4 fk=3 → 3 dense+1 moe；fk=0 全 moe（回归）；权重 moe_block 19.68GB ≫ dense_block 0.8GB |
| 2.3 | `trace_generator.py` 加 MLA 投影的查表逻辑（fused_qkv_a / q_b / kv_b 当 dense layer 处理） | `serving/core/trace_generator.py::_emit_pre_attn_layers` | P0 | ✅ 2026-06-03 完成（trace_generator 改走 yaml `sequence:` walk，所有 MLA-specific 条目都由 `_emit_sequence` 触发 `_lookup_dense`；kv_b_proj 不进 sequence，时间归 attention） |
| 2.4 | Indexer 作为独立 layer 嵌入 `pre_attn` sequence | 同上 | P0 | ✅ 2026-06-03 完成（yaml `pre_attn: [..., indexer, attention]` 自动驱动） |
| 2.5 | 修 attention 查表：MLA 的 cost 跟 `kv_lora_rank` 相关，需要在 lookup key 里加新维度，或在 yaml 中额外标注 | `serving/core/trace_generator.py::_lookup_attention*` | P1 | ⏳ 待 H20 profile（attention.csv 4D grid 当前对 MLA 直接复用，量级偏差待真实数据回归后再决定是否扩 5D） |
| 2.6 | sparse attn cost 模型：超过 `index_topk` 后 cost 平坦化 | `serving/core/trace_generator.py` | P1 | ⏳ 待 H20 profile（与 2.5 同批处理） |
| 2.7 | ~~shared expert 加进 MoE cost~~ | `serving/core/trace_generator.py` | ~~P0~~ | ❌ 2026-06-04 撤销：源码 + trace 证实 shared expert 在 `FusedMoE.forward` **内部**计算（见决策日志），moe.csv 已含其时间，simulator 无需单独 emit（否则双计）。剩 H20-time profiler 校验：确认强制路由 patch 未 bypass shared kernel |
| 2.8 | ~~`noaux_tc` 路由实现~~ → **重定义为 group-limited routing**（`n_group`/`topk_group` 掩码 + EP-to-group 对齐）。GLM-5.1 `n_group=1` 用不上，仅 DeepSeek-V3 家族（`n_group=8, topk_group=4`）受益 | `serving/core/gate_function.py::route_ep` | P3（GLM-5.1 无需） | ⏳ 延后。GLM-5.1 用 `BALANCED` 已是物理合理近似（见决策日志 2026-06-04 noaux_tc 条） |
| 2.9 | cluster config 样例 | `configs/cluster/` | P1 | ✅ 2026-06-04（三档齐：`single_node_glm51_smoke.json` tp=1 / `single_node_glm51_tp2_ep2.json` tp=2 ep=2 / `h20_8_glm5_1_fp8.json` tp=8 ep=8 8 卡。H20 模板 bw/latency/mem_bw 为占位待实测；config_builder 解析通过 (8=tp8·pp1, ep8)） |

### Phase 3 - validation（P0，1 周）

| # | 任务 | 工具 | 优先级 | 预计 |
|---|---|---|---|---|
| 3.1 | 在 H20 8 卡跑真实 vLLM benchmark：`python -m bench run` 用同 cluster config + 至少 1 个 workload | `bench/` | P0 | 1d |
| 3.2 | sim vs 真实对比：`python -m bench validate`，看 throughput / TTFT / TPOT 误差 | `bench/core/validate.py` | P0 | 1d |
| 3.3 | 误差 > 20% 时回 Phase 2 反向调试（优先盯 attention/moe 的 cost 模型）| - | P0 | 2-5d 不定 |

### Phase 4 - contribute upstream（P1，1 周）

| # | 任务 | 优先级 |
|---|---|---|
| 4.1 | LLMServingSim 仓库开 issue 描述 GLM-5.1 / DeepSeek-V3.2 架构支持需求，附本文档 | ✅ 2026-06-04 草稿完成（`GLM5_1_UPSTREAM_ISSUE_DRAFT.md`，英文，含架构差异 / 提议改动 / 3 个模型无关 bug / 已原型化清单 / 4 个待维护者确认的开放问题；**未发布**，待人工 review 后粘贴到 casys-kaist/LLMServingSim issues） |
| 4.2 | PR 分两个：(a) profiler 侧 yaml + categories.py 改动，(b) simulator 侧 memory_model + trace_generator 改动 | P1 |
| 4.3 | 写 docs/ 下的支持说明（MLA / DSA 是 vanilla 之外的第一个非标架构） | P2 |

---

## 4. 风险与未决问题

### 高风险（可能阻塞 Phase 1）
- **R1**: vLLM 0.21 在 H20（sm_90）上 FP8 + MLA + sparse_indexer 组合的 first-call JIT 编译时间未知，可能 5-30 min
  - **缓解**：先开 `VLLM_LOGGING_LEVEL=DEBUG` 跑一次，记录基线
- **R2**: profiler `worker_extension_cls` 在 vLLM 0.21 上的 hook 接口可能跟 0.19 有差异（已在 5060Ti 上观察到 `_typeshed` import bug）
  - **缓解**：先在 H20 上 dry run，看是否需要新 patch
- **R3**: ~~`DeepseekV2MLAAttention` 类在 layerwise_profile 中的命名 / 嵌套结构未亲眼确认，yaml 的 `vllm:` 字段可能要改~~
  - ✅ 2026-06-03 已源码确认：vLLM 0.21 `deepseek_v2.py` + `mla.py` 的 class 层级与 glm5_h20/trace.sqlite NVTX 嵌套一致；yaml 已按源码 leaf 类名写定。剩余风险：profiler tree 实际 ancestor chain 跟我们假设的 forward call stack 有没有 PluggableLayer / impl 注册导致的额外层 —— Phase 1.3 跑完后 dump tree 复核

### 中风险（影响 Phase 2 设计）
- **R4**: `kv_lora_rank` / sparse indexer 这些维度引入新的 lookup 轴后，attention.csv 的 4D 网格可能不够，要扩成 5D 或 6D，扩谁、怎么 backward-compatible 是个设计题
  - **缓解**：先做 Phase 2.5 的设计 doc，再动 code
- **R5**: `first_k_dense_replace` 让模型不同层成本不同 ——`HOST_ENGINE_DEFAULTS["hf_overrides"]={"num_hidden_layers": 1}` 这个核心假设需要打破，可能要 profile 两次（一次只跑 dense layer，一次只跑 MoE layer）
  - **缓解**：用 `hf_overrides` 控制 `first_k_dense_replace`：profile dense 时设 `first_k_dense_replace=1` + `num_hidden_layers=1`；profile MoE 时设 `first_k_dense_replace=0` + `num_hidden_layers=1`
- **R6**: shared expert 在 vLLM 实现里是否走 `FusedMoE` 还是单独 `DeepseekV2MLP` 实例 —— 影响 profile 的 layer 归属
  - ~~阅读 `deepseek_v2.py::DeepseekV2MoE` 已确认：shared_experts 是独立的 `DeepseekV2MLP`，**不**走 `FusedMoE`。所以 shared expert 在 dense.csv，routed experts 在 moe.csv。catalog 要相应分配~~
  - **✅ 2026-06-04 更正（前述结论错误）**：shared_experts 虽是独立 `DeepseekV2MLP` *实例*，但被**传入** `FusedMoE` 并在 `FusedMoE.forward` **内部**计算。证据三条独立一致：① `deepseek_v2.py:363-390` 的 `DeepseekV2MoE.forward` 只有一次 `self.experts(...)`（FusedMoE），无单独 `shared_experts(x)` 调用，构造时 `FusedMoE(shared_experts=self.shared_experts, ...)`（L329）；② `moe_runner.py:273` 有 shared_experts 时走 `_moe_forward_shared`，routed + shared 同一 forward 内算完；③ glm5_h20 trace(0.20.1) 的 `moe_shared_experts` 嵌套在 `moe.fused_experts` 内（见 twilight.md F2）。**结论：shared expert 归 moe.csv，不进 dense.csv；simulator 无需单独 emit（双计风险）。Phase 2.7 撤销**

### 低风险（实施时再处理）
- **R7**: H20 NVL 8 卡的拓扑 ASTRA-Sim 是否已建模 —— 大概率要新写 network.yml 模板
- **R8**: MTP head 暂不支持，real bench 时需要禁用（`--num-speculative-tokens 0`）保证对比公平
- **R9**: 模型权重 357GB FP8，H20 8 卡 × 96GB = 768GB 够，但 KV cache 留多少需要算（202752 context × 78 layer × 512 kv_lora_rank × FP8 ≈ 8GB / sequence！）

### 2026-06-11 架构评审新增（R10/R11 已修复，R12-R14 记录在案）
- **R10 ✅ 已修**: simulator 单一 `fp` 混淆 weight/activation/KV 三种精度 —— `--dtype fp8` 会把通信量减半 + KV 容量 2× 高估（真实 activation/latent-KV 是 bf16）；`--dtype bfloat16` 则权重 2× 高估且 variant 不匹配。**无参数组合能同时正确**。修复：`fp` 语义改为 activation 精度（`max(weight_bits,16)`），新增 `weight_fp` 走权重核算（commit `43b12c4`，bf16 路径 bit-exact 回归 + fp8 trace 三点验证）
- **R11 ✅ 已修**: profiler MoE hook 在 vLLM 0.21 上 API 不存在 —— `FusedMoE.forward_native` 已被移除（`forward` 委托 MoERunner），MoE profile 启动即 AttributeError；0.19 有 hook 但不认识 GlmMoeDsaForCausalLM（版本死锁）。修复：双入口 patch + `runner.router._compute_routing` 锻造 + monolithic quant method 守卫（commit `7bdde56`，5060Ti `verify_moe_hook.py` 实测 5 forced vs 8 natural）
- **R12**: Indexer 成本被建模为 O(tokens)，但 DSA indexer 要对全部缓存 token 打分（`sm90_fp8_paged_mqa_logits` 是 O(kv_len)）—— 长上下文 decode 系统性低估。**第一轮验证 workload 必须避开长上下文**；长上下文支持留待 Phase 2.6 拆 indexer category
- **R13**: KV 内存漏算 indexer k-cache（~128B/token FP8，约 11% 低估）；`modules_to_not_convert`（embed/lm_head/norms 保持 BF16）在 fp8 权重核算下被低估（per-rank 数百 MB 量级，占比小）
- **R14**: 验证方法学 —— ① profile 边界（`ATTENTION_MAX_KV`、`MAX_NUM_BATCHED_TOKENS`）必须 ≥ 验证 workload 实际分布，否则全程外推；② 单 workload 通过可能是过拟合，建议至少两个长度 regime；③ profile 与 bench 必须 pin 同一 vLLM 版本（yaml 类名按 0.21 写定 → 两边都用 0.21，**不要**用 0.20.1）

---

## 5. 依赖与前置条件

### 软件
- [ ] H20 服务器 conda 环境：vLLM >= 0.21.0（验证 `GlmMoeDsaForCausalLM` 在 registry）
- [ ] H20 服务器 CUDA >= 12.4（FP8 + MLA 需要）
- [ ] LLMServingSim 仓库（当前 branch `main`，commit `a385673`）
- [x] ASTRA-Sim + Chakra 已编译（2026-06-04 本机原生编译，`AnalyticalAstra` 二进制可用；chakra 经符号链接装进 vllm_020）

### 数据
- [ ] GLM-5.1 模型权重已下载（路径：需 H20 上确认；本机有 `/home/luoliang/workspace/glm5.1/`）
- [ ] 用于 bench 的 workload JSONL（先用 ShareGPT 即可）

### 硬件
- [ ] H20 8 卡服务器访问权
- [ ] 至少 200GB 临时盘空间（profile 期间的 log + CSV）

### 流程
- [ ] 跟 LLMServingSim 维护者邮件 / GitHub issue 对齐设计方向（Phase 4 前要做）

---

## 6. 当前已完成 / 阻塞中

### 6.0 完成度统计（2026-06-05 核对）

全部 22 个任务（Phase 1-4，权威任务表见 §3）：

| 状态 | 数 | 任务 |
|---|---|---|
| ✅ 完成 | 8 | 1.1 / 1.6 / 2.1 / 2.2 / 2.3 / 2.4 / 2.9 / 4.1(草稿) |
| ❌ 撤销(已决) | 2 | 1.2 SHARD_FIELDS / 2.7 shared expert |
| ⬇ 降 P3(GLM 无需) | 1 | 2.8 group-limited routing |
| ⏳ 阻塞 H20 | 9 | 1.3/1.4/1.5/1.7 · 2.5/2.6 · 3.1/3.2/3.3 |
| ◻ 对外/未起 | 2 | 4.2 PR / 4.3 docs |

**不依赖 GPU 的本地工作已 100% 完成**，全部 GLM 代码已提交（`af02f81` / `4c35d6d` / `5e3d18b` / `1fbbb1c`），三份文档交付物 2026-06-05 提交。剩余全为 H20 数据（按 `H20_RUNBOOK.md` 执行）+ 对外动作。

### 已完成（2026-05-28）
- ✅ Qwen3-0.6B profile 在 vllm_020 + 5060Ti 上跑通（小规模快速验证，缩水 attention grid + skip skew）
- ✅ 定位并修复 vLLM 0.21 `_typeshed` 顶层 import bug（site-packages 内 patch）
- ✅ 发现并定位 profiler 的 pandas 隐式依赖（`writer.py:548 → skew.py:43`，未提 PR）
- ✅ 确认 GLM-5.1 = FP8 DeepSeek-V3.2 架构（MLA + Lightning Indexer + 256-expert MoE + MTP）
- ✅ 确认 5060Ti 不可能装下 GLM-5.1 单层（FP8 仍需 ~13GB / 16GB，加 KV/activation 必 OOM）

### 已完成（2026-06-03）
- ✅ Phase 1.1：`profiler/models/glm_moe_dsa.yaml` 13 dense + 2 per_seq + 1 attention + 1 moe，pydantic 验证通过
- ✅ Phase 1.2：源码分析后撤销 SHARD_FIELDS 改动
- ✅ Phase 2.1：`memory_model.py` MLA 分支（is_mla 检测 / 5 个 MLA 条目 / KV size 不切 TP 不乘 2）
- ✅ Phase 2.3 + 2.4：`trace_generator.py` 改走 yaml `sequence:` walk；MLA 投影 + Indexer 自动按 sequence 触发；MoE 块 EXPERT/EXPERT END 正确包裹
- ✅ Phase 2.9 部分：`single_node_glm51_smoke.json` (tp=1) + `single_node_glm51_tp2_ep2.json` (tp=2 ep=2)；占位 perf CSV bundle 落在 `profiler/perf/RTX5060Ti/zai-org/GLM-5.1/bf16/{meta.yaml, tp1/, tp2/}`

### 已完成（2026-06-04）
- ✅ Phase 2.7 撤销 + R6 更正：shared expert 在 `FusedMoE.forward` 内部计算（源码 `DeepseekV2MoE.forward` 单次 `self.experts(...)` + `moe_runner.py:273 _moe_forward_shared` + trace `moe_shared_experts` 嵌套于 `moe.fused_experts` 三证一致），moe.csv 已含，simulator 无需单独 emit
- ✅ A1/A2/A3/B 实测通过（早已实现于工作树，未提交）：`--hf-overrides` flag / `profile.sh` 透传 / `fuse_engine_kwargs` 三层合并（num_hidden_layers=1 + CLI first_k_dense_replace=0 + TP-shard）/ `detect_model_type → glm_moe_dsa`
- ✅ Phase 2.9 收尾：`configs/cluster/h20_8_glm5_1_fp8.json`（H20 8 卡 tp=8 ep=8，bw/latency/mem_bw 占位）；config_builder 解析通过
- ✅ Phase 2.2 完成：`first_k_dense_replace` 逐层 dense/MoE 切分（trace_generator `_is_moe_layer`/`_layer_segments` + block-copy/interleaved 分段复制；memory_model `get_weight` 按段累加）。dry-run 全部边界通过（详见任务表 2.2 行）
- ✅ config_builder MoE 检测补 `n_routed_experts` fallback（`config_builder.py:35` + 88）：GLM 不写 ep_size 时自动推断 `ep_size=tp_size=8`、ep-divides-256 校验生效（实测 ep=3 被拒）。与 trace_generator:1469 / memory_model.is_moe 三处一致
- ✅ `configs/model/zai-org/GLM-5.1.json` HF config 子集
- ✅ Phase 2.x（router / scheduler / agentic 兼容）dry-run 验证：standalone driver（无 ASTRA-Sim 子进程）驱动真实 Router+Scheduler+MemoryModel，19/19 check 通过。① `get_weight` first_k 分段精确（split == 1·dense_block + 3·moe_block，moe 20.6GB/层 ≫ dense 1.76GB/层 @ ep=1 tp=1）；② MLA KV 不随 TP 切（tp1==tp2==4608 B/token = 576·4层·2B）；③ agentic 依赖链：2 flat + 2 session(3+2 sub)=7 请求在 prefix-caching 开/关两种模式下全部 `add_done` 完成、`is_free()` 无泄漏。Router 模型无关无 GLM 特定逻辑。仅 mock ASTRA-Sim cycle 反馈 + 跳过 trace/graph（已单独验证）
- ✅ trace_generator 端到端 Python 链路验证：单层 16-layer trace / 2-layer override 28-layer trace 含 MoE 块；1D 拓扑产 ALLREDUCE + ALLGATHER+REDUCESCATTER；2D DP 拓扑产 `:1,0` 后缀
- ✅ 期间修两个 simulator bug：
  - `trace_generator.py:1469` MoE 检测加 `n_routed_experts` fallback（之前 DeepSeek/GLM family 不会激活 MoE 路径）
  - `trace_generator.py:965` `comm_type.split(':')[0].lower()` 剥 dim 后缀（DP 拓扑 + power_model 之前从未走通）
- ✅ **ASTRA-Sim 子进程端到端跑通（原生编译，非 docker）**：docker registry 拉取始终被阻塞（daemon 无代理、需 root），改走**原生编译**。已编出 `AnalyticalAstra` 二进制并跑通两档 GLM-5.1 smoke：
  - tp=1 (`single_node_glm51_smoke.json`)：2 请求全栈跑通（serving → chakra converter → AnalyticalAstra → cycle 反馈），CSV 2 行、latency 全正、`All Request Has Been Exited` 干净退出
  - tp=2 ep=2 (`single_node_glm51_tp2_ep2.json`)：2 请求干净退出，**348 个 ALLREDUCE 经 ASTRA-Sim**（TP collective 验证）。无 MoE ALLGATHER/REDUCESCATTER 属预期（smoke 配置 1 层 + first_k=1 = 全 dense，符合 first_k 逻辑）
  - 原生编译三处修复（不改 astra-sim 源码）：① `PROTOBUF_FROM_SOURCE=True` 用 protobuf CONFIG target；② `--start-group` 包裹 `/usr/local` 全部 absl 静态库（循环依赖）；③ shadow-include 覆盖 `absl/base/options.h` 的 `ABSL_OPTION_USE_STD_STRING_VIEW 0→2`（系统 `/usr/local` abseil 安装不一致:头文件写 0 但库按 std::string_view 编;系统头 root 只读）
  - chakra 经 namespace-package 符号链接装进 vllm_020（源码无 `__init__.py`、pip build 被代理阻塞）；pyinstrument 已装；protobuf 6.33.6 已满足
  - 占位 perf bundle → **latency 数值不可信**，本 smoke 只验证执行路径 + IPC，不验证 cycle 真实性
  - 期间修第三个 simulator bug：`__main__.py` dtype-peek（① 缺 `../` 前缀致 chdir 到 astra-sim 后读不到 cluster config；② 读顶层 `instances` 而非 `nodes[].instances`，torch_dtype 探测一直失效）——GLM-5.1 `torch_dtype=null` + 不传 `--dtype` 时暴露

- ✅ Phase 4.1：上游 issue 草稿 `GLM5_1_UPSTREAM_ISSUE_DRAFT.md`（英文）——架构差异表 / 提议的 profiler+simulator 改动 / 3 个模型无关 bug（可拆独立小 PR）/ 已原型化清单 / 4 个待维护者确认的开放问题（DSA 是否独立 catalog、MLA attention 查表维度、noaux_tc 路由、MTP）。**未发布**，待人工 review 后粘贴到 casys-kaist/LLMServingSim

### 已完成（2026-06-11，架构评审整改）
- ✅ 资深架构师视角全方案评审：定位 2 个致命缺陷（R10 fp 三轨混淆 / R11 moe_hook 0.21 API 失效，均不在原 R1-R9 内、均会报废 Phase 3 验证），3 项中低风险记录（R12-R14）
- ✅ **Fix 1（commit `43b12c4`）**：dtype 三轨拆分 —— `fp` 改为 activation 精度、新增 `weight_fp`（serving/__main__.py / scheduler / memory_model / trace_generator 四文件）。验证：bf16 smoke CSV bit-exact；`--dtype fp8` 下 o_proj weight 100663296→50331648（减半 ✅）、ALLREDUCE comm 122880 不变 ✅、KV 1152B/token 不变 ✅。**`--dtype fp8` 现在是 H20 模拟的正确用法**
- ✅ **Fix 2（commit `7bdde56`）**：moe_hook 适配 vLLM 0.21 —— 双入口（forward_native ≤0.20 / forward ≥0.21）+ `runner.router._compute_routing` 实例级锻造 + monolithic 守卫（FlashInfer/TRT-LLM 融合路由不可强制时报错而非静默垃圾）。新增 `profiler/core/hooks/verify_moe_hook.py` 单卡自检（H20 换版本后必跑）。5060Ti vLLM 0.21.0 实测：forced 5 distinct experts vs natural 8，restore 无泄漏

### 阻塞中
- ⏸ 等 H20 服务器访问 → Phase 1.3-1.7 / Phase 2.5 / 2.6 / Phase 3 全部

---

## 7. 立即可推进项（不依赖 H20）

按依赖顺序排列：

1. ~~**Phase 1.1**：起草 `profiler/models/glm_moe_dsa.yaml`~~ ✅ 2026-06-03 完成
2. ~~**Phase 1.2**：patch `profiler/core/config.py::SHARD_FIELDS`~~ ❌ 撤销（源码确认无需改）
3. ~~**Phase 2.1+2.2**：`memory_model.py` MLA 分支~~ ✅ 2026-06-03 完成（2.1 完整，2.2 部分）
4. ~~**ASTRA-Sim 子进程端到端**~~ ✅ 2026-06-04 完成（**原生编译**，非 docker；tp=1 + tp=2 ep=2 两档 smoke 跑通，见第 6 节）
5. ~~**Phase 2.7**：shared expert 加 dense.csv 条目 或 MoE 时间补常数~~ ❌ 2026-06-04 撤销（shared expert 在 FusedMoE 内部，moe.csv 已含，见决策日志）
6. **Phase 2.9 H20 8 卡 cluster config 模板**：`configs/cluster/h20_8_glm5_1_fp8.json`，先写结构，bw / latency 留占位待 H20 实测填（~30 分钟）
7. ~~**Phase 4.1**：起草 LLMServingSim issue（描述需求 + 链接本文档，30 分钟）~~ ✅ 2026-06-04 完成（草稿落在 `GLM5_1_UPSTREAM_ISSUE_DRAFT.md`，未发布）

---

## 8. 决策日志

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-07-06 | **sim-vs-真机 误差缩小方法登记（分层 + 残差驱动，本轮仅文档，不改建模代码）** | 用户问"有哪些方法能减小模拟与真机差距"，选定=只要方法梳理、攻哪个误差源留到真机残差再定。前提：`profiler/perf/H20/.../fp8/` 现存 CSV 全是 5060Ti 合成占位（`meta.yaml: gpu: PLACEHOLDER`），真实 profile 落地前误差数值无意义、建模精修不可验证。**Tier 0（先决，H20-blocked，一阶误差源）**：真实 H20 profile 替换占位 CSV（RUNBOOK Step 1-3）——`trace_generator` 零标定（`_read_category_csv:267` time_us ×1000 直转 ns），profiler 时间原样决定精度。**Tier 1（对齐/方法学，便宜高杠杆，多数已在 RUNBOOK）**：① 两边引擎参数 pin 一致（sim 默认 max-num-seqs=128≠bench 256）；② workload 留 sweep 边界内（kv≤16384、n≤max-num-seqs），跑后 grep exceed 确认无外推；③ kv-cache-dtype 对齐真机（`memory_model.py:34` kv_fp，auto vs fp8 差 2× KV 字节→调度偏差）；④ nccl-tests busbw 标定 link_bw/link_latency（集合通信时间全在 ASTRA-Sim 侧算）；⑤ 两个长度 regime 防过拟合（R14）。**Tier 2（建模精修，等真机残差定位后按需攻，本轮不实现）**：通信-计算重叠（`llm_converter.py:413,451` 现全串行→sim 系统性偏慢，最大结构性偏差，代价重）、indexer 拆 O(kv_len) category（R12，长上下文偏快）、加密 attention 网格减外推（`_corner:777` 最近邻 / `_axis_bracket:542` 外推）、MoE straggler 惩罚（`_emit_moe_block:1095` max_rank 假设完美 overlap）、全局标定因子（过拟合风险）。**残差→方法映射**：系统性偏慢→先 nccl 标定再查通信重叠；仅长上下文偏快→indexer；吞吐系统性偏低→KV 精度或权重上界保守（`get_weight` per-rank 上界，pp>1 过估）；全程外推警告→workload 越界或扩 sweep。交叉链 R12/R13/R14。仅改本文件（决策日志）。未 push |
| 2026-06-29 | **修 profiler 漏抓 `final_layernorm`：glm yaml 的 `final_layernorm.within` 从 `DeepseekV2Model` 改为 null 兜底** | H20 dense dump 缺 `final_layernorm`（用户已手动补 tp1 临时绕过——该层 `tp_stable`，`_effective_tp`(trace_generator.py:499-506) 把所有 TP 查表重定向到 tp1，故补 tp1 即不报缺层；但根因未修，重跑 profiler 仍漏）。根因：`_match_slice`(timings.py:62-99) 要求 RMSNorm 节点祖先链含 `within` 字符串；GLM 用 `within: DeepseekV2Model`，但 0.21 实际最外层是 `GlmMoeDsaForCausalLM`、model 属性真实类名无从确认 → 字符串不在链 → 条目被跳过 → 不写行。修法：删 within 行（缺省=None），靠 deepest-match 的 null 兜底——final norm 是唯一祖先链既无 decoder-layer 又无 MLA-wrapper 类的 RMSNorm，只它命中 null，其余 RMSNorm 的真实 within(depth≥0) 永远 > null(-1) 故不被偷。**验证**：① Tier-1 纯逻辑（喂合成 module tree 给 `extract_samples`）三例全过——null 抓到 final norm 且不偷 input/qk，错误 within 复现缺层；② Tier-2 本机 5060Ti 重跑 Qwen3-0.6B（同机 vLLM 0.21.0）以 `--variant nulltest` 改 null 实测：final_layernorm 152 行抓到、层集合与 bf16 基线一致、layernorm/qk_norm 值噪声内（<2%）。硬证据：本机原有 `Qwen3-0.6B/bf16/tp1/dense.csv` 用正确 within 本就含 final_layernorm，证 0.21 确实捕获该节点。仅改 `glm_moe_dsa.yaml`（删 within 行）。未 push |
| 2026-06-27 | **GLM-5.1 profile sweep 收窄到 `TP_DEGREES="1,8"`（drop tp2/tp4）** | GLM-5.1 FP8 权重 ≈ 670GB，H20 单卡 96GB：tp4(4×96=384GB)、tp2(192GB) 都装不下权重 → 唯一可行部署是 **tp8（整 8 卡节点，768GB）**。模拟器按 cluster config 的 TP 度数精确查 `tp<N>/`，集群只会是 tp8，`tp2/`/`tp4/` 两张表永不被查、profile 它们纯浪费墙钟（每 TP 一轮引擎启动 + 全 sweep；skew 一轮 1-2h）。保留 **tp1** 作单卡调试/交叉校验基线。关键：profiler 的 "tp8" 是单卡 `hf_overrides` 切 shape 模拟（不真加载 670GB）→ 省的是重复 sweep 时间、与显存无关。Step 2（moe）本就只 tp1（EP 由 simulator 建模），不受影响；tp_stable 层由 writer 从 tp1 复制进各 `tp<N>/`，请求 1,8 时照样写进 tp8/，自洽。改 `profile-glm51-all.sh`/`profile-glm51-smoke.sh`/`H20_RUNBOOK.md`。未 push |
| 2026-06-26 | **profiler 加 `--block-size` flag（commit `36b5155`）+ GLM 脚本默认 `--dtype bfloat16` / `--block-size 64`（commit `8a0d290`）+ 新增挂机驱动 `profile-glm51-all.sh`** | H20 上跑 smoke 实测撞到 `No valid attention backend ... head_size=576, use_mla=True, use_sparse=True, dtype=torch.float16`。**两个独立根因**（都不是 block_size 报的、误导性强）：① **dtype 被解析成 float16** —— profiler `engine.py:95` 仅在显式传 `--dtype` 才设；GLM-5.1 config 用 transformers-5.x 的 `"dtype"` 键（**无 `torch_dtype`**），vLLM `dtype=auto` 找不到 `torch_dtype` → fallback float16；Hopper 的 FlashMLA/FlashMLA_Sparse 只吃 bf16/fp8 → 全 backend 拒（reasons 空）。修：脚本显式 `--dtype bfloat16`（compute/activation 精度；FP8 权重仍由 config `quantization_config` 驱动，文件夹名靠 `--variant fp8`，三者解耦）。② **block_size=16** —— Hopper FlashMLA/FlashMLA_Sparse 要求 64，而 `HOST_ENGINE_DEFAULTS["block_size"]=16` 写死、无 CLI 覆盖。修：加 `--block-size` flag（`__main__.py`/`config.py`/`engine.py` 三处贯通，`--help` + 静态断言已验证），未传保持 16（其它模型零影响），GLM 脚本默认 64。注：block_size 只影响合成 block table、不改 kernel 计时。`--block-size` 是模型无关能力 → Phase 4.2 可单独提上游 PR。两 commit 已 push fork、未 push origin |
| 2026-06-26 | **同步上游 `origin/main` 23 提交（merge `2ece922`，非 rebase）** | 集成策略选 merge：本分支长命、反复同步、按 Phase 4.2 拆干净小分支提上游（不整体提交），符合"长命分支 merge、提交前再切干净 topic 分支"主流实践（merge 只解一次冲突、不 force-push）。冲突=2 文件 7 hunk 全机械性。**核心适配**：上游 `bdfa6f7` per-instance 重构把 dtype/fp 流动改成 `inst_cfg[...]`，其 `"fp"=dtype_to_bits[dtype]`（fp8→8）**正是 `43b12c4` 修掉的单 fp bug**；解法=在 `_build_instance_runtime_configs` dict 加 `"weight_fp"` + 改 `"fp"=max(...,16)`，调用点按 use-site 分流（weight→weight_fp、comm/KV→fp），删除被上游 `_resolve_instance_dtype` 取代的全局 dtype-peek block。**验证**：bf16 smoke bit-exact（merged vs pre-merge `diff` 无差异）；`--dtype fp8` tp2 trace o_proj/down_proj weight 精确减半且 ALLREDUCE comm 不变（未合回 bug）；glm yaml 仍 load。**白拿上游收益**：prefix-cache/PD KV 记账修复、run 隔离输入路径、PIM 泛化；`c4edd0a` 独立修了我们的 power_model `:` 后缀 bug → 该改动可在 Phase 4.2 撤出不再单独提 PR。未 push |
| 2026-05-28 | 第一版不支持 MTP head | 复杂度高且对 baseline 性能数据非必需，可后续增量 |
| 2026-05-28 | profile dense layer 和 MoE layer 分两次跑（用 hf_overrides 控制 first_k_dense_replace） | 比改 profiler 假设"单层代表全模型"工作量小，且无副作用 |
| 2026-05-28 | 暂不在 5060Ti 上验证任何 GLM-5.1 相关改动 | 装不下 |
| 2026-06-03 | **`SHARD_FIELDS` 不动**，撤销 Phase 1.2 "加入 q_lora_rank / kv_lora_rank" 计划 | vLLM 0.21 源码 (`deepseek_v2.py:866-1068` + `mla.py:34-181`) 显示：`fused_qkv_a_proj` 的 LoRA 段 (`q_lora_rank=1536`, `kv_lora_rank=512`) 输出维度不沿 TP 切（输出固定 `[q_lora_rank, kv_lora_rank + qk_rope_head_dim]`，全 TP rank 复制）；Indexer 的 `wq_b` 是 `ReplicatedLinear`，`wk_weights_proj` 显式 `disable_tp=True`。glm5_h20 trace 显示 `nccl.all_reduce` 只在 `o_proj` / `down_proj` 之后出现，间接验证 LoRA 段没有 TP collective。沿用现有 4 个字段 |
| 2026-06-03 | v1 `glm_moe_dsa.yaml`：**DSA Indexer 合并到 `dense` catalog**（一条 `indexer: {vllm: Indexer, within: MultiHeadLatentAttentionWrapper}`），不单独建 indexer category | 改动量小，schema (`catalog.attention` 必须 exactly 1) 不允许把 indexer 塞到 attention 旁。Indexer 时间主要随 token 数线性增长 (`wq_b` + `wk_weights_proj` GEMM + per-token quant + topk)，一阶用 dense 1D 查表近似可接受；KV 长度依赖会在长上下文 decode 上偏低，留待 Phase 2.6 拆出 indexer category 时再补 |
| 2026-06-03 | v1 yaml：**4 个 RMSNorm 合并成 3 条 catalog 项**（`layernorm` 含 input/post，`mla_lora_layernorm` 含 q_a/kv_a，`final_layernorm`），借用 deepest-within-wins 规则 + `(vllm, within)` 唯一性约束 | DecoderLayer 下两个 RMSNorm 同类同父；MLA wrapper 下两个 RMSNorm 同类同父。schema 强制唯一所以无法分开，但合并是无损的（每层都各跑一次，合并测量是两次累加除以 invocation 数 ≈ 单次时间，sequence 走两次照样累加正确） |
| 2026-06-03 | `kv_b_proj` 加进 catalog 但不进 sequence | 否则 `q_b_proj` (within=Wrapper) 会把同类的 `kv_b_proj` 节点也吸走（Wrapper 在 kv_b_proj ancestor chain 里）。加 `kv_b_proj: {within: MLAAttention}` 借更深 ancestor 抢走它的 sample，让 `q_b_proj` 的 dense.csv 干净。`kv_b_proj` 自身的时间已经被 `attention` (MLAAttention) 整段吸收，sequence 不再走它避免双计 |
| 2026-06-03 | profile MoE 必须 CLI 加 `--hf-overrides '{"first_k_dense_replace": 0}'` | `HOST_ENGINE_DEFAULTS` 默认 `hf_overrides.num_hidden_layers=1`，GLM-5.1 配置 `first_k_dense_replace=3` 让第 0 层落在 dense MLP，导致 MoE category 拿不到样本。两次跑：(a) dense 跑默认，(b) MoE 跑 `--hf-overrides '{"first_k_dense_replace": 0}'` |
| 2026-06-03 | trace_generator MoE 检测加 `n_routed_experts` fallback | `_get_layer_function` 原只查 `num_local_experts` / `num_experts`，DeepSeek/GLM family 用 `n_routed_experts`，导致 MoE 路径永远不激活。修在 `trace_generator.py:1469-1474`。**等价地修复了所有 DeepSeek-V2/V3/GLM-MoE 模型的 simulator MoE 通路**，不只是 GLM-5.1 |
| 2026-06-03 | trace_generator power_model 加 `comm_type.split(':')[0]` 剥 dim 后缀 | `_emit_layer:965` 把 `"ALLREDUCE:1,0"` 整串传给 `power_model.total_ring_data`，后者只识别裸的 `allreduce/alltoall/allgather/reducescatter` 名，raise `Unknown collective`。所有 **DP 拓扑 + power_model** 之前从未走通，触发条件不仅 GLM-5.1，任何 MoE 模型在 2D involved_dim 配置下都会撞 |
| 2026-06-03 | v1 不实现 MoE allgather/reducescatter -> alltoall 切换 | vLLM 0.21 `moe.experts_collective_layout` 默认 `allgather_reducescatter`：dispatch=ALLGATHER + combine=REDUCESCATTER（不是 ALLTOALL）。trace_generator 当前 emit 已经匹配该默认。如果将来切到 `all2all` backend 再改 |
| 2026-06-04 | **`first_k_dense_replace` 逐层 dense/MoE 切换 ✅ 已实现** | 用户指出 GLM-5.1 前 `first_k_dense_replace` 层是 dense MLP、其余才 MoE。simulator 原本完全没处理（`is_moe` 全局布尔、所有层一刀切 MoE、block-copy 复制单一 block）。实现：① `_is_moe_layer(config, layer_num)`（`layer_num >= first_k_dense_replace` 且 `moe_layer_freq` 取模）；② `_emit_post_attn_layers` 逐层判断替换全局 `ctx.is_moe`；③ block-copy 主循环 + interleaved 中段改用 `_layer_segments()` 分段建块复制（dense 段 ×first_k + MoE 段 ×(num_layers-first_k)），`first_k=0`/`freq<=1` 退回单段快路径（零回归），`freq>1` 退回逐层；④ `memory_model.get_weight` 按段累加（heaviest-rank 上界，pp=1 精确）。dry-run：nl=2 fk=1→1+1、nl=4 fk=3→3+1、fk=0 全 moe，两条 emit 路径（block-copy / block_mode）输出一致 |
| 2026-06-04 | config_builder + memory_model MoE 检测补 `n_routed_experts` fallback ✅ 已修 | 原 `config_builder.py:35` 与 `memory_model.py:57` 的 `is_moe` 都只认 `num_local_experts`/`num_experts`，不认 DeepSeek/GLM 的 `n_routed_experts` → GLM 被当 dense（config_builder：ep_size 默认 1、ep-divides-experts 校验跳过；memory_model：256-expert 层错按 dense FFN 算权重，严重低估）。三处（trace_generator:1469 / config_builder:35,88 / memory_model:57）现一致加 `n_routed_experts` fallback，等价修复所有 DeepSeek-V2/V3/GLM-MoE 的 EP 默认、整除校验与权重核算 |
| 2026-06-04 | **撤销 Phase 2.7**：shared expert 不在 simulator 单独 emit | 源码 + trace 证实 shared expert 在 `FusedMoE.forward` 内部计算（`DeepseekV2MoE.forward` 仅一次 `self.experts(...)`；`moe_runner.py:273` `_moe_forward_shared` routed+shared 同一 forward；trace `moe_shared_experts` 嵌套于 `moe.fused_experts`）。profiler hook 整个 FusedMoE → moe.csv 已含 shared 时间。simulator 再 emit 会双计。更正了 R6 的错误结论。剩余仅 H20-time 校验：确认 profiler 强制路由 patch 未 bypass shared-expert kernel（次要风险：EP>1 时 shared 应按 rank 输入 token 数而非 routed local_tokens 缩放，moe.csv 在 ep=1 profile，留待 H20 数据回归评估） |
| 2026-06-12 | **H20_RUNBOOK 逐条代码审计：修 1 个致命流程缺陷（profiler 加 `--skip-moe`）+ 重写 runbook** | 对 runbook 每条命令对照实际 CLI/代码核实。**致命**：原 Step 1（dense 轮）会崩——GLM yaml 声明 `catalog.moe`，`categories_for()` 在 tp=1 无条件发射 moe shots，而 dense 轮单层模型无 FusedMoE → `single_moe_layer` RuntimeError，category 循环无异常捕获，tp=2/4/8 与 meta.yaml 全丢。修法：profiler 加 `--skip-moe`（`__main__.py`/`config.py`/`runner.py`/`profile.sh` 四处，flag 贯通已本机断言验证），dense 轮 SKIP_MOE=1、moe 轮去掉。附带确认：moe 轮的 resume 语义天然防 shared-expert 污染（dense shot key 已全覆盖、不再发射）。**其余 P0**：① Step 6 缺 `--max-num-seqs 256`（sim 默认 128 vs bench 256，调度不一致）；② 验收引用了不存在的 `catalog_coverage` 字段（改为 dense.csv 13 层名 diff 校验）；③ 日志写 /tmp 跨容器不可见（validate 读不到 sim log，全部改 `logs/`）；④ `--output` 绝对路径 bug 落档。**P1**：bench `--model` 应用本地 checkpoint 路径；VARIANT 理由更正（GLM `torch_dtype=null` → 默认落 `default/` 非 `bf16/`）；verify_moe_hook 是 bf16 路径、FP8 monolithic 守卫在正式 MoE 轮才触发；H20 具体规格（HBM3 4.0TB/s、NVLink 900GB/s 双向≈450 单向）；MTP 两侧都不跑的一致性说明。runbook 已全文重写（耗时/卡数总览表 + 失败模式速查表） |
| 2026-06-11 | **架构评审：修复 fp 三轨混淆（R10）+ moe_hook 0.21 失效（R11），两者均为 H20 前必修** | 评审代码核实：① `serving/__main__.py:197` 单一 `fp` 同时驱动权重/通信/KV —— GLM-5.1 真实三轨是 weight=FP8(1B)、activation=BF16(2B，trace `cross_device_reduce_1stage<__nv_bfloat16>` 直接证明)、MLA latent KV=BF16(2B)，任何 `--dtype` 取值都至少错两轨；② vLLM 0.21 `FusedMoE(PluggableLayer)` 无 `forward_native`（`forward` 委托 MoERunner），旧 hook 必 AttributeError，且 0.21 上从未被执行过（本机跑通的全是 dense 模型）。修复见 commit `43b12c4` / `7bdde56`，验证证据见 §6 2026-06-11 节。同批记录 R12（indexer O(kv_len) 缺失 → 第一轮验证避开长上下文）/ R13（indexer k-cache ~11% + modules_to_not_convert）/ R14（profile 边界 ≥ workload 分布、双长度 regime、vLLM 版本三方 pin 0.21） |
| 2026-06-04 | **Phase 2.8 重定义：`noaux_tc` 实现 → group-limited routing；GLM-5.1 无需** | 用户问"未用真实 gate 权重会否造成专家分布异常"。结论否（对 GLM-5.1）。理由链：(1) **模拟器只消费聚合负载**——`trace_generator.py:1081-1100` 只用 per-rank `(local_tokens, activated_experts)` 做 `_lookup_moe` + comm size + `max_rank_latency` barrier，从不关心"是哪个专家"；唯一相关的"异常"是 rank 间负载不均。(2) **noaux_tc 的设计目标就是专家均衡**（偏置项替代 aux-loss），方向与 `BALANCED` 的"均匀流量"假设（`gate_function.py:117-119`）一致。(3) **GLM-5.1 `n_group=1, topk_group=1`** → 无 group 结构，等于全局 top-8，正是 BALANCED 建模最准的场景。(4) **忠实 noaux_tc 打分不可实现**——sigmoid 打分 + 偏置 + top-k 依赖真实权重 × 真实 activation，模拟器两样皆无；造 score 会退化成均匀随机，反不如 BALANCED 贴合"均衡"真相。唯一**可实现**的部分是 group-limited routing（`n_group`/`topk_group` 是确定性结构，不依赖权重）：在 `route_ep` 加按组掩码 + EP-to-group 对齐，~半天工作量；但 GLM-5.1 `n_group=1` 下是 no-op，**只对 DeepSeek-V3 家族（`n_group=8, topk_group=4`）有意义**。故 Phase 2.8 对 GLM-5.1 降级 P3/无需，BALANCED 即物理合理近似；`routed_scaling_factor`/`norm_topk_prob`/sigmoid 只影响路由权重值不影响 token 计数，与延迟模型无关 |

---

## 9. 候选 backlog（未排期，暂不实现）

记录已评估、有明确价值但当前不动代码的候选特性。执行细节（逐模块触点、改动量）见
`~/.claude/plans/profiler-enumerated-twilight.md` 对应节。

| # | 候选 | 动机 | 范围概要 | 量级 | 状态 |
|---|---|---|---|---|---|
| B1 | **MoE TP 并行（Megatron 式，无 EP）** | 现框架 MoE 设计是 **EP 中心**（专家按数量切到各 rank + ALLTOALL/ALLGATHER 路由）。对**小专家数**模型（如 Mixtral 8 专家）EP 切不动、TP 切 `moe_intermediate` 更均衡，需要一条 TP-MoE 路径才能建模 | 跨 5 个模块、边界清晰：① **profiler**（最实在）moe.csv 从"tp 无关单表"变成"逐 tp 切 intermediate 的多表"，**且 Step 2 不再恒 tp1、要并回 `1,8` sweep 分开测**；② trace_generator `_emit_moe_block` 把 ALLGATHER/REDUCESCATTER(ep_dim) 换成单次 ALLREDUCE(tp_dim) + 全量 token 查表；③ memory_model expert sharding `parallel=ep`→`parallel=tp`（近一行）；④ config_builder 放开 ep 整除校验、MoE collective 挂 tp 维；⑤ gate_function `route_ep` 旁路加简化分支。**dense/attention/KV/scheduler/router 全不动** | ~2-3 天 + **重新 profile moe**；建议用 `moe_parallel: ep\|tp` 开关并存、不动现有 EP 路径 | 📋 候选，暂不实现（2026-06-27 记录） |
