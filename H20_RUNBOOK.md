# H20 操作手册 — GLM-5.1 真实 profile + 验证

> 目的:在 H20 8 卡上拿到 GLM-5.1 FP8 的真实 profile 数据,替换本机占位 CSV,
> 然后用真实 vLLM bench 验证 simulator 误差 < 20%。本机已完成全部不依赖 GPU 的
> 代码/链路工作(见 `GLM5_1_SUPPORT_PLAN.md` 第 6 节),H20 只需"喂真实数据 + 校准"。
>
> 全程在仓库根(`LLMServingSim/`)执行。命令保持英文。

---

## ⚠️ 开跑前两个必须确认的前置项(否则结果无意义)

### 前置 1 — 还原 simulator 的真实层数(当前是 smoke 缩水值!)

`configs/model/zai-org/GLM-5.1.json` 现在是 **1 层 smoke 配置**:
```
"num_hidden_layers": 1,      ← 真实 GLM-5.1 是 ~78-92 层
"first_k_dense_replace": 1,  ← 真实值是 3
"num_nextn_predict_layers": 0
```
**simulator 靠这两个字段决定 replay 多少层(几层 dense + 几层 MoE)。** 1 层配置下
simulator 只模拟单层,bench-vs-sim 对比毫无意义。

**操作**:从 H20 上真实 GLM-5.1 checkpoint 的 `config.json` 读出真实
`num_hidden_layers` / `first_k_dense_replace` / `num_nextn_predict_layers`,
改回 `configs/model/zai-org/GLM-5.1.json`。其余每层 shape 字段(hidden_size /
n_routed_experts / kv_lora_rank / q_lora_rank / index_* 等)已正确,不用动。

> 注意:这**只影响 simulator**。**profiler 不受影响**——profiler 按设计永远把
> `num_hidden_layers` 压到 1 跑单层代表(`HOST_ENGINE_DEFAULTS`),无论 config 写几层。

### 前置 2 — 确认 vLLM 版本支持 `GlmMoeDsaForCausalLM`

`scripts/docker-vllm.sh` 默认镜像是 **`vllm/vllm-openai:v0.19.0`**,大概率
**不认识 `GlmMoeDsaForCausalLM`**(该架构在 0.20.1+ 才进 vLLM;trace 用的 0.20.1,
源码分析用的 0.21)。

**操作**:把 `scripts/docker-vllm.sh` 的镜像 tag 换成 H20 上可用、且能
`from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM` 的版本
(建议 0.20.1 或 0.21,和你 trace 那台一致最稳)。换完先验一行:
```bash
python3 -c "from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM; print('ok')"
```
> 副作用提醒:profiler 的 MoE hook 是 `patch FusedMoE.forward_native` **强制专家路由**,
> 方法名 version-specific(CLAUDE.md 注)。若换了 vLLM 版本,Step 2 跑 MoE 那次若报
> hook 找不到方法,需到 `profiler/core/hooks/moe_hook.py` 对齐方法名。

---

## Step 0 — 环境(在 H20 host)

```bash
cd <repo>/LLMServingSim
export HF_TOKEN=<你的 token>          # GLM-5.1 是 gated,profiler/bench 自动下载 config 要用
# 在 scripts/docker-vllm.sh 里确认/写入 HF_TOKEN 和正确镜像 tag(前置 2)
./scripts/docker-vllm.sh             # 进容器,cwd=/workspace,仓库根已挂载
```
后面 Step 1-3(profiler)和 Step 5(bench run)都在这个 vLLM 容器里跑。
Step 4/6/7(simulator + validate)在 **simulator 容器**(`scripts/docker-sim.sh`)
或你本机原生编译好的环境里跑——别和 vLLM 容器混。

> **profiler 只要 1 张卡**:每个 TP degree 都在单 GPU 上 emulate(TP 用 hf_overrides
> 切 shape,collective 交给 ASTRA-Sim)。**8 卡只有 Step 5 真实 bench 才需要。**

---

## Step 1 — profile dense / per_sequence / attention(Phase 1.3)

编辑 `profiler/profile.sh`,改这几个变量:
```bash
MODEL="zai-org/GLM-5.1"
HARDWARE="H20"                 # 必须和 cluster config 的 hardware 字段一致
TP_DEGREES="1,2,4,8"
VARIANT="fp8"                  # ★ 关键:见下方"变体命名"。不设会默认按 bf16 命名
MAX_NUM_SEQS=256              # 决定 attention/skew 的 n 上界(见 CLAUDE.md 可行性边界)
MAX_NUM_BATCHED_TOKENS=2048
# 第一轮先 SKIP_SKEW=1 跑快的;skew 单独第二轮(每 TP 1-2 小时)
SKIP_SKEW=1
```
跑:
```bash
VLLM_LOGGING_LEVEL=DEBUG ./profiler/profile.sh 2>&1 | tee /tmp/glm_profile_dense.log
```

**变体命名(VARIANT="fp8")为什么**:GLM-5.1 是 FP8 *checkpoint*,但 config
`dtype=bfloat16` 是*计算*精度。profiler 默认按 dtype 命名 → 会落到 `bf16/` 文件夹,
名不副实。手动 `VARIANT="fp8"` 让数据落到 `perf/H20/zai-org/GLM-5.1/fp8/`,和
simulator 端 `--dtype fp8`(Step 6)对齐。**不要给 profiler 传 `--dtype fp8`**——
vLLM 的 model dtype 不接受 fp8,量化由 checkpoint 的 `quantization_config` 自动处理。

**R1 提醒**:FP8 + MLA + sparse indexer 的首次 JIT 编译可能 5-30 分钟,DEBUG 日志能看到
进度,别误判卡死。

产出:`profiler/perf/H20/zai-org/GLM-5.1/fp8/tp{1,2,4,8}/{dense,per_sequence,attention}.csv` + `meta.yaml`

---

## Step 2 — profile MoE(必须单独一轮,Phase 1.4/1.5)

config `first_k_dense_replace` 让 profiler 默认压成的"第 0 层"落在 **dense MLP**,
`moe.csv` 会是空的。强制第 0 层变 MoE:
```bash
# 在 profile.sh 里临时加一行(其余变量同 Step 1):
HF_OVERRIDES='{"first_k_dense_replace":0}'
# 用 resume 模式,只补 moe.csv,不重跑 dense
./profiler/profile.sh 2>&1 | tee /tmp/glm_profile_moe.log
```
> profiler 是 resume 语义(已有 CSV 的 key 跳过),所以这一轮主要产出 `moe.csv`。
> moe.csv 在 tp=1 profile(MoE 永远 tp=1 测,EP 由 simulator 建模)。

**Phase 1.4 校验点(重要)**:确认强制路由 patch **没 bypass shared expert**。
GLM-5.1 `n_shared_experts=1`,shared expert 在 `FusedMoE.forward` 内部算,moe.csv
应已含其时间。跑完抽查:moe.csv 的 time_us 不应明显小于"routed 8 experts + 1 shared"
的预期(若疑似漏了 shared,查 `moe_hook.py` 的强制路由是否绕过了 shared 分支)。

---

## Step 3 — profile skew(可选,Phase 1.3 第二轮,每 TP 1-2h)

attention 的 skew alpha 拟合。第一轮验证可先跳过(simulator 用 pooled 常数 alpha 兜底)。
要做就:
```bash
# profile.sh 里去掉 SKIP_SKEW,或单独 ONLY_SKEW=1 只刷 skew
ONLY_SKEW=1 ./profiler/profile.sh 2>&1 | tee /tmp/glm_profile_skew.log
```
产出每个 tp 的 `skew.csv` + `skew_fit.csv`。

**Step 1-3 完成后验收**:
```bash
ls profiler/perf/H20/zai-org/GLM-5.1/fp8/tp1/
# 应有 dense.csv per_sequence.csv attention.csv moe.csv (+ skew*.csv 若做了)
head profiler/perf/H20/zai-org/GLM-5.1/fp8/tp1/moe.csv   # 确认非空、time_us 合理
grep -A20 catalog_coverage profiler/perf/H20/zai-org/GLM-5.1/fp8/meta.yaml
# catalog_coverage 里 13 dense + 2 per_seq + 1 attention + 1 moe 应全 hooked_at_least_once: true
```

---

## Step 4 — 填 H20 cluster config 的真实硬件参数

`configs/cluster/h20_8_glm5_1_fp8.json` 目前 bw/latency/mem_bw 是**占位**:
```
"link_bw": 900,        ← H20 NVLink 实际带宽 GB/s
"link_latency": 0,     ← 实际 ns
"npu_mem.mem_bw": 4000,← H20 HBM 实际带宽 GB/s
"cpu_mem.mem_bw": 512  ← 实际
```
用 H20 真实规格填(NVLink 带宽、HBM 带宽)。`mem_size: 96`(H20 显存)一般正确。
tp_size/ep_size/num_npus=8 已对。

---

## Step 5 — 真实 vLLM bench(需要 8 卡,Phase 3.1)

在 **vLLM 容器**里,跑真实 GLM-5.1 端到端,作为 ground truth:
```bash
python -m bench run \
  --model zai-org/GLM-5.1 \
  --dataset workloads/<你的数据集>.jsonl \
  --output-dir bench/results/glm51_h20 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-seqs 256 \
  --max-num-batched-tokens 2048 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  2>&1 | tee /tmp/glm_bench.log
```
产出:`bench/results/glm51_h20/{meta.json, requests.jsonl, timeseries.csv}`。
> dataset 用真实 workload(ShareGPT 派生等)。`--num-reqs N` 可限制条数先小跑。

---

## Step 6 — simulator 跑同一 workload(Phase 3.2 前半)

在 **simulator 容器/原生环境**里(不是 vLLM 容器):
```bash
python -m serving \
  --cluster-config configs/cluster/h20_8_glm5_1_fp8.json \
  --dataset workloads/<同 Step 5 的数据集>.jsonl \
  --dtype fp8 \
  --output outputs/glm51_h20_sim.csv \
  2>&1 | tee /tmp/glm_sim.log
```
**`--dtype fp8` 关键**:① 让 variant 解析到 `fp8/`(匹配 Step 1 的 VARIANT);
② memory_model 按 1 byte/权重算(fp8 checkpoint 正确显存),不设会按 2 byte 高估 2×。

跑前确认前置 1 已还原真实层数,否则 simulator 只算 1 层。

---

## Step 7 — 对比验证(Phase 3.2 后半)

```bash
python -m bench validate \
  --bench-dir bench/results/glm51_h20 \
  --sim-csv outputs/glm51_h20_sim.csv \
  --sim-log /tmp/glm_sim.log \
  --output-subdir validation_glm51
```
看 throughput / TTFT / TPOT 误差。**目标 < 20%**。

---

## Step 8 — 误差 > 20% 的调试方向(Phase 3.3)

按影响大小排查:
1. **attention cost(Phase 2.5/2.6)**:MLA 压缩 latent KV + sparse top-2048,当前
   simulator 直接复用 4D attention 网格。若长上下文 decode 误差大,考虑给
   `_lookup_attention*` 加维度,或对超过 `index_topk` 的 kv 做平坦化。
2. **MoE cost**:moe.csv 在 ep=1 测,simulator 按 per-rank local_tokens 查表 + EP scaling。
   EP>1 时 shared expert 的 token 缩放(决策日志 2026-06-04 留的次要风险)在这里回归。
3. **layer 配比**:确认前置 1 的 first_k_dense_replace 真实值正确(dense/MoE 层数比对延迟影响大)。
4. **skew**:若跳过了 Step 3,decode 批 kv 长度不均会有系统性偏差,补跑 skew。

---

## Step 9 — 跑通后回写计划文档(标准规则)

两份都要更新:
- `GLM5_1_SUPPORT_PLAN.md`:第 6 节标 Phase 1.3-1.7 / 3.1-3.3 ✅ + 误差数字;
  任务表对应行打勾;Phase 2.5/2.6 据真实数据定结论;删第 6 节"阻塞中"。
- `~/.claude/plans/profiler-enumerated-twilight.md`:附录 F.3 的"必须等 H20"项标完成,
  附录 G 之后补一段真实数据验证结果。

跑挂就只回写决策日志(失败模式 + 已试 fallback),别留半截状态。

---

## 一页速查(命令顺序)

```bash
# 前置:改 configs/model/zai-org/GLM-5.1.json 真实层数 + 换 vLLM 镜像 tag
./scripts/docker-vllm.sh                                    # 进 vLLM 容器
# Step1 dense/attn:  profile.sh 设 MODEL/HARDWARE=H20/VARIANT=fp8/SKIP_SKEW=1 → ./profiler/profile.sh
# Step2 moe:         profile.sh 加 HF_OVERRIDES='{"first_k_dense_replace":0}' → ./profiler/profile.sh
# Step3 skew(可选): ONLY_SKEW=1 ./profiler/profile.sh
# Step4: 填 configs/cluster/h20_8_glm5_1_fp8.json 真实 bw/latency
python -m bench run --model zai-org/GLM-5.1 --dataset <ds> --output-dir bench/results/glm51_h20 \
  --tensor-parallel-size 8 --enable-expert-parallel --dtype bfloat16   # Step5(8卡)
# 切 simulator 环境:
python -m serving --cluster-config configs/cluster/h20_8_glm5_1_fp8.json --dataset <ds> \
  --dtype fp8 --output outputs/glm51_h20_sim.csv                        # Step6
python -m bench validate --bench-dir bench/results/glm51_h20 \
  --sim-csv outputs/glm51_h20_sim.csv --sim-log /tmp/glm_sim.log        # Step7
```
