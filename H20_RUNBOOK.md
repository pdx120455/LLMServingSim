# H20 操作手册 — GLM-5.1 真实 profile + 验证

> 目的:在 H20 8 卡上拿到 GLM-5.1 FP8 的真实 profile 数据,替换本机占位 CSV,
> 然后用真实 vLLM bench 验证 simulator 误差 < 20%。本机已完成全部不依赖 GPU 的
> 代码/链路工作(见 `GLM5_1_SUPPORT_PLAN.md` 第 6 节),H20 只需"喂真实数据 + 校准"。
>
> 全程在仓库根(`LLMServingSim/`)执行。命令保持英文。

## 全局约定(每条命令都遵守)

1. **所有日志 tee 到仓库内 `logs/`,不要写 `/tmp`**。vLLM 容器和 simulator
   容器只共享仓库挂载;Step 7 的 `bench validate --sim-log` 要在 vLLM 容器里
   读 Step 6(simulator 环境)写的日志,`/tmp` 跨容器不可见。
2. **simulator 的 `--output` 只能用仓库相对路径**(如 `outputs/x.csv`)。
   simulator 启动即 chdir 到 `astra-sim/` 并给 output 加 `../` 前缀,绝对路径
   会变成 `..//tmp/...` 而 FileNotFoundError(5060Ti 上实测踩过)。
3. 开跑前一次性建目录:`mkdir -p outputs logs`。

## 总览:每步耗时 / 卡数 / 环境

| Step | 内容 | 卡数 | 环境 | 预计耗时 |
|---|---|---|---|---|
| 0 | 环境 + 镜像 | 0 | host | ~30min(拉镜像) |
| 1 | profile dense/per_seq/attention × TP 1/2/4/8 | 1 | vLLM 容器 | 每 TP 0.5-1.5h(首次 JIT 另加 5-30min) |
| 2 | hook 自检 + profile moe(tp=1 一轮) | 1 | vLLM 容器 | 自检 ~1min;moe 轮 ~0.5h |
| 3 | profile skew(可选) | 1 | vLLM 容器 | 每 TP 1-2h |
| 4 | 填 cluster config 硬件参数 | 0 | 编辑器 | ~10min |
| 5 | 真实 vLLM bench(ground truth) | **8** | vLLM 容器 | 模型加载 10-30min + 跑 workload |
| 6 | simulator 跑同一 workload | 0(CPU) | simulator 环境 | 取决于 workload,通常 < bench |
| 7 | bench vs sim 对比 | 0 | vLLM 容器 | ~1min |

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

> 注意 1:这**只影响 simulator**。**profiler 不受影响**——profiler 按设计永远把
> `num_hidden_layers` 压到 1 跑单层代表(`HOST_ENGINE_DEFAULTS`),无论 config 写几层。
>
> 注意 2:simulator **不消费** `num_nextn_predict_layers`(MTP 不建模),还原它只为
> 配置文件忠实。对应地,Step 5 的 bench **不要开 speculative decoding**(vLLM 默认
> 就不开)——两边都不跑 MTP,保持一致。

### 前置 2 — 确认 vLLM 版本支持 `GlmMoeDsaForCausalLM`

`scripts/docker-vllm.sh` 默认镜像是 **`vllm/vllm-openai:v0.19.0`**,大概率
**不认识 `GlmMoeDsaForCausalLM`**(该架构在 0.20.1+ 才进 vLLM;trace 用的 0.20.1,
源码分析用的 0.21)。

**操作**:把 `scripts/docker-vllm.sh` 的镜像 tag 换成 **vLLM 0.21**(必须 pin 0.21,
不要用 0.20.1 —— 架构 yaml 的类名按 0.21 源码写定,profiler 的 MoE hook 也按 0.21
MoERunner API 重写并验证;0.20.1 可能 catalog 不命中)。换完**在容器内**验一行:
```bash
python3 -c "from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM; print('ok')"
```
> MoE hook 已适配 0.21(2026-06-11,commit `7bdde56`):双入口 patch + monolithic
> quant method 守卫。Step 2 的自检和守卫语义见 Step 2 内说明。

### 前置 3 — simulator 代码版本

Step 6 依赖两个本地修复,确认 simulator 侧代码 **≥ commit `84846a1`**
(含 `43b12c4` dtype 拆分 + `7bdde56` hook 重写):
```bash
git log --oneline -5   # 应能看到 43b12c4 / 7bdde56 / 84846a1
```
profiler 侧还需 `--skip-moe` 支持(本次 runbook 修订所在 commit,Step 1 必用)。

---

## Step 0 — 环境(在 H20 host)

```bash
cd <repo>/LLMServingSim
mkdir -p outputs logs
export HF_TOKEN=<你的 token>          # GLM-5.1 是 gated;本地已有 checkpoint 时只有 tokenizer 下载用得到
# 在 scripts/docker-vllm.sh 里确认/写入 HF_TOKEN 和正确镜像 tag(前置 2)
./scripts/docker-vllm.sh             # 进容器,cwd=/workspace,仓库根已挂载
```
后面 Step 1-3(profiler)和 Step 5(bench run)都在这个 vLLM 容器里跑。
Step 6(simulator)在 **simulator 容器**(`scripts/docker-sim.sh`)或原生编译好的
环境里跑;Step 7(validate)回 **vLLM 容器**(要 matplotlib + bench 包)。
别把两个环境混用。

> **profiler 只要 1 张卡**:每个 TP degree 都在单 GPU 上 emulate(TP 用 hf_overrides
> 切 shape,collective 交给 ASTRA-Sim)。**8 卡只有 Step 5 真实 bench 才需要。**

---

## Step 1 — profile dense / per_sequence / attention(Phase 1.3)

> **先验通路(可选,强烈建议第一次跑)**:`./profiler/profile-glm51-smoke.sh` 用极小 sweep grid(64/8/512、iter=1、SKIP_SKEW)自动跑完 dense+MoE 两轮、几分钟内出 CSV,只为确认 profiler 在 H20 上不卡死、引擎能 boot/teardown、hook 能触发。数据不准,验完正式全量前用 `FORCE=1 ./profiler/profile.sh` 清掉这些粗点再重测。

编辑 `profiler/profile.sh`,改这几个变量:
```bash
MODEL="zai-org/GLM-5.1"
HARDWARE="H20"                 # 必须和 cluster config 的 hardware 字段一致
TP_DEGREES="1,2,4,8"
VARIANT="fp8"                  # ★ 关键:见下方"变体命名"
MAX_NUM_SEQS=256              # 决定 attention/skew 的 n 上界(见 CLAUDE.md 可行性边界)
MAX_NUM_BATCHED_TOKENS=2048
SKIP_MOE=1                    # ★ 必须!不加这轮会崩,见下
# 第一轮先 SKIP_SKEW=1 跑快的;skew 单独第二轮(每 TP 1-2 小时)
SKIP_SKEW=1
```
跑:
```bash
./profiler/profile.sh 2>&1 | tee logs/glm_profile_dense.log
```

**SKIP_MOE=1 为什么必须**:GLM yaml 声明了 `catalog.moe`,profiler 在 tp=1 会无条件
发射 moe shots;但本轮 `first_k_dense_replace≥1` 让单层测试模型是 **dense** 层,
模型里没有 FusedMoE → `single_moe_layer` raise RuntimeError,**整个 sweep 中途崩、
tp=2/4/8 和 meta.yaml 全丢**。moe 在 Step 2 单独一轮补。

**变体命名(VARIANT="fp8")为什么**:GLM-5.1 是 FP8 *checkpoint*,计算精度是 bf16。
profiler 默认按 weight dtype 给 variant 文件夹命名,而 GLM-5.1 config 的
`torch_dtype` 是 **null** → 不设 VARIANT 会落到 `default/` 文件夹(名不副实且
simulator 端解析不到)。手动 `VARIANT="fp8"` 让数据落到
`perf/H20/zai-org/GLM-5.1/fp8/`,和 simulator 端 `--dtype fp8`(Step 6)对齐。
**不要给 profiler 传 `--dtype fp8`**——vLLM 的 model dtype 不接受 fp8,量化由
checkpoint 的 `quantization_config` 自动处理。

**R1 提醒**:FP8 + MLA + sparse indexer 的首次 JIT 编译可能 5-30 分钟,卡住时
`--verbose`(profile.sh 里 `VERBOSITY="--verbose"`)能看到进度,别误判卡死。
JIT 产物缓存在 `~/.cache`,容器重启后会重编;长时间作业尽量别销毁容器。

产出:`profiler/perf/H20/zai-org/GLM-5.1/fp8/tp{1,2,4,8}/{dense,per_sequence,attention}.csv`

---

## Step 2 — profile MoE(必须单独一轮,Phase 1.4/1.5)

**先跑 hook 自检**(单卡,~1 分钟,验证强制专家路由在当前 vLLM 版本下真实生效):
```bash
python -m profiler.core.hooks.verify_moe_hook 2>&1 | tee logs/glm_moe_hook_verify.log
# 期望输出:forced distinct experts == 目标值 + "MOE HOOK VERIFIED"
```
> **自检的边界**:verify 脚本用的是 bf16 非量化小 FusedMoE,验证的是路由锻造
> 机制本身。GLM-5.1 真实 profile 走 **FP8 量化路径**,若 vLLM 给它挑了
> monolithic 后端(FlashInfer/TRT-LLM 融合 MoE,路由在 kernel 内部),hook 的
> 守卫会在**正式 MoE 轮**报
> `monolithic quant method ... forced expert routing cannot apply` —— 这是
> 特性不是 bug:它阻止产出无效的 moe.csv。届时设
> `VLLM_USE_FLASHINFER_MOE_FP8=0`(或报错信息里点名的后端对应 env)后重跑,
> 且**该 env 要带满整个 MoE 轮**。

然后改 `profile.sh` 再跑一轮(其余变量同 Step 1):
```bash
SKIP_MOE=                      # 去掉(或注释),本轮就是要 moe
TP_DEGREES="1"                 # moe 永远只在 tp=1 测(EP 由 simulator 建模),省 3 次引擎启动
HF_OVERRIDES='{"first_k_dense_replace":0}'   # 强制第 0 层变 MoE,否则 moe.csv 为空
SKIP_SKEW=1
```
```bash
./profiler/profile.sh 2>&1 | tee logs/glm_profile_moe.log
```
> **resume 语义保证本轮安全**:dense/per_sequence/attention 的 shot key 在 Step 1
> 已全覆盖,本轮全部跳过不发射 → 不会被 MoE 层里的 shared expert(同为
> `DeepseekV2MLP`,会匹配 `gate_up_proj` 等 dense 条目)写入错误 shape 的数据。
> 本轮净产出就是 `tp1/moe.csv`。
>
> **meta.yaml 由最后一轮覆盖**:`meta.yaml` 的 engine 参数记录的是最后跑的那轮。
> 各轮 MAX_NUM_SEQS / MAX_NUM_BATCHED_TOKENS 保持一致(都是 256/2048)即可,
> simulator 运行时边界告警以此为基准。

**Phase 1.4 校验点(重要)**:确认强制路由 patch **没 bypass shared expert**。
GLM-5.1 `n_shared_experts=1`,shared expert 在 `FusedMoE.forward` 内部算,moe.csv
应已含其时间。跑完抽查:moe.csv 的 time_us 不应明显小于"routed 8 experts + 1 shared"
的预期(若疑似漏了 shared,查 `moe_hook.py` 的强制路由是否绕过了 shared 分支)。

---

## Step 3 — profile skew(可选,Phase 1.3 第二轮,每 TP 1-2h)

attention 的 skew alpha 拟合。第一轮验证可先跳过(simulator 用 pooled 常数 alpha 兜底)。
要做就改 `profile.sh`(TP_DEGREES 还原 `"1,2,4,8"`,HF_OVERRIDES 清掉):
```bash
# 去掉 SKIP_SKEW,加 ONLY_SKEW=1 只刷 skew(其他 CSV 不动)
ONLY_SKEW=1 ./profiler/profile.sh 2>&1 | tee logs/glm_profile_skew.log
```
产出每个 tp 的 `skew.csv` + `skew_fit.csv`。

**Step 1-3 完成后验收**(注意:profiler 没有 catalog_coverage 之类的汇总字段,
靠 CSV 内容直接验):
```bash
P=profiler/perf/H20/zai-org/GLM-5.1/fp8
ls $P/tp1/   # 应有 dense.csv per_sequence.csv attention.csv moe.csv (+ skew*.csv 若做了)

# dense 层名应恰好是 yaml 的 13 个(diff 无输出 = 通过):
awk -F, 'NR>1{print $1}' $P/tp1/dense.csv | sort -u > /tmp/got.txt
printf '%s\n' act_fn down_proj embedding final_layernorm fused_qkv_a_proj \
  gate_up_proj indexer kv_b_proj layernorm mla_lora_layernorm o_proj \
  q_b_proj rotary_emb | sort > /tmp/want.txt
diff /tmp/got.txt /tmp/want.txt && echo "dense layers OK"

# per_sequence 应恰好 lm_head + sampler:
awk -F, 'NR>1{print $1}' $P/tp1/per_sequence.csv | sort -u   # → lm_head, sampler

# attention / moe 非空且数值合理:
wc -l $P/tp1/attention.csv $P/tp1/moe.csv
head $P/tp1/moe.csv          # time_us 应随 tokens / activated_experts 单调增长
grep -E "max_num_seqs|max_num_batched_tokens" $P/meta.yaml   # 应为 256 / 2048

# tp2/4/8 同样有 dense/per_sequence/attention(tp_stable 层由 tp1 复制):
ls $P/tp2 $P/tp4 $P/tp8
```
最终兜底验收在 Step 6:simulator 加载该 profile 后,日志里**不应出现**
"missing from the profile" 一次性警告(出现 = 某 sequence 层没采到数据)。

---

## Step 4 — 填 H20 cluster config 的真实硬件参数

`configs/cluster/h20_8_glm5_1_fp8.json` 目前 bw/latency/mem_bw 是**占位**。
H20 公开规格:HBM3 96GB、显存带宽 **4.0 TB/s**、NVLink **900 GB/s(双向聚合)**。

```
"link_bw": 450,        ← NVLink 900 GB/s 是双向聚合;按单向 ≈450 GB/s 填。
                         若后续 Step 8 校准发现 ASTRA-Sim 该字段语义是双向,再改回 900
"link_latency": 500,   ← NVLink 量级 ~500ns;有条件用 nccl-tests 实测更好
"npu_mem.mem_bw": 4000,← H20 HBM3 4.0 TB/s = 4000 GB/s(占位值恰好正确)
"cpu_mem.mem_bw": 512  ← host DDR 实际值(8 通道 DDR5-4800 ≈ 300-400;按机器实情填)
```
`mem_size: 96`(H20 显存)正确。tp_size/ep_size/num_npus=8 已对。

**建议校准**(可选,30min):在 H20 上跑一次 `nccl-tests` 的 `all_reduce_perf
-b 1M -e 1G`,取 busbw 稳定段,和 ASTRA-Sim 用同 size 的 ALLREDUCE 时间对一下,
误差大就调 `link_bw`。这一步也可以留到 Step 8 误差排查时再做。

---

## Step 5 — 真实 vLLM bench(需要 8 卡,Phase 3.1)

在 **vLLM 容器**里,跑真实 GLM-5.1 端到端,作为 ground truth。

**workload 选择(重要,R12/R14)**:
- 必须是 **flat 格式** jsonl(每行 `input_toks`/`output_toks`);bench 会跳过
  agentic 行(`sub_requests`),若数据集混有 agentic 行,sim 侧也要保证同样跳过
  (最稳妥:首轮用纯 flat 数据集)。
- 首轮验证**避开长上下文**:`input_toks + output_toks` 尽量 ≤ 16384
  (= profile 的 ATTENTION_MAX_KV 上界;超界 lookup 线性外推)。且 DSA indexer
  的 O(kv_len) 成本未建模(R12),长上下文误差会系统性偏大——那是 Step 8/Phase 2.5
  的事,别混进首轮。
- **离线生成数据集**(内网无法用 ShareGPT generator):
  `python3 workloads/generators/synthetic_glm51.py --num-reqs 64 --sps 10 --input-min 128 --input-max 1024 --output-min 64 --output-max 512 --output workloads/glm51_synth_64.jsonl`
  —— 纯标准库、不联网、无需 tokenizer,从 config 读 `vocab_size` 生成合法随机 id(延迟只取决于 token 数,内容无关)。`--seq-guard`(默认 4096)守住长度边界。**Step 5 `bench run` 与 Step 6 simulator 用同一份数据集 + 同一个 `--num-reqs`**。注:随机 id ⇒ prefix-cache 命中≈0(两侧一致、可比,但不反映真实前缀复用)。

```bash
# --model 用 H20 本地 checkpoint 目录(传 HF id 会现场下数百 GB!):
MODEL_DIR=/path/to/GLM-5.1     # 容器内可见的真实权重目录
python -m bench run \
  --model "$MODEL_DIR" \
  --dataset workloads/<你的数据集>.jsonl \
  --output-dir bench/results/glm51_h20 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-seqs 256 \
  --max-num-batched-tokens 2048 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --num-reqs 32 \
  2>&1 | tee logs/glm_bench_smoke.log
```
先 `--num-reqs 32` 小跑确认能端到端出结果(模型加载 + 首批请求 ≈ 15-40min),
再去掉(或调大)`--num-reqs` 跑全量,`--output-dir` 换 `bench/results/glm51_h20_full`、
log 换 `logs/glm_bench.log`。

产出:`bench/results/<run>/{meta.json, requests.jsonl, timeseries.csv}`。

---

## Step 6 — simulator 跑同一 workload(Phase 3.2 前半)

在 **simulator 容器/原生环境**里(不是 vLLM 容器):
```bash
python -m serving \
  --cluster-config configs/cluster/h20_8_glm5_1_fp8.json \
  --dataset workloads/<同 Step 5 的数据集>.jsonl \
  --dtype fp8 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 2048 \
  --num-reqs 32 \
  --output outputs/glm51_h20_sim.csv \
  2>&1 | tee logs/glm_sim.log
```
**和 Step 5 必须逐项对齐**:同一 dataset、同一 `--num-reqs`、
`--max-num-seqs 256`(simulator 默认是 128,不显式传就和 bench 不一致,误差虚高)、
`--max-num-batched-tokens 2048`。

**`--dtype fp8` 关键**:① 让 variant 解析到 `fp8/`(匹配 Step 1 的 VARIANT);
② memory_model 按 1 byte/权重算(fp8 checkpoint 正确显存),不设会按 2 byte 高估 2×。
> 2026-06-11 起 `--dtype fp8` 语义已修正(commit `43b12c4`):只影响**权重**精度;
> 通信量(ALLREDUCE/MoE dispatch)和 KV cache 自动保持 bf16(2 byte),与真实
> vLLM 行为一致(前置 3 已确认代码版本)。

跑前 / 跑后检查:
- 跑前确认前置 1 已还原真实层数,否则 simulator 只算 1 层。
- 跑后 `grep -i "missing from the profile" logs/glm_sim.log` 应无输出
  (有 = profile CSV 缺层,回查 Step 1-3 验收)。
- `grep -i "exceed" logs/glm_sim.log` 查运行时是否超 profile 边界
  (一次性告警,出现说明 workload 超出 256/2048 采样范围,lookup 在外推)。

---

## Step 7 — 对比验证(Phase 3.2 后半)

回 **vLLM 容器**(validate 要 matplotlib;所有输入都在仓库挂载内,跨容器可见):
```bash
python -m bench validate \
  --bench-dir bench/results/glm51_h20 \
  --sim-csv outputs/glm51_h20_sim.csv \
  --sim-log logs/glm_sim.log \
  --output-subdir validation_glm51
```
看 throughput / TTFT / TPOT 误差。**目标 < 20%**。
产出落在 `bench/results/glm51_h20/validation_glm51/`。

---

## Step 8 — 误差 > 20% 的调试方向(Phase 3.3)

按影响大小排查:
1. **attention cost(Phase 2.5/2.6)**:MLA 压缩 latent KV + sparse top-2048,当前
   simulator 直接复用 4D attention 网格。若长上下文 decode 误差大,考虑给
   `_lookup_attention*` 加维度,或对超过 `index_topk` 的 kv 做平坦化。
2. **MoE cost**:moe.csv 在 ep=1 测,simulator 按 per-rank local_tokens 查表 + EP scaling。
   EP>1 时 shared expert 的 token 缩放(决策日志 2026-06-04 留的次要风险)在这里回归。
3. **网络参数**:`link_bw`/`link_latency` 占位值不准时,collective 时间整体偏移。
   用 nccl-tests `all_reduce_perf` 实测 busbw,对照 ASTRA-Sim 同 size ALLREDUCE
   校准(Step 4 的可选项,这时变必选)。
4. **layer 配比**:确认前置 1 的 first_k_dense_replace 真实值正确(dense/MoE 层数比对延迟影响大)。
5. **skew**:若跳过了 Step 3,decode 批 kv 长度不均会有系统性偏差,补跑 skew。

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
# 前置:改 configs/model/zai-org/GLM-5.1.json 真实层数;换 vLLM 镜像 tag 到 0.21;
#       git log 确认 simulator ≥ 84846a1 且 profiler 有 --skip-moe;mkdir -p outputs logs
./scripts/docker-vllm.sh                                    # 进 vLLM 容器
# Step1 dense/attn: profile.sh 设 MODEL/HARDWARE=H20/VARIANT=fp8/SKIP_MOE=1/SKIP_SKEW=1
#                    → ./profiler/profile.sh |& tee logs/glm_profile_dense.log
# Step2 moe:        先 python -m profiler.core.hooks.verify_moe_hook(自检;FP8 monolithic
#                    守卫若报错 → VLLM_USE_FLASHINFER_MOE_FP8=0 带满本轮)
#                    再 profile.sh 去 SKIP_MOE、TP_DEGREES="1"、
#                    HF_OVERRIDES='{"first_k_dense_replace":0}' → ./profiler/profile.sh
# Step3 skew(可选): ONLY_SKEW=1 ./profiler/profile.sh(TP 还原 1,2,4,8、清 HF_OVERRIDES)
# 验收: dense.csv 13 层名 / per_sequence 2 / attention+moe 非空(详见 Step 3 末尾)
# Step4: 填 configs/cluster/h20_8_glm5_1_fp8.json(HBM 4000, NVLink 450 单向, latency ~500ns)
python -m bench run --model /path/to/GLM-5.1 --dataset <ds> --output-dir bench/results/glm51_h20 \
  --tensor-parallel-size 8 --enable-expert-parallel --max-num-seqs 256 \
  --max-num-batched-tokens 2048 --dtype bfloat16 --num-reqs 32 \
  |& tee logs/glm_bench_smoke.log                                        # Step5(8卡,先小跑)
# 切 simulator 环境:
python -m serving --cluster-config configs/cluster/h20_8_glm5_1_fp8.json --dataset <ds> \
  --dtype fp8 --max-num-seqs 256 --max-num-batched-tokens 2048 --num-reqs 32 \
  --output outputs/glm51_h20_sim.csv |& tee logs/glm_sim.log              # Step6(参数对齐 Step5)
# 回 vLLM 容器:
python -m bench validate --bench-dir bench/results/glm51_h20 \
  --sim-csv outputs/glm51_h20_sim.csv --sim-log logs/glm_sim.log          # Step7
```

## 失败模式速查

| 症状 | 根因 | 处置 |
|---|---|---|
| Step 1 崩 `Expected exactly one FusedMoE ... got 0` | 忘了 SKIP_MOE=1(dense 轮无 FusedMoE) | 加 SKIP_MOE=1 重跑;已测完的 shot 由 resume 保留 |
| Step 2 崩 `monolithic quant method ... cannot apply` | vLLM 给 FP8 挑了融合 MoE 后端 | `VLLM_USE_FLASHINFER_MOE_FP8=0`(或报错点名的 env)带满本轮重跑 |
| profile 看似卡死无输出 | FP8/MLA/DSA 首次 JIT(5-30min) | `VERBOSITY="--verbose"` 看进度;JIT 缓存在容器 ~/.cache |
| Step 6 `FileNotFoundError: ..//...` | `--output` 用了绝对路径 | 只用仓库相对路径(全局约定 2) |
| Step 7 找不到 sim log | 日志写在 /tmp(跨容器不可见) | 日志全部落 `logs/`(全局约定 1) |
| Step 6 日志有 `missing from the profile` | profile CSV 缺层 | 回 Step 1-3 验收逐项核对;通常是 Step 2 没跑或 variant 不匹配 |
| Step 6 variant `FileNotFoundError` | VARIANT 与 `--dtype` 解析不一致 | profiler VARIANT="fp8" + simulator `--dtype fp8` 必须配对 |
| bench OOM / KV 不足 | 8 卡装不下 + 256 并发 | 降 `--max-num-seqs`(两边同步降!)或 `--max-model-len` |
