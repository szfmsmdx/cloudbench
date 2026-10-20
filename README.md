# CloudBench

本地运行的 OpenAI 兼容接口压测工具。网页服务、管理 API、数据库、压测和测试全部使用 Python。浏览器页面使用原生 HTML/CSS/JavaScript，无构建步骤。

| 路径 | 用途 |
| --- | --- |
| `server.py` | aiohttp 网页服务、管理 API、SQLite 存储、Python 压测进程管理 |
| `public/` | 网页与实时图表 |
| `pybench/` | 压测、数据读取、每秒采样、整场汇总和 CSV 导出 |
| `tests/` | Python 单元测试、API 和端到端测试 |
| `datasets/` | 数据适配说明、token 索引、长度分布与本机负载快照 |
| `data/cloudbench.sqlite` | 本机测试记录 |

## 启动

```powershell
python -m pip install -r requirements.txt
python server.py
```

打开 <http://127.0.0.1:3876>。服务和压测进程使用同一个 Python 解释器。本机已用 Python 3.13 验证。可通过 `CLOUDBENCH_PORT` 修改端口，或通过 `CLOUDBENCH_DATA_DIR` 指定数据库目录。

## 发压方式

| 模式 | 请求发送规则 |
| --- | --- |
| 同步 | 前一条请求完成后发送下一条；并发为 1。 |
| 异步，QPS > 0 | 按固定间隔计划发送。QPS=5 时每 200 ms 计划发送一条。 |
| 异步，QPS = 0 | 不设发送间隔，尽快全部发起。填写最大并发数时，完成一条补发一条。 |

最大并发数留空时使用请求总数。填写上限会限制实际发送速率。CSV 中保留计划时间、实际发起时间和排队延迟，便于检查是否达到目标 QPS。QPS=0 没有目标到达速率，主要看完成吞吐。

测试条数留空时，使用当前上下文上限内的全部合格样本，每条一次；填写条数时按指定请求数测试。

## 数据集

- **LongBench**：本地官方原始集，4,750 条；不使用 LongBench-E。
- **LongBench v2**：本地官方原始集，503 条；按所填模型上下文上限过滤超长样本。

新增五套公开负载，准备后可在网页直接选择：

| 数据集 | 回放方式 | 来源 |
| --- | --- | --- |
| Open-SWE-Traces | 编码 Agent，保留历史与工具调用 | [NVIDIA](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) |
| SWE-smith trajectories | SWE-agent 多轮轨迹 | [SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) |
| HotpotQA | 问题＋官方 distractor 段落 | [官方主页](https://hotpotqa.github.io/) |
| CRAG | 问题＋清理后的网页检索材料 | [Meta](https://github.com/facebookresearch/CRAG) |
| CRUD-RAG | 中文任务＋关联材料 | [官方仓库](https://github.com/IAAR-Shanghai/CRUD_RAG) |

### 数据准备

在租 GPU 前执行，只访问公开数据源，不调用模型或执行轨迹中的工具命令：

```powershell
python -m pybench.prepare all --limit 100 --max-turns 16
# 只准备一套；更多行使用 --limit，其他区段使用 --offset
python -m pybench.prepare open-swe --limit 100 --max-turns 16 --offset 100
# 已下载的原始 JSON、JSONL 或 JSONL.bz2
python -m pybench.prepare crag --input path/to/crag.jsonl.bz2 --limit 100
```

默认使用 Kimi K3 分词器，仅下载必要的分词文件，不加载模型或运行下载的 Python。可用 `--tokenizer cl100k_base` 改用另一参考编码。长度包含正文、历史和工具定义的JSON表示，未应用目标模型聊天模板；这是筛选参考，不代表 DeepSeek 等模型的实际 token 数。实际用量仍取接口 usage。

`datasets/prepared/` 保存本机 JSONL 与元数据，包含来源、准备时间、抽样区段、轮数、跳过原因、分词口径和 SHA-256。它们不上传 GitHub。默认只抽取来源顺序的一小段；CRUD 按任务交错取样。不是全量下载或生产流量分布；下载中断可重跑，也可用本地完整原始文件转换。HF Viewer 返回截断字段时会拒绝转换，避免把截断轨迹当成完整数据。

Open-SWE 使用 `v1.2/minisweagent`，SWE-smith 使用 `default/tool`；HotpotQA 使用 `distractor/validation`；CRAG 使用 Task 1/2 dev v4，流式解压所需部分；CRUD 使用官方 `crud_split/split_merged.json`（原始文件约27 MB，本地缓存）。来源与许可按上游发布，引用信息保留在元数据中。

### Agent 与 RAG 的统计范围

- Agent 的条数、QPS和并发按**会话**计算，同会话逐轮等待完成；输出吞吐、延迟和进度按实际逐轮请求计算。可设置轮间等待，CSV保留来源ID、会话ID、轮次和结束原因。
- 固定轨迹回放使用原始历史，模型新输出不替换下一轮的原始消息；不执行工具，不把原轨迹的 `resolved` 当成本次任务成功率。缺失工具返回ID的 Open-SWE 记录按原始调用顺序对应；未发布工具定义的轨迹根据调用名称补通用参数定义。
- CRUD 按官方六类任务转换：摘要、续写、纠错及单/双/三文档问答。续写仅提供原始开头；问答保留关联新闻，排除答案与标注推理。
- RAG 只测“给定材料后的生成”，参考答案单独保存，不发给模型；不包含检索、重排或向量库耗时，也未实现答案准确率评分。
- 最短输入可筛选长度档；Agent按可回放末轮长度筛选，保留较短的前置轮以建立缓存。超长后续轮停止回放，不裁剪历史。
- 显式条数大于样本池时会重复使用样本，可能提高缓存命中；冷前缀测试应使用不同区段和不重叠的样本。热前缀命中率以服务端数据为准。

本地原始文件保存在 `datasets/raw/`。长度索引用 Kimi K3 分词器预先计算，未包括聊天模板；筛选时为模板预留 512 token，并扣除输出上限。实际 token 用量以模型服务返回的 `usage` 为准。长度分布见 [统计表](datasets/length-distribution.md)。

GitHub 仓库包含代码、token 索引和分布统计；原始数据、分词器和本地测试数据库不上传。首次使用请按[统计表中的来源和路径](datasets/length-distribution.md#离线原始文件)准备原始数据。已有 token 索引，运行 CloudBench 无须下载模型或分词器。

## 结果

统计 TTFT、TPOT、E2E、输出吞吐、请求吞吐、成功率和缓存命中；可选录入 token 单价与 GPU 小时成本。逐请求结果和汇总均可导出 CSV。API Key 只保留在当前服务进程内存中；测试记录保存在 `data/cloudbench.sqlite`。

单请求超时可配置1–7200秒，网页默认900秒，API未填写时保持120秒。数据集用于性能负载，不计算官方任务准确率；输出长度由模型实际生成结果决定。

结果页每秒记录并绘制请求发起/成功/失败速率、usage 上报的输出吞吐、在途请求数和窗口内完成请求的 TTFT p50/p95。末尾不足 1 秒时与前一窗口合并计算速率，实际窗口时长一并保存。曲线保存在 SQLite 中，可在历史记录回看，或导出时间序列 CSV。没有 token usage 时输出吞吐显示未知；流式片段/s 单独统计，不当作 token/s。

## 验证

```powershell
python -m unittest discover -s tests -v
```
