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

公开数据集最大并发留空时默认最多16；LongBench 留空时使用请求数（最多5000）。填写上限会限制实际发送速率。CSV 保留计划时间、实际发起时间和排队延迟，便于检查是否达到目标 QPS。QPS=0 没有目标到达速率，主要看完成吞吐。

测试条数留空时，使用当前长度范围内的全部合格样本，每条一次；填写条数时从完整池中抽取指定数量，超过可用量会报错，不循环补足。

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
python -m pybench.prepare all
# 只导入一套完整来源；默认不限制样本数或会话轮数
python -m pybench.prepare open-swe
# Hugging Face 无法连接时，使用镜像下载文件
python -m pybench.prepare open-swe --endpoint https://hf-mirror.com
# 已下载的原始 JSON、JSONL 或 JSONL.bz2
python -m pybench.prepare crag --input path/to/crag.jsonl.bz2
# 只有主动指定 --limit / --max-turns 才会生成抽样池或限制轮次
python -m pybench.prepare hotpotqa --limit 100
```

默认使用 Kimi K3 分词器，仅下载必要的分词文件，不加载模型或运行下载的 Python。可用 `--tokenizer cl100k_base` 改用另一参考编码。长度包含正文、历史和工具定义的JSON表示；多轮历史按消息累加参考长度，未应用目标模型聊天模板。筛选口径与目标模型可能不同，实际用量取接口 usage。

`datasets/raw/` 保留完整来源文件和清单；`datasets/prepared/` 保存 SQLite 回放索引、压缩记录和元数据。Parquet 的非回放字段（如数百MB的补丁）保留在原文件，索引记录文件、行号及字段位置，避免发压时加载无关内容。转换失败的记录也保留来源和错误。网页区分来源总量、可回放量、来源分区及导入状态；完整导入通过计数校验后才替换旧池。大文件不上传 GitHub。

完整导入针对下列明确分区，不把其他版本或分区混入同一个池：Open-SWE `v1.2/minisweagent`，SWE-smith `default/tool`，HotpotQA `distractor/validation`，CRAG Task 1/2 dev v4，CRUD 官方 `crud_split/split_merged.json` 六类任务。HF 来源使用完整 Parquet 文件清单，记录文件 SHA-256 并校验总行数。各池实际数量与上游版本见网页及元数据，不固定为100。

若服务器无法访问 HF 元数据接口，可在可访问的机器用 `pybench.sources.source_manifest(key)` 保存官方清单，传到服务器后加 `--manifest manifest.json --endpoint https://hf-mirror.com`。文件下载可断点重试；已完成文件会保留。元数据记录来源和许可链接。

### Agent 与 RAG 的统计范围

- Agent 的条数、QPS和并发按**会话**计算，同会话逐轮等待完成；输出吞吐、延迟和进度按实际逐轮请求计算。可设置轮间等待，CSV保留来源ID、会话ID、轮次和结束原因。
- 固定轨迹回放使用原始历史，模型新输出不替换下一轮的原始消息；不执行工具，不把原轨迹的 `resolved` 当成本次任务成功率。缺失工具返回ID的 Open-SWE 记录按原始调用顺序对应；未发布工具定义的轨迹根据调用名称补通用参数定义。
- CRUD 按官方六类任务转换：摘要、续写、纠错及单/双/三文档问答。续写仅提供原始开头；问答保留关联新闻，排除答案与标注推理。
- RAG 只测“给定材料后的生成”，参考答案单独保存，不发给模型；不包含检索、重排或向量库耗时，也未实现答案准确率评分。
- 最短输入可筛选长度档；Agent按可回放末轮长度筛选，保留较短的前置轮以建立缓存。超长后续轮停止回放，不裁剪历史。
- 单次测试不重复抽样，超出当前长度范围可用量时会报错。多场测试仍可能选中相同前缀；热前缀命中率以服务端数据为准。

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
