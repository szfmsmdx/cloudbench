# CloudBench

本地运行的 OpenAI 兼容接口压测工具。网页服务、管理 API、数据库、压测和测试全部使用 Python。浏览器页面使用原生 HTML/CSS/JavaScript，无构建步骤。

| 路径 | 用途 |
| --- | --- |
| `server.py` | aiohttp 网页服务、管理 API、SQLite 存储、Python 压测进程管理 |
| `public/` | 网页与实时图表 |
| `pybench/` | 压测、数据读取、每秒采样、整场汇总和 CSV 导出 |
| `tests/` | Python 单元测试、API 和端到端测试 |
| `datasets/` | 两套原始数据、token 索引和长度分布 |
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

本地原始文件保存在 `datasets/raw/`。长度索引用 Kimi K3 分词器预先计算，未包括聊天模板；筛选时为模板预留 512 token，并扣除输出上限。实际 token 用量以模型服务返回的 `usage` 为准。长度分布见 [统计表](datasets/length-distribution.md)。

GitHub 仓库包含代码、token 索引和分布统计；原始数据、分词器和本地测试数据库不上传。首次使用请按[统计表中的来源和路径](datasets/length-distribution.md#离线原始文件)准备原始数据。已有 token 索引，运行 CloudBench 无须下载模型或分词器。

## 结果

统计 TTFT、TPOT、E2E、输出吞吐、请求吞吐、成功率和缓存命中；可选录入 token 单价与 GPU 小时成本。逐请求结果和汇总均可导出 CSV。API Key 只保留在当前服务进程内存中；测试记录保存在 `data/cloudbench.sqlite`。

当前单请求超时为 120 秒。LongBench 在这里用于性能负载，不计算官方任务准确率；输出长度由模型实际生成结果决定。

结果页每秒记录并绘制请求发起/成功/失败速率、usage 上报的输出吞吐、在途请求数和窗口内完成请求的 TTFT p50/p95。末尾不足 1 秒时与前一窗口合并计算速率，实际窗口时长一并保存。曲线保存在 SQLite 中，可在历史记录回看，或导出时间序列 CSV。没有 token usage 时输出吞吐显示未知；流式片段/s 单独统计，不当作 token/s。

## 验证

```powershell
python -m unittest discover -s tests -v
```
