# LongBench 与 LongBench v2 输入长度分布

统计日期：2026-10-16。单位是 Kimi K3 token，1K = 1024 token。区间左闭右开。

| 输入 token 数 | LongBench（4,750 条） | LongBench v2（503 条） |
| --- | ---: | ---: |
| <1K | 38（0.80%） | 0 |
| 1K–2K | 360（7.58%） | 0 |
| 2K–4K | 787（16.57%） | 0 |
| 4K–8K | 1,233（25.96%） | 0 |
| 8K–16K | 1,764（37.14%） | 30（5.96%） |
| 16K–32K | 472（9.94%） | 88（17.50%） |
| 32K–64K | 96（2.02%） | 72（14.31%） |
| 64K–128K | 0 | 128（25.45%） |
| 128K–256K | 0 | 89（17.69%） |
| 256K–512K | 0 | 48（9.54%） |
| 512K–1M | 0 | 17（3.38%） |
| ≥1M | 0 | 31（6.16%） |

| 数据集 | 最短 | 平均 | 最长 |
| --- | ---: | ---: | ---: |
| LongBench | 131 | 9,693 | 65,026 |
| LongBench v2 | 10,019 | 241,745 | 4,009,489 |

## 统计口径

- 分词器：`moonshotai/Kimi-K3` 官方 `tiktoken.model` 和 `tokenization_kimi.py` 中的正则模式。分词器文件 SHA-256：`B6C497A7469B33CED9C38AFB1AD6E47F03F5E5DC05F15930799210EC050C5103`。
- LongBench：逐行统计 `context + 换行 + input`。LongBench v2：统计 `context + question + choice_A/B/C/D`，字段间以换行连接。未加入聊天模板和系统提示词，因此实际 API 输入 token 会略多。
- 压测只使用 LongBench 基础版和 LongBench v2；原始压缩包中的 `_e.jsonl` 文件不会载入。
- 计入原始文件全部有效行，没有截断到模型上下文上限。LongBench v2 中 ≥1M token 的 31 条需要在真实压测前按模型和服务的上限筛除或单独处理。
- 机器可读结果：[LongBench](longbench-distribution-v1.json)、[LongBench v2](longbench-distribution-v2.json)。

## 离线原始文件

| 数据集 | 项目内文件 | 来源 | SHA-256 |
| --- | --- | --- | --- |
| LongBench | `raw/longbench/data.zip`；已解压至 `raw/longbench/extracted/data/` | [官方 Hugging Face 数据库](https://huggingface.co/datasets/zai-org/LongBench/blob/main/data.zip) | `CB45B11A4133C6BC1D6A44B0F8E701335FF1E543195DB1103472E575857F7F64` |
| LongBench v2 | `raw/longbench-v2/data.json` | [官方 Hugging Face 数据库](https://huggingface.co/datasets/zai-org/LongBench-v2/blob/main/data.json) | `15D61C22D92C96900B3C4948B6AEEA218D3214B676A65DF48E7B8555604C7FE2` |
| Kimi K3 分词器 | `tokenizers/kimi-k3/` | [官方模型库](https://huggingface.co/moonshotai/Kimi-K3/tree/main) | 见统计口径 |

CloudBench 从本地文件读取这两个数据集，不经过网页导入器。长度索引用 Kimi K3 分词器计算；筛选时还会为聊天模板和输出 token 留出空间。
