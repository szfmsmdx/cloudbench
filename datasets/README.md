# 公开数据负载

准备方式见根目录 [README](../README.md#数据准备)。大文件和本地快照不进入版本库。

每套准备结果为 `prepared/<dataset>.jsonl` 和 `.meta.json`。

JSONL一行表示一个RAG请求或一个Agent会话：

```json
{"id":"source-id","turns":[{"messages":[{"role":"user","content":"问题与材料"}],"input_tokens":120}],"reference_answer":"仅供后续评分，不发给模型"}
```

Agent会话的每个turn保存该轮完整原始前缀与可选tools；回放严格按turn顺序，后续不使用当前生成结果。元数据记录实际抽取样本数、逐轮请求数、长度分布、来源、抽样区段、tokenizer与数据哈希。长度分布按请求计，样本数按会话计，两者不同。

默认抽样用于检查链路；性能定容前应按业务的会话轮数、上下文长度、工具输出量和RAG材料量重新抽样。
