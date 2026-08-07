---
name: 数据分析助手
description: 分析上传的 CSV/XLSX 表格，返回行列数、列类型、缺失值和数值列均值。
triggers:
  - 数据分析
  - 表格分析
  - 数据处理
  - data_analysis_skill
mode: workflow
workflow:
  kind: table_analysis
  steps:
    - load_latest_table
    - clean_table
    - profile_columns
    - numeric_means
    - summarize
examples:
  - 上传 CSV 后输入：用数据分析助手处理上传的 csv
---

当用户要求分析上传表格时，调用后端 `table_analysis` 工作流。

该工作流会读取 CSV/XLSX，规范列名，删除全空行列，识别数值列，统计行列规模、列类型、缺失值和数值列均值。

回答时直接使用后端工作流结果，不要输出伪代码，不要伪造未计算的统计量。
