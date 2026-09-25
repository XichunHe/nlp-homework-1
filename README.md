# 自然语言处理课程作业一 文本分类实验

作者：何熹淳（2410721）

包含原始 CSV、七组实验代码、固定数据划分、实测结果、逐条测试预测和中文 PDF 报告。

## 环境准备

建议 Python 3.11 或 3.12。在本目录运行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS 则执行 source .venv/bin/activate
python -m pip install -r requirements.txt
```

BERT 已在 RTX 3090 上完成 3 轮训练。复现时建议使用 CUDA GPU；先按 PyTorch 官网为机器的 CUDA 驱动安装相应版本，再安装其他依赖。检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 运行实验

```bash
# 三种词袋方法
python train.py --task bow
# 两组自训练 Word2Vec：固定单线程，便于复现
python train.py --task embeddings --embedding-methods w2v_ag,w2v_nyt
# 官方 GloVe 下载约 862 MB 压缩包，仅提取 100 维文件
python download_glove.py
python train.py --task embeddings --embedding-methods glove --glove assets/glove.6B.100d.txt
# BERT 全参数微调，max_length=64，完整训练 3 epochs
python train.py --task bert --batch-size 16
# 逐条预测复核指标，并补充排除训练重复文本后的测试结果
python verify_results.py
# 审计测试集在 64-token 上限下的截断比例
python audit_truncation.py
# 依据现有 results 文件生成报告
python build_report.py
```

已有 GloVe 文件时直接通过 `--glove` 指定路径，无须重复下载。BERT 可用 `--bert-model /path/to/bert-base-uncased` 加载提前下载的官方模型目录，不能换成其他型号冒充本作业要求。GPU 内存不足时将 `--batch-size` 改成 8；结果中会保存实际设置。

## GPU 服务器运行

本次在用户指定的 3090 主机上，仅使用第 0 张卡，工作目录为 `/defaultShare/archive/caojiaolong/Interns/hexinchun/hw1/HW-1`，模型权重放在同级 `bert-base-uncased/`。保留 `data/` 和 `results/splits.json` 后，本次实际执行：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
../.venv/bin/python train.py --task bert --bert-model ../bert-base-uncased --batch-size 16 --threads 4
```

测试 Accuracy=0.9782986111，Macro-F1=0.9510509105；验证集选择第 3 轮。训练记录见 `results/bert_train.log`，运行环境见 `results/environment_bert.json`。测试预测与 JSON 已取回本地并通过独立复算。最佳权重仍保留在服务器 `results/bert_best.pt`，未纳入普通 GitHub 提交包；可按此命令复现或在服务器继续推理。

## 方法与复现约定

- seed=42，先分层随机留出 20%，再将留出集均分为验证集和测试集；实际数量 9215 / 1152 / 1152。`splits.json` 保存原始 CSV 的零起始行号及 SHA-256，CSV 内容变化时拒绝复用错误划分。
- 词袋、TF-IDF 的词表及 IDF、NYT Word2Vec 仅使用训练集；AG Word2Vec 使用给定的全部 90000 条 AG 文本。
- 分词使用正则表达式，将英文转小写并保留英文词、词内单引号和数字，不删除停用词。词袋不截断、不限制词表、不做额外特征筛选。
- 三种词袋共用同一分词规则；题目称“三种”但只列出两种，因此 TF-IDF 是为补足数量而加入的明确假设。
- 所有词向量为 100 维；按文档内有效词的出现次数取算术平均。OOV 不参与平均，没有有效词时使用零向量。
- Word2Vec 使用 Skip-gram、window=5、min_count=2、negative=5、epochs=10、workers=1 和稳定哈希。
- Logistic Regression 使用 L2 正则和 lbfgs；在 C={0.1,1,10} 上按验证集 Macro-F1 选择，平局优先较小 C。选定模型不使用测试集调参，也不把验证集合并回训练集。
- BERT 使用全参数微调、AdamW、学习率 2e-5、weight_decay=0.01、10% warmup、线性衰减、梯度范数上限 1；完整训练 3 轮后选择验证 Macro-F1 最高的轮次，最后才评测测试集。
- 发现 72 条标准化重复文本；主实验遵循题目要求保留原始行随机划分，并额外报告排除与训练集重复的测试文本后的指标。该补充检查不能排除语义近重复。
- BERT 测试集 1,152 篇新闻全部超过 64 token；原始 token 长度中位数 888.5，90 分位数 1,242.8。BERT 只见到开头 64 token，传统方法则读取全文；比较时需考虑输入范围差异。
- `verify_results.py` 不依赖 scikit-learn，独立计数计算 Accuracy 和 Macro-F1，核对预测行号、标签、集合互斥和结果 JSON。

## 文件说明

| 路径 | 内容 |
|---|---|
| train.py | 数据划分、所有模型训练与评测 |
| download_glove.py | 下载官方 6B GloVe |
| verify_results.py | 独立指标复核与重复敏感性检查 |
| audit_truncation.py | 测量 BERT 输入截断比例 |
| build_report.py | 读取真实结果生成中文 PDF |
| data/ | 用户提供的 nyt.csv 和 ag.csv |
| results/splits.json | 全实验共享的固定划分及数据校验值 |
| results/data_audit.json | 类别、缺失、重复、跨集合重叠统计 |
| results/*.json | 超参数、验证与测试指标、混淆矩阵 |
| results/*_predictions.csv | 测试样本原始行号、真实和预测标签 |
| report.pdf | 实验报告，运行状态见报告首页 |

## GitHub 提交

全部 7 个实验已有结果。提交前确认老师是否接受 TF-IDF 补充项，并按课程要求提交 GitHub 仓库链接。已配置 `.gitignore` 排除环境、预训练大权重和 BERT checkpoint。数据文件分别约 22 MB 和 44 MB，在一般单文件限制内；分享时遵循课程的数据使用要求。

## 来源

- 课程材料：作业一要求.docx、nyt.csv、ag.csv。
- GloVe 官方说明：https://nlp.stanford.edu/projects/glove/
- BERT 指定模型：https://huggingface.co/google-bert/bert-base-uncased
- 本次实测 GloVe 从 SLU-CSCI4750 镜像下载，解压文件 SHA-256：`95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544`。下载脚本默认仍使用 Stanford 官方源。
