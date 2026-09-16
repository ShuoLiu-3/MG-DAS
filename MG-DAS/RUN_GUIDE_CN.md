# MG-DAS 运行指南

## 1. 环境准备

建议新建独立 Python 3.10 环境，然后在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

请使用 `numpy<2.0`，避免旧版 PyTorch 二进制包与 NumPy 2.x 不兼容。

## 2. 准备文件

代码不附带模型权重和数据集。请准备：

- 本地 Llama-2-Chat-7B 模型目录；
- 按 `docs/DATA_FORMAT.md` 整理的 BBQ JSONL；
- 相互独立的 MMLU 认证集和冻结测试集 JSONL。

修改 `configs/paper_protocol.json` 中的四个路径，并确保输出目录为空。

## 3. 运行前检查

```powershell
python -m mgdas_v2 --config configs/paper_protocol.json --stage validate
python -m mgdas_v2 --config configs/paper_protocol.json --stage preflight
python -m pytest -q
```

随后逐项核对 `PROTOCOL_AUDIT.md`。在读取认证集和测试结果之前，冻结全部
阈值、搜索网格和随机种子。

## 4. 分阶段运行

```powershell
python -m mgdas_v2 --config configs/paper_protocol.json --stage prepare-splits
python -m mgdas_v2 --config configs/paper_protocol.json --stage directions
python -m mgdas_v2 --config configs/paper_protocol.json --stage ablation
python -m mgdas_v2 --config configs/paper_protocol.json --stage aggregate
```

不要手工修改输出目录中的协议摘要、冻结测试账本或属性级结果。若修改代码、
数据或实验配置，请使用新的输出目录重新运行，避免混合不同协议下的结果。

## 5. 正式实验前的最小冒烟测试

正式运行前，可在独立临时配置中仅保留少量样本，确认模型能够加载、钩子路径
正确、激活维度为 4096，并且各阶段能生成预期文件。冒烟测试输出不能与正式
实验输出共用目录，也不能用于论文结果汇总。

