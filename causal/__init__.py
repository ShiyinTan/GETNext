"""因果版 next-POI 代码包（对应 docs/causal-nextpoi-thinking.md 附录 D）。

这个文件夹和原来的 GETNext 并排存在：
  - 训练 / 预测都从这里的 train.py、predict.py 启动
  - 只会「只读」地借用 GETNext 里现成的工具（例如 Time2Vec、Acc@k 计算）
  - 不会改仓库根目录的 train.py / model.py

如果不太熟代码，建议先看 causal/README.md 开头的「不懂代码时怎么读」。
"""
