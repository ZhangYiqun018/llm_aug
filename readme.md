# 文件结构（暂定）

dataset -> 存放训练数据和测试数据
result -> 存放训练结果

infer.py, infer.sh -> 推理脚本
fintune_moss.py -> chat版moss sft代码

## 运行finetune程序

使用tmux把程序放到后台跑了

```
tmux ls  # 查看已经存在的session
tmux new -s <session-name> # 新建session
Ctrl+b d  # 临时退出当前的session  (松开ctrl+b再按d)
tmux a -t <session-name>  # 进入已经存在的session
Ctrl+b :kill-session  #删除当前进入的session 
Ctrl+b n # 当前session中创建新的window
```

