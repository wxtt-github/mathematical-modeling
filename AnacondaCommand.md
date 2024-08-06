查看当前conda所有环境

```shell
conda info --envs
```

创建环境

```shell
conda create -n <环境名> python=3.10
```

激活环境

```shell
conda activate <环境名>
```

pip源（需要时在pip install <库名> 后加上）

```shell
# 清华源
-i https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里源
-i http://mirrors.aliyun.com/pypi/simple
```

使用和生成requirements.txt

```shell
# 安装requirements.txt中的库
pip install -r requirements.txt

# 生成requirements.txt
pip freeze > requirements.txt
```

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

