### 记录我学习《Python数学实验与建模》这本书的过程及其笔记

> 本人参加过一次美国大学生数学建模竞赛，单人独立建模并撰写论文文案（翻译美化是另一位大佬，其实这比赛还是更关注美化论文，图文好看就能得奖，下限靠美化，上限靠建模＋美化，编程成分可以比较少），最后是得了M奖(7.9%)。如果你在备赛时有什么疑问或者关于其他方面的小问题，非常欢迎你提出issue，我看到会解答。

- #### 安装过程

首先Python环境选择3.7系列（如3.7.2），为了后续库的安装与Python环境版本的匹配，你也可以选择其他版本的Python环境，但是库的版本也要做出相应改变，具体我会在后面说，所以推荐你是直接重装Python，这样你可以直接用我给你的库源，比你自己去网站下半天要方便的多。

~~Python官网：https://www.python.org/~~

~~去到Python官网进行安装包的下载，顺便可以把Vscode的环境也配了，在Vscode上写Python会非常舒服，大体流程是下载一个Vscode的安装包安装，然后选择Python的解释器版本，最后你可以敲一段简单代码右键在终端运行测试下，就可以了。（当然你用别的环境也ok）~~

> 更新：下载anaconda管理python，再结合pycharm或者jupyter用来编辑代码才是王道，点击[AnacondaCommand](AnacondaCommand.md)查看常用命令（会那么几个就够了，创建环境，激活，安库）

##### **接下来就是重中之重**

相信很多人为了学习这本书，卡在了安装特定的库这一步骤，接下来我会详细说明如何手动安装你学习这本书所需要的库（针对于容量比较大的库，无法直接pip install下来的库）和使用pip命令安装库（一些小库，一般我们写代码缺库可以直接用该命令下载）。

①手动安装（必须）

如果你想完全学完这本书不卡壳，按照我下面的操作进行即可

```
你需要安装的库如下，源文件我会放一个链接下载
cvxopt-1.3.0-cp37-cp37m-win_amd64
cvxpy-1.1.13-cp37-cp37m-win_amd64
ecos-2.0.7.post1-cp37-cp37m-win_amd64
fastcache-1.1.0-cp37-cp37m-win_amd64
numpy-1.21.0+mkl-cp37-cp37m-win_amd64
osqp-0.6.2.post0-cp37-cp37m-win_amd64
pandas-1.2.5-cp37-cp37m-win_amd64
scipy-1.7.3-cp37-cp37m-win_amd64
```

首先你选择一个特定的位置比如E盘，创建一个名为pythonPack的文件夹，然后放上你下载好的这8个文件。**注意，最好是下载我的源，因为在接下来敲命令时，名称要一一对应。**

然后，你打开你的Vscode，可以看到TERMINAL选项，也就是终端命令行，在里面依次键入如下命令（你也可以直接win+r，输入cmd唤出命令行，然后在里面输入命令，一样的）。注意：我选择的是E盘的pythonPack，如果你选择的是别的文件夹，那你要改一下相应的路径，如果你用的库源和我不同，也要改一下相应的名字。这里的cp37对应的就是3.7版本的python，所以如果你用的3.8版本python，你的库源也要用cp38，具体的库源的网站我会在后面给出。

```
pip install E:\pythonPack\cvxopt-1.3.0-cp37-cp37m-win_amd64.whl
pip install E:\pythonPack\cvxpy-1.1.13-cp37-cp37m-win_amd64.whl
pip install E:\pythonPack\ecos-2.0.7.post1-cp37-cp37m-win_amd64.whl
pip install E:\pythonPack\fastcache-1.1.0-cp37-cp37m-win_amd64.whl
pip install E:\pythonPack\numpy-1.21.0+mkl-cp37-cp37m-win_amd64.whl
pip install E:\pythonPack\osqp-0.6.2.post0-cp37-cp37m-win_amd64.whl
pip install E:\pythonPack\pandas-1.2.5-cp37-cp37m-win_amd64.whl
pip install E:\pythonPack\cvxpy-1.1.13-cp37-cp37m-win_amd64.whl
```

输入上述命令后，应该能够正常安装，你可以执行一条后输入命令pip list，这条命令的作用是列出你所安装的所有库，此时你应该能看到相关库名和版本号。

②自动安装

除了上述的8个库，一般我们安装库都是使用pip install的命令自动安装。具体使用命令格式如下

pip install <库名>

如：pip install networkx

所以手动安装完上面8个库后，以后代码遇到需要导入的库，直接使用pip install命令安装即可。

**资源：**

**手动安装的8个库（适用于3.7版本）（推荐使用，github库源中还有一个200m的库因太大未上传，numpy-1.21.0+mkl-cp37-cp37m-win_amd64.whl）：**

链接：https://pan.baidu.com/s/1UE6old8v3VZupz3QNmLQMw 
提取码：1esg

**其他版本的库的下载链接：**

https://www.lfd.uci.edu/~gohlke/pythonlibs/#lxml

文件名繁多，注意别下错了，所以推荐你直接重新安装python环境，省得找和下了

我总结的知识点为了能显示图片我会以pdf的格式放在github上

完成安装以后，恭喜你，可以开始这本书的学习了。——By wxtt

- #### 使用介绍

1. 先看readme将环境搭建好
2. 看书进行各个章节的学习，需要源代码和数据可以参考文件夹“书中源程序及其数据”，里面有各个章节的所有代码及Excel等数据
3. modelForPython和知识点文件夹是我自己学习时的产物，过程不太具有普适性，知识点可以参考一下
4. 如果觉得我写的不错，可以给我点个star，也方便查找源程序及数据，非常感谢
5. 不建议转载，引用请附上原文链接
