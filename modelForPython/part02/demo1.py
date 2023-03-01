import numpy as np
import pandas as pd
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats

# a = np.array([2,4,6,8])
# b = np.array(((1,2,3,4,5),(6,7,8,9,10),(10,9,1,2,3),(4,5,6,8,9.0)))
# print(a,'\n')
# print(b)

# c = np.empty((2,3),int)
# print(c)

# a = np.arange(0,3,0.5).reshape(2,3)
# np.savetxt('data1.txt', a)
# b = np.loadtxt('data1.txt')
# print(b)
# np.savetxt('data2.txt', a, fmt="%d", delimiter=',')
# c = np.loadtxt('data2.txt', delimiter=',')
# print(c)

# s1 = pd.Series(np.array([10.5, 20.5, 30.5]))
# s2 = pd.Series({"北京":10.5, "上海": 20.5, "广东": 30.5})
# s3 = pd.Series([10.5, 20.5, 30.5], index=['b', 'c', 'd'])
# print(s1,'\n')
# print(s2,'\n')
# print(s3,'\n')
# a = s3['c']
# print(a,'\n')
# print(np.mean(s3),'\n')
# print(s3.mean())

# a = np.arange(1,7).reshape(3,2)
# print(a)
# df1 = pd.DataFrame(a)
# df2 = pd.DataFrame(a, index=['a','b','c'], columns=['x1','x2'])
# df3 = pd.DataFrame({'x1':a[:,0],'x2':a[:,1]})
# print(df1)
# print(df2)
# print(df3)

'''
读取txt
'''
# a = pd.read_csv('data3.txt',sep=',',parse_dates={'birthday':[0,1,2]},skiprows=2,skipfooter=2,comment='#',thousands='&',engine='python')
# print(a)

'''
读取Excel
'''
# a = pd.read_excel('data4.xlsx',usecols=range(1,4))
# b = a.values
# c = a.describe()
# print(c)
# d = pd.DataFrame(b,index=np.arange(1,11),columns=['用户D','用户E','用户F'])
# f = pd.ExcelWriter('data5.xlsx')
# d.to_excel(f,'sheet1')
# d.to_excel(f,'sheet2')
# f.save()

'''
画柱形图
'''
# a = pd.read_excel('data4.xlsx',usecols=range(1,4))
# c = np.sum(a)
# ind = np.array([1,2,3])
# width = 0.2
# rc('font',size=16)
# bar(ind,c,width)
# ylabel('消费数据')
# xticks(ind,['用户A','用户B','用户C'],rotation=20)
# rcParams['font.sans-serif'] = ['SimHei']
# savefig('figure1.png',dpi=500)
# show()

'''
画散点图
'''
# x = np.array(range(8))
# y = '27.0 26.8 26.5 26.3 26.1 25.7 25.3 24.8'
# y = ','.join(y.split())
# y = np.array(eval(y))
# scatter(x,y)
# savefig('figure2.png',dpi=500)
# show()

'''
多个图形显示在一个图形画面
'''
# x = np.linspace(0,2*np.pi,200)
# # 参数分别为起点，终点，个数
# y1 = np.sin(x)
# y2 = np.cos(x**2)
# rc('font',size=16)
# rc('text',usetex=True)
# # 需要LaTeX环境
# plot(x,y1,'r',label='$sin(x)$',linewidth=2)
# plot(x,y2,'b--',label='$cos(x^2)$')
# xlabel('$x$')
# ylabel('$y$',rotation=0)
# savefig('figure3.png',dpi=500)
# legend()
# show()

'''
多个图形单独显示
'''
# x = np.linspace(0,2*np.pi,200)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.sin(x**2)
# rc('font',size=16)
# # rc('text',usetex=True)
# ax1 = subplot(2,2,1)
# # 划为2行2列，使用第一个格子
# ax1.plot(x,y1,'r',label='$sin(x)$')
# legend()
# ax2 = subplot(2,2,2)
# # 划为2行2列，使用第二个格子
# ax2.plot(x,y2,'b--',label='$cos(x)$')
# legend()
# ax3 = subplot(2,1,2)
# # 划为2行1列，使用第二个格子
# ax3.plot(x,y3,'k--',label='$sin(x^2)$')
# legend()
# savefig('figure4.png',dpi=500)
# show()

'''
绘制三维曲线
'''
# ax = plt.axes(projection='3d')
# z = np.linspace(0,100,1000)
# x = np.sin(z)*z
# y = np.cos(z)*z
# ax.plot3D(x,y,z,'k')
# plt.savefig('figure5.png',dpi=500)
# plt.show()

'''
绘制三维曲面
'''
# x = np.linspace(-6,6,30)
# y = np.linspace(-6,6,30)
# X,Y = np.meshgrid(x,y)
# Z = np.sin(np.sqrt(X**2+Y**2))
# # Z = np.sin(np.sqrt(x**2+y**2))
# # 上一行这么写是错误的,因为要多对多
# ax1 = plt.subplot(1,2,1,projection='3d')
# ax1.plot_surface(X,Y,Z,cmap='viridis')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_zlabel('z')
# ax2 = plt.subplot(1,2,2,projection='3d')
# ax2.plot_wireframe(X,Y,Z,color='c')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_zlabel('z')
# plt.savefig('figure6.png',dpi=500)
# plt.show()

'''
画等高线图
'''
# z = np.loadtxt('data6.txt')
# x = np.arange(0,1500,100)
# y = np.arange(1200,-10,-100)
# contr = plt.contour(x,y,z)
# plt.clabel(contr)
# plt.xlabel('$x$')
# plt.ylabel('$y$',rotation=0)
# plt.savefig('figure7.png',dpi=500)
# plt.figure()
# ax = plt.axes(projection='3d')
# X,Y=np.meshgrid(x,y)
# ax.plot_surface(X,Y,z,cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.savefig('figure7.png',dpi=500)
# plt.show()

'''
画向量图
'''
# x = np.linspace(1,15,11)
# y = np.linspace(0,10,12)
# x,y = np.meshgrid(x,y)
# v1 = y*np.cos(x)
# v2 = y*np.sin(x)
# plt.quiver(x,y,v1,v2)
# plt.savefig('figure8.png',dpi=500)
# plt.show()

'''
subplot另一种用法
'''
# x=np.linspace(0,2*np.pi,200)
# y1=np.sin(x); y2=np.cos(x); y3=np.sin(x*x); y4=x*np.sin(x)
# rc('font',size=16); rc('text', usetex=True)  #调用tex字库
# ax1=subplot(2,3,1)  #新建左上1号子窗口
# ax1.plot(x,y1,'r',label='$sin(x)$') #画图
# legend()  #添加图例
# ax2=subplot(2,3,2)  #新建2号子窗口
# ax2.plot(x,y2,'b--',label='$cos(x)$'); legend() 
# ax3=subplot(2,3,(3,6))  #3、6子窗口合并
# ax3.plot(x,y3,'k--',label='$sin(x^2)$'); legend()
# ax4=subplot(2,3,(4,5))  #4、5号子窗口合并
# ax4.plot(x,y4,'k--',label='$xsin(x)$'); legend()
# savefig('figure9.png',dpi=500)
# show()
# clf()

'''
可视化Excel,饼图，折线图，柱形图
'''
# a=pd.read_excel("Trade.xlsx")
# a['year']=a.Date.dt.year  #添加交易年份字段
# a['month']=a.Date.dt.month  #添加交易月份字段
# rc('font',family='SimHei') #用来正常显示中文标签

# ax1=subplot(2,3,1)   #建立第一个子图窗口
# Class_Counts=a.Order_Class[a.year==2012].value_counts()
# Class_Percent=Class_Counts/Class_Counts.sum()
# ax1.set_aspect(aspect='equal')  #设置纵横轴比例相等
# ax1.pie(Class_Percent,labels=Class_Percent.index,
#         autopct="%.1f%%")  #添加格式化的百分比显示
# ax1.set_title("2012年各等级订单比例")

# ax2=subplot(232)  #建立第2个子图窗口
# #统计2012年每月销售额
# Month_Sales=a[a.year==2012].groupby(by='month').aggregate({'Sales':np.sum})
# #下面使用Pandas画图
# Month_Sales.plot(title="2012年各月销售趋势",ax=ax2, legend=False)
# ax2.set_xlabel('')

# ax3=subplot(2,3,(3,6))
# cost=a['Trans_Cost'].groupby(a['Transport'])
# ts = list(cost.groups.keys())
# dd = np.array(list(map(cost.get_group, ts)))
# boxplot(dd); gca().set_xticklabels(ts)

# ax4=subplot(2,3,(4,5))
# hist(a.Sales[a.year==2012],bins=40, density=True)
# ax4.set_title("2012年销售额分布图");
# ax4.set_xlabel("销售额");
# savefig("figure10.png"); show()


'''
绘制伽马分布的概率密度曲线
'''
# x=np.linspace(0,15,100); rc('font',size=15)
# rc('text', usetex=True) 
# # 离散型
# plot(x,scipy.stats.gamma.pdf(x,4,0,2),'r*-',label="$\\alpha=4, \\beta=2$")
# plot(x,scipy.stats.gamma.pdf(x,4,0,1),'bp-',label="$\\alpha=4, \\beta=1$")
# plot(x,scipy.stats.gamma.pdf(x,4,0,0.5),'.k-',label="$\\alpha=4, \\beta=0.5$")
# plot(x,scipy.stats.gamma.pdf(x,2,0,0.5),'>g-',label="$\\alpha=2, \\beta=0.5$")
# legend(); xlabel('$x$'); ylabel('$f(x)$')
# savefig("figure11.png",dpi=500); show()

'''
绘制正态分布的概率密度曲线
'''
# mu0 = [-1, 0]; s0 = [0.5, 1]
# x = np.linspace(-7, 7, 100); plt.rc('font',size=15)
# plt.rc('text', usetex=True); plt.rc('axes',unicode_minus=False)
# f, ax = plt.subplots(len(mu0), len(s0), sharex=True, sharey=True)
# for i in range(2):
#     for j in range(2):
#         mu = mu0[i]; s = s0[j]
#         y = scipy.stats.norm(mu, s).pdf(x)
#         ax[i,j].plot(x, y)
#         ax[i,j].plot(1,0,label="$\\mu$ = {:3.2f}\n$\\sigma$ = {:3.2f}".format(mu,s))
#         ax[i,j].legend(fontsize=12)
# ax[1,1].set_xlabel('$x$')
# ax[0,0].set_ylabel('pdf($x$)')
# plt.savefig('figure12.png'); plt.show()

'''
绘制二项分布b(5,0.4)
'''
# # 法一：
# n, p=5, 0.4
# x=np.arange(6); y=scipy.stats.binom.pmf(x,n,p)
# plt.subplot(121); plt.plot(x, y, 'ro')
# plt.vlines(x, 0, y, 'k', lw=3, alpha=0.5)  #vlines(x, ymin, ymax)画竖线图
# #lw设置线宽度，alpha设置图的透明度
# plt.subplot(122); plt.stem(x, y, use_line_collection=True)
# plt.savefig("figure13.png", dpi=500); plt.show()
# # 法二：
# n,p = 5,0.4
# x = np.arange(0,6)
# y = scipy.stats.binom.pmf(x,n,p)
# plt.subplot(121); plt.plot(x, y, 'ro')
# plt.vlines(x, 0, y, 'k', lw=3, alpha=0.5)  #vlines(x, ymin, ymax)画竖线图
# #lw设置线宽度，alpha设置图的透明度
# plt.subplot(122); plt.stem(x, y, use_line_collection=True)
# plt.show()
























