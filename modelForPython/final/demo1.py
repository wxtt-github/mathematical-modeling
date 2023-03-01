'''
分析几种LED光对于LED屏幕亮度的影响
目的得出一个线性方程
总共有6个因子，红黄白蓝绿灰，对于光照强度的影响

主成分分析，分析6个因子的贡献率
'''

# import numpy as np
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_1.txt")   #加载表中x1,x2,x3,x4,x5,x6,y的13行7列数据
# md=LinearRegression().fit(a[:,:6],a[:,6])    #构建并拟合模型
# y=md.predict(a[:,:6])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:6],a[:,6])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f%10.4f%10.4f%10.4f%10.4f"%(b0,b12[0],b12[1],b12[2],b12[3],b12[4],b12[5]))
# print("拟合优度R^2=%.4f"%R2)

# # 分界线
# import numpy as np
# from sklearn.decomposition import PCA
# a = np.loadtxt("Pdata20_1.txt")
# b = np.r_[a[:,0:6]]
# md=PCA().fit(b)  #构造并训练模型
# print("特征值为：",md.explained_variance_)
# print("各主成分的贡献率：",md.explained_variance_ratio_)
# print("奇异值为：",md.singular_values_)
# print("各主成分的系数：\n",md.components_)  #每行是一个主成分
# """下面直接计算特征值和特征向量，和库函数进行对比"""
# cf=np.cov(b.T)  #计算协方差阵
# c,d=np.linalg.eig(cf) #求特征值和特征向量
# print("特征值为：",c)
# print("特征向量为：\n",d)
# print("各主成分的贡献率为：",c/np.sum(c))


'''
画眩光贡献率图
'''
# import numpy as np
# import pandas as pd
# from matplotlib.pyplot import *
# # x = np.array(range(1,17))
# x=np.arange(1,17,1)
# y = '87.7 12.2 1.79e-12 2.45e-12 1.96e-12 9.76e-12 8.94e-12 1.29e-12 3.33e-12 6.16e-12 9.59e-12 1.39e-12 1.96e-12 2.89e-12 3.62e-12 9.46e-12'
# y = ','.join(y.split())
# y = np.array(eval(y))
# rc('font',size=10)
# rc('text',usetex=True)
# # 需要LaTeX环境
# xlabel('Sample Number')
# ylabel('Rate of Contribution',rotation=90)
# scatter(x,y)
# savefig('figure2.png',dpi=500)
# show()



'''
双回归模型拟合眩光
'''
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_2.txt")   
# md=LinearRegression().fit(a[:,:3],a[:,3])    #构建并拟合模型
# y=md.predict(a[:,:3])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:3],a[:,3])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f%10.4f"%(b0,b12[0],b12[1],b12[2]))
# print("拟合优度R^2=%.4f"%R2)

# x=np.arange(1,35,1)
# plt.rc('font',size=10); plt.rc('text',usetex=True)
# plt.plot(x,a[:,3],'r*',label="raw data")
# plt.plot(x,62.9707+11.5108*a[:,0]+2.2955*a[:,1]-8.5552*a[:,2],'g',label="fitting curve")
# plt.legend()
# plt.xlabel('Sample Number')
# plt.ylabel('CBC',rotation=90)
# plt.savefig("figure20_2.png")
# plt.show()

# print("相关系数矩阵:\n",np.corrcoef(a[:,:3].T))
# for k in range(0,34):
#     print(a[k,0]**a[k,1])


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_4.txt")   
# md=LinearRegression().fit(a[:,:4],a[:,4])    #构建并拟合模型
# y=md.predict(a[:,:4])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:4],a[:,4])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f%10.4f%10.4f"%(b0,b12[0],b12[1],b12[2],b12[3]))
# print("拟合优度R^2=%.4f"%R2)

# x=np.arange(1,35,1)
# plt.rc('font',size=10); plt.rc('text',usetex=True)
# plt.plot(x,a[:,4],'r*',label="raw data")
# plt.plot(x,129.2294+67.6860*a[:,0]-5.6876*a[:,1]-10.1205*a[:,2]+2.1722*a[:,3],'g',label="fitting curve")
# plt.legend()
# plt.xlabel('Sample Number')
# plt.ylabel('CBC',rotation=90)
# plt.savefig("figure20_2.png")
# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_3.txt")   
# md=LinearRegression().fit(a[:,:5],a[:,5])    #构建并拟合模型
# y=md.predict(a[:,:5])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:5],a[:,5])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f%10.4f%10.4f%10.4f"%(b0,b12[0],b12[1],b12[2],b12[3],b12[4]))
# print("拟合优度R^2=%.4f"%R2)

# x=np.arange(1,35,1)
# plt.rc('font',size=10); plt.rc('text',usetex=True)
# plt.plot(x,a[:,5],'r*',label="raw data")
# plt.plot(x,-50.7498+101.7299*a[:,0]+6.9210*a[:,1]+10.9290*a[:,2]-1.2575*a[:,3]-30.9637*a[:,4],'g',label="linear regression")

# plt.plot(x,-4.684143622643461+89.5295042*a[:,0]+3.99565818*a[:,1]+6.58558765*a[:,2]+0.30167616*a[:,3]-25.24637061*a[:,4],'b',label="ridge regression")
# plt.legend()
# plt.xlabel('Sample Number')
# plt.ylabel('CBC',rotation=90)
# plt.savefig("figure20_2.png")
# plt.show()
# print("相关系数矩阵:\n",np.corrcoef(a[:,:5].T))





'''
Logistic模型(黑)
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.pyplot import *
a=[]; b=[];
with open("Pdata20_5.txt") as f:    #打开文件并绑定对象f
    s=f.read().splitlines()  #返回每一行的数据
for i in range(0, len(s),2):  #读入奇数行数据
    d1=s[i].split("\t")
    for j in range(len(d1)):
        if d1[j]!="": a.append(eval(d1[j]))  #把非空的字符串转换为年代数据
for i in range(1, len(s), 2):  #读入偶数行数据
    d2=s[i].split("\t")
    for j in range(len(d2)):
        if d2[j] != "": b.append(eval(d2[j])) #把非空的字符串转换为人口数据
c=np.vstack((a,b))  #构造两行的数组
# np.savetxt("Pdata8_10_2.txt", c)  #把数据保存起来供下面使用
x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
bd=((0, 200), (0.1,1000))  #约束两个参数的下界和上界
popt, pcov=curve_fit(x, a[1:], b[1:], bounds=bd)
print(popt)
print("8年的预测值为：", x(8, *popt))


# 画图
rc('font',size=10)
rc('text',usetex=True)
d=np.loadtxt("Pdata20_5.txt")
x=np.arange(8,14,1)
# rc('font',size=10);
# # plt.rc('font',family='SimHei')
# rc('text',usetex=True)
ax1=subplot(1,4,1)
ax1.plot(x,d[1,:],'r*',label="raw data")
ax1.plot(x,1.0000000e+03/(1+(1.0000000e+03/26.3-1)*np.exp(-2.3112453e-02*(x-8))),'g',label="fitting curve\n(380-780nm)")
ax1.legend()
xlabel('$Time/year$')
ylabel('$W(t)/g$',rotation=90)
legend()
d=np.loadtxt("Pdata20_6.txt")
ax2=subplot(1,4,2)
ax2.plot(x,d[1,:],'r*',label="raw data")
ax2.plot(x,1.00000000e+03/(1+(1.00000000e+03/26.3-1)*np.exp(-2.58442778e-02*(x-8))),'g',label="fitting curve\n(580-680nm)")
ax2.legend()
xlabel('$Time/year$')
ylabel('$W(t)/g$',rotation=90)
legend()
d=np.loadtxt("Pdata20_7.txt")
ax3=subplot(1,4,3)
ax3.plot(x,d[1,:],'r*',label="raw data")
ax3.plot(x,9.99999907e+02/(1+(9.99999907e+02/26.3-1)*np.exp(-3.38881644e-03*(x-8))),'g',label="fitting curve\n(380-480nm)")
ax3.legend()
xlabel('$Time/year$')
ylabel('$W(t)/g$',rotation=90)
legend()
d=np.loadtxt("Pdata20_8.txt")
ax4=subplot(1,4,4)
ax4.plot(x,d[1,:],'r*',label="raw data")
ax4.plot(x,1.0000000e+03/(1+(1.0000000e+03/26.3-1)*np.exp(-1.48095213e-02*(x-8))),'g',label="fitting curve\n(480-580nm)")
ax4.legend()
xlabel('$Time/year$')
ylabel('$W(t)/g$',rotation=90)
legend()
# plt.savefig("figure20_5.png")
show()


SST=0
SSR=0
print(d[1,:])
mean = 0
for i in range(0,6):
    mean += d[1,i]
mean /= 6
print(mean)

x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
for i in range(8,14):
    SST += (d[1, i-8] - mean)**2
    SSR += (x(i, *popt) - mean)**2
R2 = SSR / SST
print("拟合优度为",R2)


'''
Logistic模型(红)
'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# a=[]; b=[];
# with open("Pdata20_6.txt") as f:    #打开文件并绑定对象f
#     s=f.read().splitlines()  #返回每一行的数据
# for i in range(0, len(s),2):  #读入奇数行数据
#     d1=s[i].split("\t")
#     for j in range(len(d1)):
#         if d1[j]!="": a.append(eval(d1[j]))  #把非空的字符串转换为年代数据
# for i in range(1, len(s), 2):  #读入偶数行数据
#     d2=s[i].split("\t")
#     for j in range(len(d2)):
#         if d2[j] != "": b.append(eval(d2[j])) #把非空的字符串转换为人口数据
# c=np.vstack((a,b))  #构造两行的数组
# # np.savetxt("Pdata8_10_2.txt", c)  #把数据保存起来供下面使用
# x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
# bd=((0, 200), (0.1,1000))  #约束两个参数的下界和上界
# popt, pcov=curve_fit(x, a[1:], b[1:], bounds=bd)
# print(popt)
# print("8年的预测值为：", x(8, *popt))


# # 画图
# plt.rc('font',size=10)
# plt.rc('text',usetex=True)
# d=np.loadtxt("Pdata20_6.txt")
# x=np.arange(8,14,1)
# # plt.rc('font',size=16); plt.rc('font',family='SimHei')
# plt.plot(x,d[1,:],'r*',label="raw data")
# plt.plot(x,1.00000000e+03/(1+(1.00000000e+03/26.3-1)*np.exp(-2.58442778e-02*(x-8))),'g',label="fitting curve")
# plt.legend()
# plt.xlabel('$Time/year$')
# plt.ylabel('$W(t)/g$',rotation=90)
# plt.savefig("figure20_5.png")
# plt.show()


# SST=0
# SSR=0
# print(d[1,:])
# mean = 0
# for i in range(0,6):
#     mean += d[1,i]
# mean /= 6
# print(mean)

# x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
# for i in range(8,14):
#     SST += (d[1, i-8] - mean)**2
#     SSR += (x(i, *popt) - mean)**2
# R2 = SSR / SST
# print("拟合优度为",R2)




'''
Logistic模型(蓝)
'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# a=[]; b=[];
# with open("Pdata20_7.txt") as f:    #打开文件并绑定对象f
#     s=f.read().splitlines()  #返回每一行的数据
# for i in range(0, len(s),2):  #读入奇数行数据
#     d1=s[i].split("\t")
#     for j in range(len(d1)):
#         if d1[j]!="": a.append(eval(d1[j]))  #把非空的字符串转换为年代数据
# for i in range(1, len(s), 2):  #读入偶数行数据
#     d2=s[i].split("\t")
#     for j in range(len(d2)):
#         if d2[j] != "": b.append(eval(d2[j])) #把非空的字符串转换为人口数据
# c=np.vstack((a,b))  #构造两行的数组
# # np.savetxt("Pdata8_10_2.txt", c)  #把数据保存起来供下面使用
# x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
# bd=((0, 200), (0.1,1000))  #约束两个参数的下界和上界
# popt, pcov=curve_fit(x, a[1:], b[1:], bounds=bd)
# print(popt)
# print("8年的预测值为：", x(8, *popt))


# # 画图
# plt.rc('font',size=10)
# plt.rc('text',usetex=True)
# d=np.loadtxt("Pdata20_7.txt")
# x=np.arange(8,14,1)
# # plt.rc('font',size=16); plt.rc('font',family='SimHei')
# plt.plot(x,d[1,:],'r*',label="raw data")
# plt.plot(x,9.99999907e+02/(1+(9.99999907e+02/26.3-1)*np.exp(-3.38881644e-03*(x-8))),'g',label="fitting curve")
# plt.legend()
# plt.xlabel('$Time/year$')
# plt.ylabel('$W(t)/g$',rotation=90)
# plt.savefig("figure20_5.png")
# plt.show()


# SST=0
# SSR=0
# print(d[1,:])
# mean = 0
# for i in range(0,6):
#     mean += d[1,i]
# mean /= 6
# print(mean)

# x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
# for i in range(8,14):
#     SST += (d[1, i-8] - mean)**2
#     SSR += (x(i, *popt) - mean)**2
# R2 = SSR / SST
# print("拟合优度为",R2)





'''
Logistic模型(绿)
'''
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# a=[]; b=[];
# with open("Pdata20_8.txt") as f:    #打开文件并绑定对象f
#     s=f.read().splitlines()  #返回每一行的数据
# for i in range(0, len(s),2):  #读入奇数行数据
#     d1=s[i].split("\t")
#     for j in range(len(d1)):
#         if d1[j]!="": a.append(eval(d1[j]))  #把非空的字符串转换为年代数据
# for i in range(1, len(s), 2):  #读入偶数行数据
#     d2=s[i].split("\t")
#     for j in range(len(d2)):
#         if d2[j] != "": b.append(eval(d2[j])) #把非空的字符串转换为人口数据
# c=np.vstack((a,b))  #构造两行的数组
# # np.savetxt("Pdata8_10_2.txt", c)  #把数据保存起来供下面使用
# x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
# bd=((0, 200), (0.1,1000))  #约束两个参数的下界和上界
# popt, pcov=curve_fit(x, a[1:], b[1:], bounds=bd)
# print(popt)
# print("8年的预测值为：", x(8, *popt))


# # 画图
# plt.rc('font',size=10)
# plt.rc('text',usetex=True)
# d=np.loadtxt("Pdata20_8.txt")
# x=np.arange(8,14,1)
# # plt.rc('font',size=16); plt.rc('font',family='SimHei')
# plt.plot(x,d[1,:],'r*',label="raw data")
# plt.plot(x,1.0000000e+03/(1+(1.0000000e+03/26.3-1)*np.exp(-1.48095213e-02*(x-8))),'g',label="fitting curve")
# plt.legend()
# plt.xlabel('$Time/year$')
# plt.ylabel('$W(t)/g$',rotation=90)
# plt.savefig("figure20_5.png")
# plt.show()


# SST=0
# SSR=0
# print(d[1,:])
# mean = 0
# for i in range(0,6):
#     mean += d[1,i]
# mean /= 6
# print(mean)

# x=lambda t, r, xm: xm/(1+(xm/26.3-1)*np.exp(-r*(t-8)))
# for i in range(8,14):
#     SST += (d[1, i-8] - mean)**2
#     SSR += (x(i, *popt) - mean)**2
# R2 = SSR / SST
# print("拟合优度为",R2)










'''
聚类分析（眩光）
'''
# import numpy as np
# from sklearn import preprocessing as pp
# import scipy.cluster.hierarchy as sch
# import matplotlib.pyplot as plt
# a=np.loadtxt("Pdata20_9.txt")
# b=pp.minmax_scale(a.T)   #数据规格化
# d = sch.distance.pdist(b,metric='euclidean')  #求对象之间的两两距离向量
# # d = sch.distance.euclidean(b)
# dd = sch.distance.squareform(d)  #转换为矩阵格式
# z=sch.linkage(d); print(z) #进行聚类并显示
# s=[str(i+1) for i in range(36)]; plt.rc('font',size=10)
# plt.rc('text',usetex=True)
# sch.dendrogram(z,labels=s,leaf_rotation=0,leaf_font_size=10); 
# plt.xlabel('Sample Number')
# plt.ylabel('Euclidean Distance',rotation=90)
# plt.show()  #画聚类图





'''
聚类分析（侵害光）
'''
# import numpy as np
# from sklearn import preprocessing as pp
# import scipy.cluster.hierarchy as sch
# import matplotlib.pyplot as plt
# a=np.loadtxt("Pdata20_10.txt")
# b=pp.minmax_scale(a.T)   #数据规格化
# d = sch.distance.pdist(b,metric='euclidean')  #求对象之间的两两距离向量
# # d = sch.distance.euclidean(b)
# dd = sch.distance.squareform(d)  #转换为矩阵格式
# z=sch.linkage(d); print(z) #进行聚类并显示
# s=[str(i+1) for i in range(81)]; plt.rc('font',size=10)
# plt.rc('text',usetex=True)
# sch.dendrogram(z,labels=s,leaf_rotation=0,leaf_font_size=5)
# plt.xlabel('Sample Number')
# plt.ylabel('Euclidean Distance',rotation=90)
# plt.show()  #画聚类图





'''
差异度分析
'''
# a = 0.7700929979369869
# b = 0.6280301835572776
# c = 0.3526624592584649
# d = 0.4000782670777438

# print(a-b)
# print(a-c)
# print(a-d)






'''
计算眩光度
'''
# x1 = 1.3
# x2 = 26.21
# x3 = 8
# x4 = x1*x2
# x5 = x1*x3
# y = -4.684143622643461+ 89.5295042*x1+ 3.99565818*x2+ 6.58558765*x3 -0.30167616*x4 -25.24637061*x5
# print("眩光度为",y)




'''
计算不同波长的灯的度
'''
# a = 0.52
# b = 0.18
# c = 0.3
# y = 0.142063*a+0.417431*b+0.370015*c
# print('波长侵害为',y)





'''
分析Logistic模型对于参数的灵敏度
'''
# import numpy as np
# from matplotlib.pyplot import *
# x = np.linspace(0,800,200)
# # y1 = np.sin(x)
# y1 = 1.2000000e+03/(1+(1.2000000e+03/26.3-1)*np.exp(-2.3112453e-02*(x-8)))
# y2 = 1.2000000e+03/(1+(1.2000000e+03/26.3-1)*np.exp(-2.5112453e-02*(x-8)))
# y3 = 1.0000000e+03/(1+(1.0000000e+03/26.3-1)*np.exp(-2.5112453e-02*(x-8)))
# y4 = 1.0000000e+03/(1+(1.0000000e+03/20.8-1)*np.exp(-2.3112453e-02*(x-8)))
# y5 = 1.0000000e+03/(1+(1.0000000e+03/20.8-1)*np.exp(-2.5112453e-02*(x-8)))
# y6 = 0.8000000e+03/(1+(0.8000000e+03/20.8-1)*np.exp(-2.5112453e-02*(x-8)))
# rc('font',size=10)
# rc('text',usetex=True)
# # ax1 = subplot(2,2,1)
# # 划为2行2列，使用第一个格子
# plot(x,y1,'r',label='$r=0.0231,w(8)=26.3,Xm=120$',linewidth=1)
# plot(x,y2,'b',label='$r=0.0251,w(8)=26.3,Xm=120$',linewidth=1)
# plot(x,y3,'g',label='$r=0.0251,w(8)=26.3,Xm=100$',linewidth=1)
# plot(x,y4,'k',label='$r=0.0231,w(8)=20.8,Xm=100$',linewidth=1)
# plot(x,y5,'m',label='$r=0.0251,w(8)=20.8,Xm=100$',linewidth=1)
# plot(x,y6,'y',label='$r=0.0251,w(8)=20.8,Xm=80$',linewidth=1)
# legend()
# xlabel('$Time/year$')
# ylabel('$W(t)/g$',rotation=90)

# # legend()
# # ax2 = subplot(2,2,2)
# # # 划为2行2列，使用第二个格子
# # ax2.plot(x,y2,'b--',label='$cos(x)$')
# # legend()
# # ax3 = subplot(2,1,2)
# # # 划为2行1列，使用第二个格子
# # ax3.plot(x,y3,'k--',label='$sin(x^2)$')
# # legend()
# savefig('figure20_11.png',dpi=500)
# show()



'''
岭回归模型灵敏度分析
'''
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_2.txt")   
# md=LinearRegression().fit(a[:,:3],a[:,3])    #构建并拟合模型
# y=md.predict(a[:,:3])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:3],a[:,3])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f%10.4f"%(b0,b12[0],b12[1],b12[2]))
# print("拟合优度R^2=%.4f"%R2)

# x=np.arange(1,35,1)
# plt.rc('font',size=16); plt.rc('font',family='SimHei')
# plt.plot(x,a[:,3],'r*',label="原始数据")
# plt.plot(x,62.9707+11.5108*a[:,0]+2.2955*a[:,1]-8.5552*a[:,2],'g',label="拟合曲线")
# plt.legend()
# plt.savefig("figure20_2.png")
# plt.show()

# print("相关系数矩阵:\n",np.corrcoef(a[:,:3].T))
# for k in range(0,34):
#     print(a[k,0]**a[k,1])


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_4.txt")   
# md=LinearRegression().fit(a[:,:4],a[:,4])    #构建并拟合模型
# y=md.predict(a[:,:4])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:4],a[:,4])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f%10.4f%10.4f"%(b0,b12[0],b12[1],b12[2],b12[3]))
# print("拟合优度R^2=%.4f"%R2)

# x=np.arange(1,35,1)
# plt.rc('font',size=16); plt.rc('font',family='SimHei')
# plt.plot(x,a[:,3],'r*',label="原始数据")
# plt.plot(x,129.2294+67.6860*a[:,0]-5.6876*a[:,1]-10.1205*a[:,2]+2.1722*a[:,3],'g',label="拟合曲线")
# plt.legend()
# plt.savefig("figure20_2.png")
# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_3.txt")   
# md=LinearRegression().fit(a[:,:5],a[:,5])    #构建并拟合模型
# y=md.predict(a[:,:5])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:5],a[:,5])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f%10.4f%10.4f%10.4f"%(b0,b12[0],b12[1],b12[2],b12[3],b12[4]))
# print("拟合优度R^2=%.4f"%R2)

# x=np.arange(1,35,1)
# plt.rc('font',size=10); plt.rc('text',usetex=True)
# # plt.rc('font',size=16); plt.rc('font',family='SimHei')
# plt.plot(x,a[:,3],'r*',label="raw data")
# plt.plot(x,68.0151013613946+37.67543191*a[:,0]+1.71722566*a[:,1]-2.04682819*a[:,2]+0.63541866*a[:,3]-14.29585552*a[:,4],'g',label="curve with noise")

# plt.plot(x,-4.684143622643461+89.5295042*a[:,0]+3.99565818*a[:,1]+6.58558765*a[:,2]+0.30167616*a[:,3]-25.24637061*a[:,4],'b',label="right curve")
# plt.legend()
# plt.xlabel('Sample Number')
# plt.ylabel('CBC',rotation=90)
# plt.savefig("figure20_12.png")
# plt.show()
# print("相关系数矩阵:\n",np.corrcoef(a[:,:5].T))

