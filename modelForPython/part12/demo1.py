'''
根据x1，x2，x3等自变量确定线性回归模型
y=a0+a1x1+a2x2...

输入为txt文件，格式如下
x1 x2 y
. . .
. . .
当自变量参数大于2时，只需改变a中的边界2即可
'''
# import numpy as np
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata12_1.txt")   #加载表中x1,x2,y的13行3列数据
# md=LinearRegression().fit(a[:,:2],a[:,2])    #构建并拟合模型
# y=md.predict(a[:,:2])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:2],a[:,2])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f"%(b0,b12[0],b12[1]))
# print("拟合优度R^2=%.4f"%R2)


'''
利用statsmodels库确定线性回归方程
输入为txt文件，字典格式
'''
# import numpy as np; import statsmodels.api as sm
# a=np.loadtxt("Pdata12_1.txt")
# #加载表中x1,x2,y的13行3列数据（数据见封底二维码）
# d={'x1':a[:,0],'x2':a[:,1],'y':a[:,2]}
# md=sm.formula.ols('y~x1+x2',d).fit()  #构建并拟合模型
# print(md.summary(),'\n------------\n')  #显示模型所有信息
# ypred=md.predict({'x1':a[:,0],'x2':a[:,1]})  #计算预测值
# print(ypred)  #输出预测值


'''
将txt中x1和x2构造增广矩阵
'''
# import numpy as np; import statsmodels.api as sm
# a=np.loadtxt("Pdata12_1.txt")
# #加载表中x1,x2,y的13行3列数据（数据见封底二维码）
# X = sm.add_constant(a[:,:2])  #增加第一列全部元素为1得到增广矩阵
# md=sm.OLS(a[:,2],X).fit()  #构建并拟合模型
# print(md.params,'\n------------\n')  #提取所有回归系数
# y=md.predict(X)      #求已知自变量值的预测值
# print(y,'\n------------\n')  #输出预测值y
# print(md.summary2())  #输出模型的所有结果


'''
正常求线性回归方程，顺便求自变量相关系数矩阵
'''
# import numpy as np; import statsmodels.api as sm
# a=np.loadtxt("Pdata12_3.txt")   #加载表中x1,x2,x3,y的11行4列数据
# x=a[:,:3]  #提出自变量观测值矩阵
# X=sm.add_constant(x)  #增加第一列全部元素为1得到增广矩阵
# md=sm.OLS(a[:,3],X).fit()  #构建并拟合模型
# b=md.params          #提取所有回归系数
# y=md.predict(X)      #求已知自变量值的预测值
# print(md.summary())  #输出模型的所有结果
# print("相关系数矩阵:\n",np.corrcoef(x.T))
# X1=sm.add_constant(a[:,0])
# md1=sm.OLS(a[:,2],X1).fit()
# print("回归系数为：",md1.params)


'''
岭回归
'''
import numpy as np; import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import zscore
#plt.rc('text', usetex=True)  #没装LaTeX宏包把该句注释
# a=np.loadtxt("Pdata12_3.txt")
a=np.loadtxt("Pdata20_3.txt")
n=a.shape[1]-1  #自变量的总个数
aa=zscore(a)  #数据标准化
x=aa[:,:n]; y=aa[:,n]  #提出自变量和因变量观测值矩阵
b=[]  #用于存储回归系数的空列表
kk=np.logspace(-4,1,100)  #循环迭代的不同k值
for k in kk:
    md=Ridge(alpha=k).fit(x,y)
    b.append(md.coef_)
st=['s-r','*-k','p-b','p-g','*-c']  #下面画图的控制字符串
for i in range(5): plt.plot(kk,np.array(b)[:,i],st[i]);
plt.legend(['$x_1$','$x_2$','$x_3$','$x_4$','$x_5$'],fontsize=15); plt.show()
mdcv=RidgeCV(alphas=np.logspace(-4,1,100)).fit(x,y);
print("最优alpha=",mdcv.alpha_) 
md0=Ridge(mdcv.alpha_).fit(x,y)  #构建并拟合模型
# md0=Ridge(0.4).fit(x,y)  #构建并拟合模型
cs0=md0.coef_  #提出标准化数据的回归系数b1,b2,b3
print("标准化数据的所有回归系数为：",cs0)
mu=np.mean(a,axis=0); s=np.std(a,axis=0,ddof=1) #计算所有指标的均值和标准差
params=[mu[-1]-s[-1]*sum(cs0*mu[:-1]/s[:-1]),s[-1]*cs0/s[:-1]] 
print("原数据的回归系数为：",params)
print("拟合优度：",md0.score(x,y))


'''
LASSO回归
'''
# import numpy as np; import matplotlib.pyplot as plt
# from sklearn.linear_model import Lasso, LassoCV
# from scipy.stats import zscore
# plt.rc('font',size=16)
# plt.rc('text', usetex=True)  #没装LaTeX宏包把该句注释
# # a=np.loadtxt("Pdata12_3.txt")
# a=np.loadtxt("Pdata20_2.txt")
# n=a.shape[1]-1  #自变量的总个数
# aa=zscore(a)  #数据标准化
# x=aa[:,:n]; y=aa[:,n]  #提出自变量和因变量观测值矩阵
# b=[]  #用于存储回归系数的空列表
# kk=np.logspace(-4,0,100)  #循环迭代的不同k值
# for k in kk:
#     md=Lasso(alpha=k).fit(x,y)
#     b.append(md.coef_)
# st=['s-r','*-k','p-b']  #下面画图的控制字符串
# for i in range(3): plt.plot(kk,np.array(b)[:,i],st[i]);
# plt.legend(['$x_1$','$x_2$','$x_3$'],fontsize=15); plt.show()
# mdcv=LassoCV(alphas=np.logspace(-4,0,100)).fit(x,y);
# print("最优alpha=",mdcv.alpha_) 
# #md0=Lasso(mdcv.alpha_).fit(x,y)  #构建并拟合模型
# md0=Lasso(0.21).fit(x,y)  #构建并拟合模型
# cs0=md0.coef_  #提出标准化数据的回归系数b1,b2,b3
# print("标准化数据的所有回归系数为：",cs0)
# mu=np.mean(a,axis=0); s=np.std(a,axis=0,ddof=1) #计算所有指标的均值和标准差
# params=[mu[-1]-s[-1]*sum(cs0*mu[:-1]/s[:-1]),s[-1]*cs0/s[:-1]] 
# print("原数据的回归系数为：",params)
# print("拟合优度：",md0.score(x,y))


'''
解决实际问题的处理方式
'''
# import numpy as np; import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from sklearn.linear_model import Lasso
# from scipy.stats import zscore
# #plt.rc('text', usetex=True)  #没装LaTeX宏包把该句注释
# a=np.loadtxt("Pdata12_6.txt")  #加载表中的9行5列数据
# n=a.shape[1]-1  #自变量的总个数
# x=a[:,:n]  #提出自变量观测值矩阵
# X = sm.add_constant(x)
# md=sm.OLS(a[:,n],X).fit()  #构建并拟合模型
# print(md.summary())  #输出模型的所有结果

# aa=zscore(a)  #数据标准化
# x=aa[:,:n]; y=aa[:,n]  #提出自变量和因变量观测值矩阵
# b=[]  #用于存储回归系数的空列表
# kk=np.logspace(-4,0,100)  #循环迭代的不同k值
# for k in kk:
#     md=Lasso(alpha=k).fit(x,y)
#     b.append(md.coef_)
# st=['s-r','*-k','p-b','^-y']  #下面画图的控制字符串
# for i in range(n): plt.plot(kk,np.array(b)[:,i],st[i]);
# plt.legend(['$x_1$','$x_2$','$x_3$','$x_4$'],fontsize=15); plt.show()
# md0=Lasso(0.05).fit(x,y)  #构建并拟合模型
# cs0=md0.coef_  #提出标准化数据的回归系数b1,b2,b3,b4
# print("标准化数据的所有回归系数为：",cs0)
# mu=a.mean(axis=0); s=a.std(axis=0,ddof=1) #计算所有指标的均值和标准差
# params=[mu[-1]-s[-1]*sum(cs0*mu[:-1]/s[:-1]),s[-1]*cs0/s[:-1]] 
# print("原数据的回归系数为：",params)
# print("拟合优度：",md0.score(x,y))


'''
Logistic回归
'''
# import numpy as np
# import statsmodels.api as sm
# a=np.loadtxt("Pdata12_7_1.txt")   #加载表中x,ni,mi的9行3列数据
# x=a[:,0]; pi=a[:,2]/a[:,1]
# X=sm.add_constant(x); yi=np.log(pi/(1-pi))
# md=sm.OLS(yi,X).fit()  #构建并拟合模型
# print(md.summary())  #输出模型的所有结果
# b=md.params  #提出所有的回归系数
# p0=1/(1+np.exp(-np.dot(b,[1,9])))
# print("所求概率p0=%.4f"%p0)
# np.savetxt("Pdata12_7_2.txt", b)  #把回归系数保存到文本文件


'''
Logistic实际运用1
'''
# import numpy as np
# import statsmodels.api as sm
# a=np.loadtxt("Pdata12_9.txt")
# n=a.shape[1] #提取矩阵的列数
# x=a[:,:n-1]; y=a[:,n-1]
# md=sm.Logit(y,x)
# md=md.fit(method="bfgs")  #这里必须使用bfgs方法，使用默认牛顿方法出错
# print(md.params,'\n----------\n'); print(md.summary2())
# print(md.predict([[-49.2,-17.2,0.3],[40.6,26.4,1.8]]))  #求预测值

'''
Logistic实际运用2
'''
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# a=np.loadtxt("Pdata12_9.txt")
# n=a.shape[1]  #提取矩阵的列数
# x=a[:,:n-1]; y=a[:,n-1]
# md=LogisticRegression(solver='lbfgs')
# md=md.fit(x,y)
# print(md.intercept_,md.coef_)
# print(md.predict(x))   #检验预测模型
# print(md.predict([[-49.2,-17.2,0.3],[40.6,26.4,1.8]]))  #求预测值









# '''
# 分析几种LED光对于LED屏幕亮度的影响
# 目的得出一个线性方程
# 总共有6个因子，红黄白蓝绿灰，对于光照强度的影响
# '''

# import numpy as np
# from sklearn.linear_model import LinearRegression
# a=np.loadtxt("Pdata20_1.txt")   #加载表中x1,x2,x3,x4,x5,x6,y的13行7列数据
# md=LinearRegression().fit(a[:,:6],a[:,6])    #构建并拟合模型
# y=md.predict(a[:,:6])       #求预测值
# b0=md.intercept_; b12=md.coef_   #输出回归系数
# R2=md.score(a[:,:6],a[:,6])      #计算R^2
# print("b0=%.4f\nb12=%.4f%10.4f"%(b0,b12[0],b12[1]),b12[2]),b12[3]),b12[4]),b12[5]))
# print("拟合优度R^2=%.4f"%R2)

