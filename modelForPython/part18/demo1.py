'''
一次移动平均法预测
'''
# import numpy as np
# y=np.array([423,358,434,445,527,429,426,502,480,384,427,446])
# def MoveAverage(y,N):
#     Mt=['*']*N
#     for i in range(N+1,len(y)+2):
#         M=y[i-(N+1):i-1].mean()
#         Mt.append(M)
#     return Mt
# yt3=MoveAverage(y,3) 
# s3=np.sqrt(((y[3:]-yt3[3:-1])**2).mean())
# yt5=MoveAverage(y,5)
# s5=np.sqrt(((y[5:]-yt5[5:-1])**2).mean())
# print('N=3时,预测值：',yt3,'，预测的标准误差：',s3)
# print('N=5时,预测值：',yt5,'，预测的标准误差：',s5)


'''
一次指数平滑法预测
'''
# import numpy as np
# import pandas as pd
# y=np.array([4.81,4.8,4.73,4.7,4.7,4.73,4.75,4.75,5.43,5.78,5.85])
# def ExpMove(y,a):
#     n=len(y); M=np.zeros(n); M[0]=(y[0]+y[1])/2;
#     for i in range(1,len(y)):
#         M[i]=a*y[i-1]+(1-a)*M[i-1]
#     return M
# yt1=ExpMove(y,0.2); yt2=ExpMove(y,0.5)
# yt3=ExpMove(y,0.8); s1=np.sqrt(((y-yt1)**2).mean())
# s2=np.sqrt(((y-yt2)**2).mean())
# s3=np.sqrt(((y-yt3)**2).mean())
# d=pd.DataFrame(np.c_[yt1,yt2,yt3])
# f=pd.ExcelWriter("Pdata18_2.xlsx");
# d.to_excel(f); f.close()  #数据写入Excel文件，便于做表
# print("预测的标准误差分别为：",s1,s2,s3)  #输出预测的标准误差
# yh=0.8*y[-1]+0.2*yt3[-1]
# print("下一期的预测值为：",yh)


'''
二次指数平滑法预测
'''
# import numpy as np
# import pandas as pd
# y=np.loadtxt('Pdata18_3.txt')
# n=len(y); alpha=0.3; yh=np.zeros(n)
# s1=np.zeros(n); s2=np.zeros(n)
# s1[0]=y[0]; s2[0]=y[0]
# for i in range(1,n):
#     s1[i]=alpha*y[i]+(1-alpha)*s1[i-1]
#     s2[i]=alpha*s1[i]+(1-alpha)*s2[i-1];
#     yh[i]=2*s1[i-1]-s2[i-1]+alpha/(1-alpha)*(s1[i-1]-s2[i-1])
# at=2*s1[-1]-s2[-1]; bt=alpha/(1-alpha)*(s1[-1]-s2[-1])
# m=np.array([1,2])
# yh2=at+bt*m
# print("预测值为：",yh2)
# d=pd.DataFrame(np.c_[s1,s2,yh])
# f=pd.ExcelWriter("Pdata18_3.xlsx");
# d.to_excel(f); f.close()



'''
季节性时间序列预测
'''
# import numpy as np
# a=np.loadtxt('Pdata18_4.txt')
# m,n=a.shape
# amean=a.mean()  #计算所有数据的平均值
# cmean=a.mean(axis=0)   #逐列求均值
# r=cmean/amean   #计算季节系数
# w=np.arange(1,m+1)
# yh=w.dot(a.sum(axis=1))/w.sum()  #计算下一年的预测值
# yj=yh/n   #计算预测年份的季度平均值
# yjh=yj*r  #计算季度预测值
# print("下一年度各季度的预测值为：",yjh)



'''
ARMA模型预测
'''
# import pandas as pd, numpy as np
# import statsmodels.api as sm
# import matplotlib.pylab as plt
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plt.rc('axes',unicode_minus=False)
# plt.rc('font',family='SimHei'); plt.rc('font',size=16)
# d=pd.read_csv('sunspots.csv'); dd=d['counts']
# years=d['year'].values.astype(int)
# plt.plot(years,dd.values,'-*'); plt.figure()
# ax1=plt.subplot(121); plot_acf(dd,ax=ax1,title='自相关')
# ax2=plt.subplot(122); plot_pacf(dd,ax=ax2,title='偏自相关')

# for i in range(1,6):
#     for j in range(1,6):
#         md=sm.tsa.ARMA(dd,(i,j)).fit()
#         print([i,j,md.aic,md.bic])
# zmd=sm.tsa.ARMA(dd,(4,2)).fit()
# print(zmd.summary())  #显示模型的所有信息

# residuals = pd.DataFrame(zmd.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="残差", ax=ax[0])
# residuals.plot(kind='kde', title='密度', ax=ax[1])
# plt.legend(''); plt.ylabel('') 

# dhat=zmd.predict(); plt.figure()
# plt.plot(years[-20:],dd.values[-20:],'o-k')
# plt.plot(years[-20:],dhat.values[-20:],'P--')
# plt.legend(('原始观测值','预测值'))
# dnext=zmd.predict(d.shape[0],d.shape[0])
# print(dnext)  #显示下一期的预测值
# plt.show()



'''
ARIMA
'''
# import pandas as pd
# from statsmodels.graphics.tsaplots import plot_acf
# import pylab as plt
# from statsmodels.tsa.arima_model import ARIMA

# plt.rc('axes',unicode_minus=False)
# plt.rc('font',size=16); plt.rc('font',family='SimHei')
# df=pd.read_csv('austa.csv')
# plt.subplot(121); plt.plot(df.value.diff())
# plt.title('一次差分')
# ax2=plt.subplot(122)
# plot_acf(df.value.diff().dropna(), ax=ax2,title='自相关')

# md=ARIMA(df.value, order=(2,1,0))
# mdf=md.fit(disp=0)
# print(mdf.summary())

# residuals = pd.DataFrame(mdf.resid)
# fig, ax = plt.subplots(1,2)
# residuals.plot(title="残差", ax=ax[0])
# residuals.plot(kind='kde', title='密度', ax=ax[1])
# plt.legend(''); plt.ylabel('')          

# mdf.plot_predict()  #原始数据与预测值对比图
# plt.show()




'''
已实现的ARIMA模型
'''
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
import numpy as np
import seaborn as sns


ChinaBank = pd.read_csv('ChinaBank.csv',index_col = 'Date',parse_dates=['Date'])
ChinaBank.head()
# 导入数据
ChinaBank.index = pd.to_datetime(ChinaBank.index)
sub = ChinaBank.loc['2014-01':'2014-06','Close']
sub.head()
# 提取Close列
train = sub.loc['2014-01':'2014-03']
test = sub.loc['2014-04':'2014-06']

plt.figure(figsize=(12,6))
plt.plot(train)
plt.xticks(rotation=45)
plt.show()
# 划分训练测试集

ChinaBank['diff_1'] = ChinaBank['Close'].diff(1)
ChinaBank['diff_2'] = ChinaBank['diff_1'].diff(1)
# 做二阶差分

fig = plt.figure(figsize=(12,10))
#原数据
ax1 = fig.add_subplot(311)
ax1.plot(ChinaBank['Close'])
#1阶差分
ax2 = fig.add_subplot(312)
ax2.plot(ChinaBank['diff_1'])
#2阶差分
ax3 = fig.add_subplot(313)
ax3.plot(ChinaBank['diff_2'])
plt.show()
# 差分法


#绘制
fig = plt.figure(figsize=(12,7))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom') # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部
#fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
#fig.tight_layout()
plt.show()
# 参数确定



#确定pq的取值范围
p_min = 0
d_min = 0
q_min = 0
p_max = 5
d_max = 0
q_max = 5

#Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(train, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue

#得到结果后进行浮点型转换
results_bic = results_bic[results_bic.columns].astype(float)

#绘制热力图
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 cmap="Purples"
                 )

ax.set_title('BIC')
plt.show()

results_bic.stack().idxmin()
# 模型建立


train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)
# 利用模型取p和q的最优值


#根据以上求得
p = 1
d = 0
q = 0

model = sm.tsa.ARIMA(train, order=(p,d,q))
results = model.fit()
resid = results.resid #获取残差

#绘制
#查看测试集的时间序列与数据(只包含测试集)
fig, ax = plt.subplots(figsize=(12, 5))

ax = sm.graphics.tsa.plot_acf(resid, lags=40,ax=ax)

plt.show()
# 模型检验



predict_sunspots = results.predict(dynamic=False)
print(predict_sunspots)

#查看测试集的时间序列与数据(只包含测试集)
plt.figure(figsize=(12,6))
plt.plot(train)
plt.xticks(rotation=45) #旋转45度
plt.plot(predict_sunspots)
plt.show()

#绘图
fig, ax = plt.subplots(figsize=(12, 6))
ax = sub.plot(ax=ax)
#预测数据
predict_sunspots.plot(ax=ax)
plt.show()
# 模型预测








