'''
GM(1,1)模型预测，适用于指数规律
'''
# import numpy as np
# import sympy as sp
# from matplotlib.pyplot import plot,show,rc,legend,xticks
# rc('font',size=16); rc('font',family='SimHei')
# x0=np.array([25723,30379,34473,38485,40514,42400,48337])
# n=len(x0); jibi=x0[:-1]/x0[1:]  #求级比
# bd1=[jibi.min(),jibi.max()]    #求级比范围
# bd2=[np.exp(-2/(n+1)),np.exp(2/(n+1))]   #q求级比的容许范围
# x1=np.cumsum(x0)  #求累加序列
# z=(x1[:-1]+x1[1:])/2.0
# B=np.vstack([-z,np.ones(n-1)]).T
# u=np.linalg.pinv(B)@x0[1:] #最小二乘法拟合参数

# sp.var('t'); sp.var('x',cls=sp.Function)  #定义符号变量和函数
# eq=x(t).diff(t)+u[0]*x(t)-u[1]  #定义符号微分方程
# xt=sp.dsolve(eq,ics={x(0):x0[0]})  #求解符号微分方程
# xt=xt.args[1]  #提取方程中的符号解
# xt=sp.lambdify(t,xt,'numpy')  #转换为匿名函数
# t=np.arange(n+1)
# xt1=xt(t)  #求模型的预测值 
# x0_pred=np.hstack([x0[0],np.diff(xt1)]) #还原数据
# x2002=x0_pred[-1]  #提取2002年的预测值
# cha=x0-x0_pred[:-1]; delta=np.abs(cha/x0)*100
# print('1995~2002的预测值：',x0_pred)
# print('\n-------------------\n','相对误差',delta)
# t0=np.arange(1995,2002); plot(t0,x0,'*--')
# plot(t0,x0_pred[:-1],'s-'); legend(('实际值','预测值'));
# xticks(np.arange(1995,2002)); show()



'''
GM(1,N)预测
'''
# import numpy as np
# from scipy.integrate import odeint
# a=np.loadtxt("Pdata15_3.txt")  #加载表中的后4列数据
# n=a.shape[0]  #观测数据的个数
# x10=a[:,0]; x20=a[:,1]; x30=a[:,2]; x40=a[:,3]
# x11=np.cumsum(x10); x21=np.cumsum(x20)
# x31=np.cumsum(x30); x41=np.cumsum(x40)
# z1=(x11[:-1]+x11[1:])/2.; z2=(x21[:-1]+x21[1:])/2.
# z3=(x31[:-1]+x31[1:])/2.; z4=(x41[:-1]+x41[1:])/2.
# B1=np.c_[z1,np.ones((n-1,1))]
# u1=np.linalg.pinv(B1).dot(x10[1:]); print(u1)
# B2=np.c_[z1,z2]
# u2=np.linalg.pinv(B2).dot(x20[1:]); print(u2)
# B3=np.c_[z3,np.ones((n-1,1))];
# u3=np.linalg.pinv(B3).dot(x30[1:]); print(u3)
# B4=np.c_[z1,z3,z4]
# u4=np.linalg.pinv(B4).dot(x40[1:]); print(u4)
# def Pfun(x,t):
#     x1, x2, x3, x4 = x;
#     return np.array([u1[0]*x1+u1[1], u2[0]*x1+u2[1]*x2,
#            u3[0]*x3+u3[1], u4[0]*x1+u4[1]*x3+u4[2]*x4])
# t=np.arange(0, 14);
# X0=np.array([7.1230,0.7960,13.1080,27.475])
# s1=odeint(Pfun, X0, t); s2=np.diff(s1,axis=0)
# xh=np.vstack([X0,s2])
# cha=a-xh[:-1,:]  #计算残差
# delta=np.abs(cha/a)  #计算相对误差
# maxd=delta.max(0)  #计算每个指标的最大相对误差
# pre=xh[-1,:]; print("最大相对误差：",maxd,"\n预测值为：",pre)


'''
GM(2,1)模型预测，适用于S型曲线
'''
# import numpy as np
# import sympy as sp
# x0=np.array([41,49,61,78,96,104])
# n=len(x0)
# lamda=x0[:-1]/x0[1:]  #计算级比
# rang=[lamda.min(), lamda.max()]  #计算级比的范围
# theta=[np.exp(-2/(n+1)),np.exp(2/(n+1))] #计算级比容许范围
# x1=np.cumsum(x0)  #累加运算
# z=0.5*(x1[1:]+x1[:-1])
# B=np.vstack([-x0[1:],-z,np.ones(n-1)]).T
# u=np.linalg.pinv(B)@np.diff(x0)  #最小二乘法拟合参数
# print("参数u：",u)
# sp.var('t'); sp.var('x',cls=sp.Function)  #定义符号变量和函数
# eq=x(t).diff(t,2)+u[0]*x(t).diff(t)+u[1]*x(t)-u[2]
# s=sp.dsolve(eq,ics={x(0):x0[0],x(5):x1[-1]})  #求微分方程符号解
# xt=s.args[1]  #提取解的符号表达式
# print('xt=',xt)
# fxt=sp.lambdify(t,xt,'numpy')  #转换为匿名函数
# yuce1=fxt(np.arange(n))  #求预测值
# yuce=np.hstack([x0[0],np.diff(yuce1)])  #还原数据
# epsilon=x0-yuce[:n]  #计算已知数据预测的残差
# delta=abs(epsilon/x0)  #计算相对误差
# print('相对误差：',np.round(delta*100,2))  #显示相对误差





