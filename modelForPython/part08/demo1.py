"""
求解微分方程表达式
"""
# from sympy.abc import x
# from sympy import diff, dsolve, simplify, Function
# y=Function('y')
# eq=diff(y(x),x,2)+2*diff(y(x),x)+2*y(x)  #定义方程
# con={y(0): 0, diff(y(x),x).subs(x,0): 1}  #定义初值条件
# y=dsolve(eq, ics=con)
# print(simplify(y))
# # 输出结果为Eq(y(x), exp(-x)*sin(x))，即y(x) = exp(-x)*sin(x)


"""
求微分方程离散的数学解
"""
# from scipy.integrate import odeint
# from numpy import arange
# dy=lambda y, x: -2*y+x**2+2*x
# x=arange(1, 10.5, 0.5)
# sol=odeint(dy, 2, x)
# print("x={}\n对应的数值解y={}".format(x, sol.T))


'''
将二阶微分方程降阶，画出符号和数值解的曲线
'''
# from scipy.integrate import odeint
# from sympy.abc import t
# import numpy as np
# import matplotlib.pyplot as plt
# def Pfun(y,x):
#     y1, y2=y;
#     return np.array([y2, -2*y1-2*y2])
# x=np.arange(0, 10, 0.1)  #创建时间点
# sol1=odeint(Pfun, [0.0, 1.0], x)  #求数值解
# plt.rc('font',size=16); plt.rc('font',family='SimHei')
# plt.plot(x, sol1[:,0],'r*',label="数值解")
# plt.plot(x, np.exp(-x)*np.sin(x), 'g', label="符号解曲线")
# plt.legend(); plt.savefig("figure8_5.png"); plt.show()


'''
Lorenz模型,参数为Σ，ρ，β，分别是普兰特数，瑞利数，与几何形状有关的系数
'''
# from scipy.integrate import odeint
# import numpy as np
# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt
# def lorenz(w,t):
#     sigma=10; rho=28; beta=8/3
#     x, y, z=w;
#     return np.array([sigma*(y-x), rho*x-y-x*z, x*y-beta*z])
# t=np.arange(0, 50, 0.01)  #创建时间点
# sol1=odeint(lorenz, [0.0, 1.0, 0.0], t)  #第一个初值问题求解
# sol2=odeint(lorenz, [0.0, 1.0001, 0.0], t)  #第二个初值问题求解
# plt.rc('font',size=16); plt.rc('text',usetex=True)
# ax1=plt.subplot(121,projection='3d')
# ax1.plot(sol1[:,0], sol1[:,1], sol1[:,2],'r')
# ax1.set_xlabel('$x$'); ax1.set_ylabel('$y$'); ax1.set_zlabel('$z$')
# ax2=plt.subplot(122,projection='3d')
# ax2.plot(sol1[:,0]-sol2[:,0], sol1[:,1]-sol2[:,1], sol1[:,2]-sol2[:,2],'g')
# ax2.set_xlabel('$x$'); ax2.set_ylabel('$y$'); ax2.set_zlabel('$z$')
# plt.savefig("figure8_6.png", dpi=500); plt.show()
# print("sol1=",sol1, '\n\n', "sol1-sol2=", sol1-sol2)


'''

'''
# import sympy as sp
# sp.var('t, k')  #定义符号变量t,k
# u = sp.var('u', cls=sp.Function)  #定义符号函数
# eq = sp.diff(u(t), t) + k * (u(t) - 24)  #定义方程
# uu = sp.dsolve(eq, ics={u(0): 150}) #求微分方程的符号解
# print(uu)
# # 求微分方程表达式
# kk = sp.solve(uu, k)  #kk返回值是列表，可能有多个解
# # 求k
# k0 = kk[0].subs({t: 10.0, u(t): 100.0})
# print(kk, '\t', k0)
# u1 = uu.args[1]  #提出符号表达式
# u0 = u1.subs({t: 20, k: k0})  #代入具体值
# print("20分钟后的温度为：", u0)


'''
Logistic模型
'''
import numpy as np
from scipy.optimize import curve_fit
a=[]; b=[];
with open("Pdata8_10_1.txt") as f:    #打开文件并绑定对象f
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
np.savetxt("Pdata8_10_2.txt", c)  #把数据保存起来供下面使用
x=lambda t, r, xm: xm/(1+(xm/3.9-1)*np.exp(-r*(t-1790)))
bd=((0, 200), (0.1,1000))  #约束两个参数的下界和上界
popt, pcov=curve_fit(x, a[1:], b[1:], bounds=bd)
print(popt); print("2010年的预测值为：", x(2010, *popt))



'''
Kermack-Mckendrick模型
'''















