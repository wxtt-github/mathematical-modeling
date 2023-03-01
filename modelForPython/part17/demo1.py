'''
退火算法求解最小值问题
'''
# from numpy import loadtxt,radians,sin,cos,inf,exp
# from numpy import array,r_,c_,arange,savetxt
# from numpy.lib.scimath import arccos
# from numpy.random import shuffle,randint,rand
# from matplotlib.pyplot import plot, show, rc
# a=loadtxt("Pdata17_1.txt")
# x=a[:,::2]. flatten(); y=a[:,1::2]. flatten()
# d1=array([[70,40]]); xy=c_[x,y]
# xy=r_[d1,xy,d1]; N=xy.shape[0]
# t=radians(xy)  #转化为弧度
# d=array([[6370*arccos(cos(t[i,0]-t[j,0])*cos(t[i,1])*cos(t[j,1])+
#   sin(t[i,1])*sin(t[j,1])) for i in range(N)]
#        for j in range(N)]).real
# savetxt('Pdata17_2.txt',c_[xy,d])  #把数据保存到文本文件，供下面使用
# path=arange(N); L=inf
# for j in range(1000):
#     path0=arange(1,N-1); shuffle(path0)
#     path0=r_[0,path0,N-1]; L0=d[0,path0[1]]  #初始化
#     for i in range(1,N-1):L0+=d[path0[i],path0[i+1]]
#     if L0<L: path=path0; L=L0
# print(path,'\n',L)        
# e=0.1**30; M=20000; at=0.999; T=1
# for k in range(M):
#     c=randint(1,101,2); c.sort()
#     c1=c[0]; c2=c[1]
#     df=d[path[c1-1],path[c2]]+d[path[c1],path[c2+1]]-\
#     d[path[c1-1],path[c1]]-d[path[c2],path[c2+1]]  #续行
#     if df<0:
#         path=r_[path[0],path[1:c1],path[c2:c1-1:-1],path[c2+1:102]]; L=L+df
#     else:
#         if exp(-df/T)>=rand(1):
#             path=r_[path[0],path[1:c1],path[c2:c1-1:-1],path[c2+1:102]]
#             L=L+df
#     T=T*at
#     if T<e: break
# print(path,'\n',L)  #输出巡航路径及路径长度
# xx=xy[path,0]; yy=xy[path,1]; rc('font',size=16)
# plot(xx,yy,'-*'); show()  #画巡航路径



'''
遗传算法
'''
# import numpy as np
# from numpy.random import randint, rand, shuffle
# from matplotlib.pyplot import plot, show, rc
# a=np.loadtxt("Pdata17_2.txt")
# xy,d=a[:,:2],a[:,2:]; N=len(xy)
# w=50; g=10  #w为种群的个数，g为进化的代数
# J=[]; 
# for i in np.arange(w):
#     c=np.arange(1,N-1); shuffle(c)
#     c1=np.r_[0,c,101]; flag=1
#     while flag>0:
#         flag=0
#         for m in np.arange(1,N-3):
#             for n in np.arange(m+1,N-2):
#                 if d[c1[m],c1[n]]+d[c1[m+1],c1[n+1]]<\
#                    d[c1[m],c1[m+1]]+d[c1[n],c1[n+1]]:
#                     c1[m+1:n+1]=c1[n:m:-1]; flag=1
#     c1[c1]=np.arange(N); J.append(c1)
# J=np.array(J)/(N-1)
# for k in np.arange(g):
#     A=J.copy()
#     c1=np.arange(w); shuffle(c1) #交叉操作的染色体配对组
#     c2=randint(2,100,w)  #交叉点的数据
#     for i in np.arange(0,w,2):
#         temp=A[c1[i],c2[i]:N-1]  #保存中间变量
#         A[c1[i],c2[i]:N-1]=A[c1[i+1],c2[i]:N-1]
#         A[c1[i+1],c2[i]:N-1]=temp
#     B=A.copy()
#     by=[]  #初始化变异染色体的序号
#     while len(by)<1: by=np.where(rand(w)<0.1)
#     by=by[0]; B=B[by,:]
#     G=np.r_[J,A,B]
#     ind=np.argsort(G,axis=1)  #把染色体翻译成0,1，…，101
#     NN=G.shape[0]; L=np.zeros(NN)
#     for j in np.arange(NN):
#         for i in np.arange(101):
#             L[j]=L[j]+d[ind[j,i],ind[j,i+1]]
#     ind2=np.argsort(L)
#     J=G[ind2,:]
# path=ind[ind2[0],:]; zL=L[ind2[0]]
# xx=xy[path,0]; yy=xy[path,1]; rc('font',size=16)
# plot(xx,yy,'-*'); show()  #画巡航路径
# print("所求的巡航路径长度为：",zL)


'''
神经网络感知器分类
'''
# from sklearn.linear_model import Perceptron
# import numpy as np
# x0=np.array([[-0.5,-0.5,0.3,0.0],[-0.5,0.5,-0.5,1.0]]).T
# y0=np.array([1,1,0,0])
# md = Perceptron(tol=1e-3)   #构造模型
# md.fit(x0, y0)              #拟合模型
# print(md.coef_,md.intercept_)  #输出系数和常数项
# print(md.score(x0,y0))   #模型检验
# print("预测值为：",md.predict(np.array([[-0.5,0.2]])))


'''
BP神经网络预测分析
'''
# from sklearn.neural_network import MLPRegressor
# from numpy import array, loadtxt
# from pylab import subplot, plot, show, xticks,rc,legend
# rc('font',size=15); rc('font',family='SimHei')
# a=loadtxt("Pdata17_5.txt"); x0=a[:,:3]; y1=a[:,3]; y2=a[:,4];
# md1=MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=10)
# md1.fit(x0, y1); x=array([[73.39,3.9635,0.988],[75.55,4.0975,1.0268]])
# pred1=md1.predict(x); print(md1.score(x0,y1)); 
# print("客运量的预测值为：",pred1,'\n----------------'); 
# md2=MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=10)
# md2.fit(x0, y2); pred2=md2.predict(x); print(md2.score(x0,y2)); 
# print("货运量的预测值为：",pred2); yr=range(1990,2010)
# subplot(121); plot(yr,y1,'o'); plot(yr,md1.predict(x0),'-*')
# xticks(yr,rotation=55); legend(("原始数据","网络输出客运量"))
# subplot(122); plot(yr,y2,'o'); plot(yr,md2.predict(x0),'-*')
# xticks(yr,rotation=55)
# legend(("原始数据","网络输出货运量"),loc='upper left'); show()








