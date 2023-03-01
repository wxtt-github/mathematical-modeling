'''
判别分析,基于协方差
根据马氏距离和欧氏距离分类,一般是倾向使用马氏距离
输入值为给定类别的数据,待观测值,输出预测类别
'''
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# x0=np.array([[1.24,1.27], [1.36,1.74], [1.38,1.64], [1.38,1.82], [1.38,1.90], [1.40,1.70],
#     [1.48,1.82], [1.54,1.82], [1.56,2.08], [1.14,1.78], [1.18,1.96], [1.20,1.86],
#     [1.26,2.00], [1.28,2.00], [1.30,1.96]])   #输入已知样本数据
# x=np.array([[1.24,1.80], [1.28,1.84], [1.40,2.04]])  #输入待判样本点数据
# g=np.hstack([np.ones(9),2*np.ones(6)])  #g为已知样本数据的类别标号
# v=np.cov(x0.T)  #计算协方差
# knn=KNeighborsClassifier(2,metric='mahalanobis',metric_params={'V': v}) #马氏距离分类
# knn.fit(x0,g); pre=knn.predict(x); print("马氏距离分类结果：",pre)
# print("马氏距离已知样本的误判率为：",1-knn.score(x0,g))
# knn2=KNeighborsClassifier(2)  #欧氏距离分类
# knn2.fit(x0,g); pre2=knn2.predict(x); print("欧氏距离分类结果：",pre2)
# print("欧氏距离已知样本的误判率为：",1-knn2.score(x0,g))


'''
马氏距离分类,误判率为15%
'''
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# a=pd.read_excel("Pdata11_2.xlsx",header=None)
# b=a.values
# x0=b[:-2,1:-1].astype(float)  #提取已知样本点的观测值
# y0=b[:-2,-1].astype(int)
# x=b[-2:,1:-1]  #提取待判样本点的观察值
# v=np.cov(x0.T)  #计算协方差
# knn=KNeighborsClassifier(3,metric='mahalanobis',metric_params={'V': v}) #马氏距离分类
# knn.fit(x0,y0); pre=knn.predict(x); print("分类结果：",pre)
# print("已知样本的误判率为：",1-knn.score(x0,y0))


'''
Fisher判别法,基于方差分析,分析昆虫类别题
'''
# import numpy as np
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# x0=np.array([[1.24,1.27], [1.36,1.74], [1.38,1.64], [1.38,1.82], [1.38,1.90], [1.40,1.70],
#     [1.48,1.82], [1.54,1.82], [1.56,2.08], [1.14,1.78], [1.18,1.96], [1.20,1.86],
#     [1.26,2.00], [1.28,2.00], [1.30,1.96]])   #输入已知样本数据
# x=np.array([[1.24,1.80], [1.28,1.84], [1.40,2.04]])  #输入待判样本点数据
# y0=np.hstack([np.ones(9),2*np.ones(6)])  #y0为已知样本数据的类别
# clf = LDA()
# clf.fit(x0, y0)
# print("判别结果为：",clf.predict(x))
# print("已知样本的误判率为：",1-clf.score(x0,y0))


'''
Fisher判别法,基于方差分析,分析心脏病类别题
在该问题中,Fisher判别法比马氏判别法表现的好,误判率为0
'''
# import numpy as np
# import pandas as pd
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# a=pd.read_excel("Pdata11_2.xlsx",header=None)
# b=a.values
# x0=b[:-2,1:-1].astype(float)  #提取已知样本点的观测值
# y0=b[:-2,-1].astype(int)
# x=b[-2:,1:-1]  #提取待判样本点的观察值
# clf = LDA()
# clf.fit(x0, y0)
# print("判别结果为：",clf.predict(x))
# print("已知样本的误判率为：",1-clf.score(x0,y0))


'''
贝叶斯判别法,分析昆虫问题
'''
# import numpy as np
# from sklearn.naive_bayes import GaussianNB
# x0=np.array([[1.24,1.27], [1.36,1.74], [1.38,1.64], [1.38,1.82], [1.38,1.90], [1.40,1.70],
#     [1.48,1.82], [1.54,1.82], [1.56,2.08], [1.14,1.78], [1.18,1.96], [1.20,1.86],
#     [1.26,2.00], [1.28,2.00], [1.30,1.96]])   #输入已知样本数据
# x=np.array([[1.24,1.80], [1.28,1.84], [1.40,2.04]])  #输入待判样本点数据
# y0=np.hstack([np.ones(9),2*np.ones(6)])  #y0为已知样本数据的类别
# clf = GaussianNB()
# clf.fit(x0, y0)
# print("判别结果为：",clf.predict(x))
# print("已知样本的误判率为：",1-clf.score(x0,y0))


'''
计算Fisher法分析昆虫问题的交叉验证准确率
'''
# import numpy as np
# import pandas as pd
# from sklearn.discriminant_analysis import \
# LinearDiscriminantAnalysis
# from sklearn.model_selection import cross_val_score
# a=pd.read_excel("Pdata11_2.xlsx",header=None)
# b=a.values
# x0=b[:-2,1:-1].astype(float)  #提取已知样本点的观测值
# y0=b[:-2,-1].astype(int)
# model = LinearDiscriminantAnalysis ()


'''
主成分分析,分别用PCA函数和相关系数矩阵计算主成分
'''
# import numpy as np
# from sklearn.decomposition import PCA
# a=np.loadtxt("Pdata11_7.txt")
# b=np.r_[a[:,1:4],a[:,-3:]]  #构造数据矩阵
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
利用主成分分析计算贡献率,然后进行评价分析
'''
# import numpy as np
# from scipy.stats import zscore
# a=np.loadtxt("Pdata11_8.txt")
# print("相关系数阵为：\n",np.corrcoef(a.T))
# b=np.delete(a,0,axis=1) #删除第1列数据
# c=zscore(b); r=np.corrcoef(c.T) #数据标准化并计算相关系数阵
# d,e=np.linalg.eig(r) #求特征值和特征向量
# rate=d/d.sum()  #计算各主成分的贡献率
# print("特征值为：",d)
# print("特征向量为：\n",e)
# print("各主成分的贡献率为：",rate)
# k=1; #提出主成分的个数
# F=e[:,:k]; score_mat=c.dot(F) #计算主成分得分矩阵
# score1=score_mat.dot(rate[0:k])  #计算各评价对象的得分
# score2=-score1  #通过观测，调整得分的正负号
# print("各评价对象的得分为：",score2) 
# index=score1.argsort()+1   #排序后的每个元素在原数组中的位置
# print("从高到低各个城市的编号排序为：",index)


'''
因子分析
'''
# import numpy as np; import pandas as pd
# from sklearn import decomposition as dc
# from scipy.stats import zscore
# import matplotlib.pyplot as plt
# c=pd.read_excel("Pan11_1_1.xlsx",usecols=np.arange(1,7))
# c=c.values.astype(float)
# d=zscore(c)          #数据标准化
# r=np.corrcoef(d.T)   #求相关系数矩阵
# f=pd.ExcelWriter('Pan11_1_2.xlsx')
# pd.DataFrame(r).to_excel(f); f.save()
# val,vec=np.linalg.eig(r)
# cs=np.cumsum(val)  #求特征值的累加和
# print("特征值为：",val,"\n累加和为：",cs)
# fa = dc.FactorAnalysis(n_components = 2)  #构建模型
# fa.fit(d)   #求解最大方差的模型
# print("载荷矩阵为：\n",fa.components_)
# print("特殊方差为：\n",fa.noise_variance_)
# dd=fa.fit_transform(d)   #计算因子得分
# w=val[:2]/sum(val[:2])  #计算两个因子的权重
# df=np.dot(dd,w)        #计算每个评价对象的因子总分
# tf=np.sum(c,axis=1)     #计算每个评价对象的实分总分
# #构造pandas数据框,第1列到第5列数据分别为因子1得分，因子2得分，因子总分、实分总分和序号
# pdf=pd.DataFrame(np.c_[dd,df,tf,np.arange(1,53)],columns=['f1','f2','yf','tf','xh'])
# spdf1=pdf.sort_values(by='yf',ascending = False)  #y因子总分从高到低排序
# spdf2=pdf.sort_values(by='tf',ascending=False)  #实分总分从高到低排序
# print("排序结果为：\n",spdf1,'\n',spdf2)
# s=['A'+str(i) for i in range(1,53)]
# plt.rc('font',family='SimHei'); plt.rc('axes',unicode_minus=False)
# plt.plot(dd[:,0],dd[:,1],'.')
# for i in range(len(s)): plt.text(dd[i,0],dd[i,1]+0.03,s[i])
# plt.xlabel("基础课因子得分"); plt.ylabel("开闭卷因子得分"); plt.show()


'''
聚类分析
'''
import numpy as np
from sklearn import preprocessing as pp
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
a=np.loadtxt("Pdata11_11.txt")
b=pp.minmax_scale(a.T)   #数据规格化
d = sch.distance.pdist(b)  #求对象之间的两两距离向量
dd = sch.distance.squareform(d)  #转换为矩阵格式
z=sch.linkage(d); print(z) #进行聚类并显示
s=[str(i+1) for i in range(7)]; plt.rc('font',size=16)
sch.dendrogram(z,labels=s); plt.show()  #画聚类图

