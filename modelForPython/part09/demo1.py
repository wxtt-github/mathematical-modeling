'''
获得评价矩阵
'''
# import numpy as np
# import pandas as pd
# a=np.loadtxt("Pdata9_1_1.txt",)
# R1=a.copy(); R2=a.copy(); R3=a.copy()  #初始化
# #注意R1=a,它们的内存地址一样，R1改变时，a也改变
# for j in [0,1,2,4,5]:
#     R1[:,j]=R1[:,j]/np.linalg.norm(R1[:,j]) #向量归一化
#     R2[:,j]=R1[:,j]/max(R1[:,j])     #比例变换
#     R3[:,j]=(R3[:,j]-min(R3[:,j]))/(max(R3[:,j])-min(R3[:,j]));
# R1[:,3]=1-R1[:,3]/np.linalg.norm(R1[:,3])
# R2[:,3]=min(R2[:,3])/R2[:,3]
# R3[:,3]=(max(R3[:,3])-R3[:,3])/(max(R3[:,3])-min(R3[:,3]))
# np.savetxt("Pdata9_1_2.txt", R1); #把数据写入文本文件，供下面使用
# np.savetxt("Pdata9_1_3.txt", R2); np.savetxt("Pdata9_1_4.txt", R3)
# DR1=pd.DataFrame(R1)  #生成DataFrame类型数据
# DR2=pd.DataFrame(R2); DR3=pd.DataFrame(R3)
# f=pd.ExcelWriter('Pdata9_1_5.xlsx')  #创建文件对象
# DR1.to_excel(f,"sheet1")  #把DR1写入Excel文件1号表单中,方便做表
# DR2.to_excel(f,"sheet2"); DR3.to_excel(f, "Sheet3"); f.save()




'''
分别为
TOPSIS法
灰色关联度评价
熵值法
利用秩和比法进行综合评价
'''
import numpy as np
from scipy.stats import rankdata
a=np.loadtxt("Pdata9_1_3.txt")

cplus=a.max(axis=0)   #逐列求最大值
cminus=a.min(axis=0)  #逐列求最小值
print("正理想解=",cplus,"负理想解=",cminus)
d1=np.linalg.norm(a-cplus, axis=1)  #求到正理想解的距离
d2=np.linalg.norm(a-cminus, axis=1) #求到负理想解的距离
print(d1, d2)   #显示到正理想解和负理想解的距离
f1=d2/(d1+d2); print("TOPSIS的评价值为：", f1)

t=cplus-a   #计算参考序列与每个序列的差
mmin=t.min(); mmax=t.max()  #计算最小差和最大差
rho=0.5  #分辨系数
xs=(mmin+rho*mmax)/(t+rho*mmax)  #计算灰色关联系数
f2=xs.mean(axis=1)  #求每一行的均值
print("\n关联系数=", xs,'\n关联度=',f2)  #显示灰色关联系数和灰色关联度

[n, m]=a.shape
cs=a.sum(axis=0)  #逐列求和
P=1/cs*a   #求特征比重矩阵
e=-(P*np.log(P)).sum(axis=0)/np.log(n)  #计算熵值
g=1-e   #计算差异系数
w = g / sum(g)  #计算权重
F = P @ w       #计算各对象的评价值
print("\nP={}\n,e={}\n,g={}\n,w={}\nF={}".format(P,e,g,w,F))

R=[rankdata(a[:,i]) for i in np.arange(6)]  #求每一列的秩
R=np.array(R).T   #构造秩矩阵
print("\n秩矩阵为：\n",R)
RSR=R.mean(axis=1)/n; print("RSR=", RSR)



    



