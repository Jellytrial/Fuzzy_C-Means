@@ -0,0 +1,124 @@
#import pylab
from numpy import*  
import pandas as pd  
import numpy as np  
import operator  
import math  
import matplotlib.pyplot as plt  
import random  

df_full = pd.read_csv("input.csv")
value=df_full.values  
columns = list(df_full.columns)  
features = columns[:len(columns)]  
class_labels = list(df_full[columns[-1]])  
df = df_full[features]  
# Number of Attributes  
num_attr = len(df.columns)  
# Number of Clusters  
k =5  
# Maximum number of iterations  
MAX_ITER = 1000  
# Number of data points  
n = len(df)    #the number of row  
# Fuzzy parameter  
m = 2.0  
  
#initializa fuzzy matrix  
def initializeMembershipMatrix():  
    membership_mat = list()  
    for i in range(n):  
        random_num_list = [random.random() for i in range(k)]  
        summation = sum(random_num_list)  
        temp_list = [x/summation for x in random_num_list]  
        membership_mat.append(temp_list)  
    return membership_mat  
  
#compute cluster center  
def calculateClusterCenter(membership_mat):
    count = 0
    while(count < 100):
        cluster_mem_val = zip(*membership_mat)  
        cluster_centers = list()  
        cluster_mem_val_list = list(cluster_mem_val)  
        for j in range(k):  
            x=cluster_mem_val_list[j]  
            xraised = [e ** m for e in x]  
            denominator = sum(xraised)  
            temp_num = list()  
            for i in range(n):  
                data_point = list(df.iloc[i])  
                prod = [xraised[i] * val for val in data_point]  
                temp_num.append(prod)  
            numerator = map(sum, zip(*temp_num))  
            center = [z/denominator for z in numerator]
            cluster_centers.append(center)
        count = count + 1
    return cluster_centers  
  
#uploade membership  
def updateMembershipValue(membership_mat, cluster_centers):  
    p = float(2/(m-1))  
    data=[]  
    for i in range(n):  
        x = list(df.iloc[i])#取出文件中的每一行数据  
        data.append(x)  
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]  
        for j in range(k):  
            den = sum([math.pow(float(distances[j]/distances[c]), 2) for c in range(k)])  
            membership_mat[i][j] = float(1/den)         
    return membership_mat, data  
  
#getting cluster result 
def getClusters(membership_mat):  
    cluster_labels = list()  
    for i in range(n):  
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))  
        cluster_labels.append(idx)
    return cluster_labels  
  
def fuzzyCMeansClustering():  
    # main function  
    membership_mat = initializeMembershipMatrix()  
    curr = 0  
    while curr <= MAX_ITER:  
        cluster_centers = calculateClusterCenter(membership_mat)  
        membership_mat,data = updateMembershipValue(membership_mat, cluster_centers)  
        cluster_labels = getClusters(membership_mat)  
        curr += 1  
    print(membership_mat)  
    return cluster_labels, cluster_centers, data, membership_mat  
  
def J(membership_mat,center,data):  
    sum_cluster_distance=0  
    min_cluster_center_distance=inf  
    for i in range(k):  
        for j in range(n):  
            sum_cluster_distance=sum_cluster_distance + membership_mat[j][i]** 2 * sum(power(data[j,:]- center[i,:],2))#计算类一致性  
    for i in range(k-1):  
        for j in range(i+1,k):  
            cluster_center_distance=sum(power(center[i,:]-center[j,:],2))  
            if cluster_center_distance<min_cluster_center_distance:  
                min_cluster_center_distance=cluster_center_distance
    return sum_cluster_distance/(n*min_cluster_center_distance)  

labels,centers,data,membership= fuzzyCMeansClustering()  
print(labels)  
print(centers)  
center_array=array(centers)  
label=array(labels)  
datas=array(data)  
  
#J objective function  
print("minimum of objective function: \n",J(membership,center_array,datas))  
plt.xlim((0.25, 0.75))  
plt.ylim((0.25, 0.75))


plt.scatter(datas[nonzero(label==0),0],datas[nonzero(label==0),1],marker='o',color='r',label='1',s=30, alpha=0.6)
plt.scatter(datas[nonzero(label==1),0],datas[nonzero(label==1),1],marker='o',color='b',label='1',s=30, alpha=0.6)  
plt.scatter(datas[nonzero(label==2),0],datas[nonzero(label==2),1],marker='o',color='g',label='2',s=30, alpha=0.6)
plt.scatter(datas[nonzero(label==3),0],datas[nonzero(label==3),1],marker='o',color='m',label='1',s=30, alpha=0.6)
plt.scatter(datas[nonzero(label==4),0],datas[nonzero(label==4),1],marker='o',color='y',label='1',s=30, alpha=0.6)  
plt.scatter(center_array[ :,0],center_array[ :,1],marker = 'x', color = 'm', s = 50)  
plt.show()  
