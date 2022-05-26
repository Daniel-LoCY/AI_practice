from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from sklearn.datasets import load_iris
import random

iris = load_iris()
pd.set_option('display.max_rows', None)
df_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])

X = df_data.drop(labels=['Species'],axis=1).values # 移除Species並取得剩下欄位資料
y = df_data['Species']
# X

from sklearn.cluster import KMeans

ran = random.randint(1,1000)
init = random.randint(1,100)
print(ran, init)

kmeansModel = KMeans(n_clusters=3, random_state=ran, n_init=init)
clusters_pred = kmeansModel.fit_predict(X)

sns.lmplot("PetalLengthCm", "PetalWidthCm", hue='Species', data=df_data, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.show()

df_data['Predict']=clusters_pred
sns.lmplot("PetalLengthCm", "PetalWidthCm", data=df_data, hue="Predict", fit_reg=False, legend=False)
plt.scatter(kmeansModel.cluster_centers_[:, 2], kmeansModel.cluster_centers_[:, 3], s=200,c="r",marker='*')
plt.legend(title='target', loc='upper left', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.show()

print(df_data)

# # k = 1~9 做9次kmeans, 並將每次結果的inertia收集在一個list裡
# kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(X)
#                 for k in range(1, 10)]
# inertias = [model.inertia_ for model in kmeans_list]

# plt.figure(figsize=(8, 3.5))
# plt.plot(range(1, 10), inertias, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Inertia", fontsize=14)
# plt.annotate('Elbow',
#              xy=(3, inertias[3]),
#              xytext=(0.55, 0.55),
#              textcoords='figure fraction',
#              fontsize=16,
#              arrowprops=dict(facecolor='black', shrink=0.1)
#             )
# plt.axis([1, 8.5, 0, 1300])

# plt.show()

# from sklearn.metrics import silhouette_score
# silhouette_scores = [silhouette_score(X, model.labels_)
#                      for model in kmeans_list[1:]]

# plt.figure(figsize=(8, 3))
# plt.plot(range(2, 10), silhouette_scores, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Silhouette score", fontsize=14)

# plt.show()