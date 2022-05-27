import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
df_data

from sklearn.model_selection import train_test_split
X = df_data.drop(labels=['Species'],axis=1).values # 移除Species並取得剩下欄位資料
y = df_data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print('train shape:', X_train.shape)
print('test shape:', X_test.shape)

import autosklearn.classification
automlclassifierV1 = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=180,
    per_run_time_limit=40,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5}
)
automlclassifierV1.fit(X_train, y_train)

# 預測成功的比例
print('automlclassifierV1 訓練集: ',automlclassifierV1.score(X_train,y_train))
print('automlclassifierV1 測試集: ',automlclassifierV1.score(X_test,y_test))

# 查看模型參數
df_cv_results = pd.DataFrame(automlclassifierV1.cv_results_).sort_values(by = 'mean_test_score', ascending = False)
df_cv_results

# 模型聚合結果
automlclassifierV1.leaderboard(detailed = True, ensemble_only=True)

from autosklearn.experimental.askl2 import AutoSklearn2Classifier

automlclassifierV2 = AutoSklearn2Classifier(time_left_for_this_task=180, per_run_time_limit=40)
automlclassifierV2.fit(X_train, y_train)

# 預測成功的比例
print('automlclassifierV2 訓練集: ',automlclassifierV2.score(X_train,y_train))
print('automlclassifierV2 測試集: ',automlclassifierV2.score(X_test,y_test))

# 建立測試集的 DataFrme
df_test=pd.DataFrame(X_test, columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
df_test['Species'] = y_test
pred = automlclassifierV2.predict(X_test)
df_test['Predict'] = pred

sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", hue='Species', data=df_test, fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.show()

sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=df_test, hue="Predict", fit_reg=False, legend=False)
plt.legend(title='target', loc='upper left', labels=['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica'])
plt.show()

automlclassifierV2.leaderboard(detailed = True, ensemble_only=True)

from joblib import dump, load

# 匯出模型
dump(automlclassifierV2, 'model.joblib') 

# 匯入模型
clf = load('model.joblib') 

# 模型預測測試
clf.predict(X_test)

import PipelineProfiler

profiler_data= PipelineProfiler.import_autosklearn(automlclassifierV2)
PipelineProfiler.plot_pipeline_matrix(profiler_data)