# 13-3 SVM分類器-找C及gamma

### 由GridSearch找出較好的C及gamma.

#### 參考:http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html


```
專案
 |__ iris.csv
 |__ main.py
```



## 1.讀入檔 iris.csv
```
(從11-1下載)
```




## 2.main.py
```
#--------------------------------
# 匯入外部模組
#--------------------------------
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#-------------------------------------------------
# 讀取鳶尾花資料 (山鳶尾花, 變色鳶尾花, 維吉尼亞鳶尾花)
# 花萼長:0, 花萼寬:1, 花瓣長:2, 花瓣寬:3, 花種編號:4
#-------------------------------------------------
data=np.genfromtxt('iris.csv', delimiter=',')

#---------------------------
# 亂數重排資料
#---------------------------
np.random.shuffle(data)

#---------------------------
# 訓練資料個數
#---------------------------
tn=120

#---------------------------
# 訓練資料及標籤
#---------------------------
training_data  = data[:tn, [0,1,2,3]]
training_label = data[:tn, 4]

#---------------------------
# 測試資料及標籤
#---------------------------
testing_data  = data[tn:, [0,1,2,3]]
testing_label = data[tn:, 4]


#---------------------------
# 找較好的C及gamma值
#---------------------------
param_grid = {'C': [0.1, 0.5, 1, 5, 10, 15, 20, 100, 150, 175, 200, 250, 1000],
              'gamma': [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 10], }
clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
clf.fit(training_data, training_label)

print("找出較好的C及gamma")
best_C=clf.best_estimator_.C
best_gamma=clf.best_estimator_.gamma

print('C =', best_C)
print('gamma =', best_gamma)
print('-'*60)


#***********************************************
# 建立自動分類機器人
#***********************************************
svm_rbf = svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C)
svm_rbf.fit(training_data, training_label)

print('分類機器人參數')
print(svm_rbf)
print('-'*60)


#---------------------------
# 分類機器人測試
#---------------------------
print('正確:', testing_label)
print('-'*60)

predict = svm_rbf.predict(testing_data)
print('預測:', predict)
print('-'*60)

#---------------------------
# 和正確資料比對
#---------------------------
results = testing_label == predict
print('比對:', results)
print('-'*60)

#---------------------------
# 正確率
#---------------------------
print('正確率:', round(np.sum(results)/len(results), 2))
print('-'*60)
```




## 3.執行結果
```
找出較好的C及gamma
C = 1
gamma = 0.25
------------------------------------------------------------
分類機器人參數
SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.25, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
------------------------------------------------------------
正確: [1. 3. 3. 3. 1. 1. 1. 2. 1. 3. 1. 2. 1. 1. 2. 1. 2. 2. 2. 3. 1. 2. 2. 1.
 1. 3. 2. 3. 2. 3.]
------------------------------------------------------------
預測: [1. 3. 3. 3. 1. 1. 1. 2. 1. 3. 1. 2. 1. 1. 2. 1. 2. 2. 2. 3. 1. 2. 2. 1.
 1. 3. 2. 3. 2. 3.]
------------------------------------------------------------
比對: [ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True]
------------------------------------------------------------
正確率: 1.0
------------------------------------------------------------
```

