# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 00:42:17 2021

@author: bi19053
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df1 = pd.read_csv('data_all.csv', encoding="cp932")
y_train = ｄｆ1[u'外部発電'].values.tolist()
data_train = df1.drop([u'計測日時', u'太陽光発電(PV1)', u'外部発電', u'日照時間(時間)', u'平均風速(m/s)', u'月日', u'平均湿度(％)', u'降水量の合計(mm)', u'曜日', u'休日'], axis=1)
x_train = data_train.values.tolist()
df2 = pd.read_csv('estimate.csv', encoding="cp932")
y_test = ｄｆ2[u'外部発電'].values.tolist()
data_test = df2.drop([u'計測日時', u'太陽光発電(PV1)', u'外部発電', u'日照時間(時間)', u'平均風速(m/s)', u'月日', u'平均湿度(％)', u'降水量の合計(mm)', u'曜日', u'休日'], axis=1)
x_test = data_test.values.tolist()
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)
# テストデータで予測実行
predict_y = rfr.predict(x_test)

# R2決定係数で評価
print(predict_y)
r2_score = r2_score(y_test, predict_y)
print(r2_score)
# 特徴量の重要度を取得
feature = rfr.feature_importances_
# 特徴量の名前ラベルを取得
label = data_train.columns[0:]
# 特徴量の重要度順（降順）に並べて表示
indices = np.argsort(feature)[::-1]
for i in range(len(feature)):
    print(str(i + 1) + "   " +
          str(label[indices[i]]) + "   " + str(feature[indices[i]]))

# 最大電力量の実績と予測値の比較グラフ
plt.subplot(121, facecolor='white')
plt.title('7月の電力量')
plt_label = [i for i in range(1, 366)]
plt.plot(plt_label, y_test, color='blue')
plt.plot(plt_label, predict_y, color='red')
# 特徴量の重要度の棒グラフ
plt.subplot(122, facecolor='white')
plt.title('特徴量の重要度')
plt.bar(
    range(
        len(feature)),
    feature[indices],
    color='blue',
    align='center')
plt.xticks(range(len(feature)), label[indices], rotation=45)
plt.xlim([-1, len(feature)])
plt.tight_layout()
# グラフの表示
plt.show()