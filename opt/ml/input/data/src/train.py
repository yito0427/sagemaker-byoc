#!/usr/bin/env python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# データ読み込み
#train_df = pd.read_csv('../opt/ml/input/data/train/train.csv')
#valid_df = pd.read_csv('../opt/ml/input/data/valid/valid.csv')
#test_df = pd.read_csv('../opt/ml/input/data/test/test.csv')
train_df = pd.read_csv('/opt/ml/input/data/train/train.csv')
valid_df = pd.read_csv('/opt/ml/input/data/valid/valid.csv')
test_df = pd.read_csv('/opt/ml/input/data/test/test.csv')


tr_x, tr_y = train_df.drop(['target'], axis=1), train_df['target']
va_x, va_y = valid_df.drop(['target'], axis=1), valid_df['target']
test_x, test_y = test_df.drop(['target'], axis=1), test_df['target']

# 学習準備
# LightGBM が扱うデータセットの形式に直す
dtrain = lgb.Dataset(tr_x, label=tr_y)
dvalid = lgb.Dataset(va_x, label=va_y)
dtest = lgb.Dataset(test_x)

# 学習用のパラメータ
lgb_params = {
    # 二値分類問題
    'objective': 'binary',
    # 評価指標
    'metrics': 'binary_logloss',
}

# 学習
# モデルを学習する
# バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする
# watchlistには学習データおよびバリデーションデータをセットする
#watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = lgb.train(lgb_params,
                dtrain,
                num_boost_round=50,  # 学習ラウンド数は適当
                #evals=watchlist
                valid_names=['train','valid'], valid_sets=[dtrain, dvalid]
                )
# 予測
# 予測：検証用データが各クラスに分類される確率を計算する
pred_proba = model.predict(test_x)
# しきい値 0.5 で 0, 1 に丸める
pred = np.where(pred_proba > 0.5, 1, 0)
# 評価
# 精度 (Accuracy) を検証する
acc = accuracy_score(test_y, pred)
print('Accuracy:', acc)
# 予測結果を出力
