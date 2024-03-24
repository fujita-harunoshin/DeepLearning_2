# DeepLearning_2

# パスの指定

import sys, os
current_dir = os.path.dirname(os.path.abspath(**file**))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ピクルにおける path 指定

## ピクルの使い方

pkl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cbow_params.pkl')
with open(pkl_file, 'wb') as f:
  pickle.dump(params, f, -1)

## trainer での使い方

pkl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rnnlm_params.pkl')
model.save_params(file_name=pkl_file)

# Chapter4

## エラー

### np と cp によるエラー

・cp 使用時、他モジュールで numpy を使用でのエラー
　 GPU 使用する場合、全ての import numpy as np を from common.np import \*に変換
・scatter_add でのエラー
　バージョンによるもの
　 cupyx.scatter_add(dW, self.idx, dout)を使用
　https://www.arbk.net/wp/%E4%BB%8A%E6%97%A5%E3%82%82%E4%B8%80%E6%97%A5%E3%81%82%E3%82%8A%E3%81%8C%E3%81%A8%E3%81%86-470/
