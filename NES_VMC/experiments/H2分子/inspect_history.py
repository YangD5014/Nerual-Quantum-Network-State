# -*- coding: utf-8 -*-
import pickle
import numpy as np

HISTORY_FILE = r'./data/26-09-05-03-48_history_natural_gradient_H2_molecule_K4.pkl'
with open(HISTORY_FILE, "rb") as f:
    history = pickle.load(f)

print("history keys:", list(history.keys()))
for k, v in history.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={getattr(v,'dtype',None)}")
    elif isinstance(v, list):
        print(f"  {k}: list len={len(v)}")

# 能量相关
for ek in ['energy', 'energies', 'E', 'E_mean', 'loss', 'loss_mean', 'E_L']:
    if ek in history:
        arr = np.asarray(history[ek])
        print(f"\n== {ek} (last 5) ==")
        print(arr[-5:] if arr.ndim > 0 else arr)

# 看看有没有逐状态的能量记录
for k in list(history.keys()):
    if any(s in k.lower() for s in ['e', 'loss', 'energy']):
        v = history[k]
        arr = np.asarray(v)
        print(f"\n{k}: last3 =")
        print(arr[-3:] if arr.ndim > 0 else arr)
