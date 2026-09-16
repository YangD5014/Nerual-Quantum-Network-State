# -*- coding: utf-8 -*-
"""冒烟测试: exec notebook 的 imports/machines/monitor 三个单元, 用事故 pickle 数据验证监控函数。"""
import json, pickle, os, sys
import numpy as np

NB = "/Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/h2-6-31G-K4-0915-monitored.ipynb"
os.chdir(os.path.dirname(NB))
sys.path.insert(0, ".")

nb = json.load(open(NB))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
ns = {}
for c in cells[:3]:  # imports/config, machines, monitor-utils
    src = "".join(c["source"])
    exec(compile(src, c.get("id", "<cell>"), "exec"), ns)
print(">> notebook 前 3 个代码单元执行成功")

from H2_631G import K, SINGLE_SIZE

with open("./data/26-09-15-17-57_history_natural_gradient_H2_molecule_K4.pkl", "rb") as f:
    hist = pickle.load(f)

import jax.numpy as jnp

for s in [216, 219, 221, 225]:
    params = hist["params"][s]
    samples = np.asarray(hist["samples"][s])
    x = jnp.asarray(samples.reshape(-1, K, SINGLE_SIZE))
    tr_n, O, cond_psi_all, shift_b = ns["monitor_fn"](params, x, jnp.zeros(K, jnp.complex64))
    tr_n = np.asarray(tr_n)
    jk_ratio, jk_cos, top_idx = ns["gradient_concentration"](tr_n, np.asarray(O))
    sh = ns["sampling_health"](samples.reshape(-1, K, SINGLE_SIZE))
    print(f"step {s}: tr_n{tr_n.shape} med={np.median(tr_n):.4f} O{O.shape} condΨ_max={float(np.max(np.asarray(cond_psi_all))):.1f} "
          f"JKρ={jk_ratio:.3f} cos={jk_cos:.3f} uniq={sh['n_uniq']} maxDup={sh['max_dup_frac']:.2%} minChain={sh['min_chain_uniq']}")
print(">> 监控函数全部验证通过")
