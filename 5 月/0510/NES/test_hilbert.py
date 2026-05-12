"""
测试不同 Hilbert 空间的创建方式
"""
import netket as nk
import netket.experimental as nkx

# 尝试不同的创建方式
try:
    print("尝试 nk.hilbert.SpinOrbitalFermions...")
    hi = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2,
        s=1/2,
        n_fermions_per_spin=(1, 1)
    )
    print(f"成功: {hi}")
except AttributeError as e:
    print(f"失败: {e}")

try:
    print("\n尝试 nkx.hilbert.SpinOrbitalFermions...")
    hi = nkx.hilbert.SpinOrbitalFermions(
        n_orbitals=2,
        s=1/2,
        n_fermions_per_spin=(1, 1)
    )
    print(f"成功: {hi}")
except AttributeError as e:
    print(f"失败: {e}")

try:
    print("\n尝试 nk.hilbert.Fermions...")
    hi = nk.hilbert.Fermions(
        n_sites=4,
        n_fermions=2
    )
    print(f"成功: {hi}")
except AttributeError as e:
    print(f"失败: {e}")

# 列出所有可用的 hilbert 类
print("\n=== nk.hilbert 中的可用类 ===")
print([x for x in dir(nk.hilbert) if not x.startswith('_')])

print("\n=== nkx.hilbert 中的可用类 ===")
print([x for x in dir(nkx.hilbert) if not x.startswith('_')])
