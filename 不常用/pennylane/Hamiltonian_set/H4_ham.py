from pyscf import gto, scf, fci
import netket as nk
import netket.experimental as nkx




def H4_ham(bond_length:float):
    """
    生成H4分子的哈密顿量和FCI能量 Netket框架

    参数:
    bond_length (float): H-H键长，单位为埃

    返回:
    ha (nkx.operator): H4分子的哈密顿量
    E_fci (float): H4分子的FCI能量
    """
    # 设置H4分子的几何构型
    # 使用线性构型，H-H键长为1.0埃
    geometry = [
        ('H', (0., 0., 0.)),
        ('H', (bond_length, 0., 0.)),
        ('H', (2*bond_length, 0., 0.)),
        ('H', (3*bond_length, 0., 0.))
    ]

    # 创建分子对象，使用STO-3G基组
    mol = gto.M(atom=geometry, basis='STO-3G')

    # 进行Hartree-Fock计算
    mf = scf.RHF(mol).run(verbose=0)
    # E_hf = sum(mf.scf_summary.values())
    # print(f"Hartree-Fock能量: {E_hf:.8f} Ha")

    # 进行FCI计算作为参考
    E_fci = fci.FCI(mf).kernel()[0]
    print(f"FCI能量: {E_fci:.8f} Ha")

    # 使用NetKet创建哈密顿量
    ha = nkx.operator.from_pyscf_molecule(mol)

    return ha, E_fci

