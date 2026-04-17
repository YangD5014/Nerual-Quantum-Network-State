import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pyscf import gto, scf, fci
import netket.experimental as nkx
import expect_grad_ex
import vmc_ex
import jax
from functools import reduce



class H2ExcitedEnergy:
    def __init__(self, bond_length: float):
        self.bond_length = bond_length
        self.geometry = [
            ('H', (0., 0., 0.)),
            ('H', (self.bond_length, 0., 0.)),
        ]
        self.mol = gto.M(atom=self.geometry, basis='STO-3G')
        self.mf = scf.RHF(self.mol).run(verbose=0)
        self.E_hf = self.mf.e_tot
        print(f"Hartree-Fock能量: {self.E_hf:.8f} Ha")
        self.FCI_compute()
        self.ha =nkx.operator.from_pyscf_molecule(self.mol)
        self.hi = nk.hilbert.SpinOrbitalFermions(n_orbitals=2,s=1/2,n_fermions_per_spin=(1, 1))
        self.graph = nk.graph.Graph(edges=[(0,1),(2,3)])
        self.sampler = nk.sampler.MetropolisFermionHop(
            self.hi, graph=self.graph, n_chains=16, spin_symmetric=True, sweep_size=64
        )
        
    def set_model(self, model,n_discard_per_chain:int=10,n_samples:int=512,learning_rate:float=0.01):
        self.model = model
        self.n_discard_per_chain = n_discard_per_chain
        self.n_samples = n_samples
        self.vstate = nk.vqs.MCState(
            self.sampler, self.model, n_discard_per_chain=n_discard_per_chain, n_samples=n_samples
        )
        # 设置优化器
        self.optimizer = nk.optimizer.Sgd(learning_rate=0.01)
        self.sr = nk.optimizer.SR(diag_shift=0.01, holomorphic=True) #随机重构
        
    def compute_ground_state(self,iterations:int=500,file_path:str='./Data_H2/RBM'):
        self.gs_logger_name = file_path+f'/H2_ground_state_{self.bond_length:.2f}A'
        self.groundstate_driver = nk.driver.VMC(self.ha, self.optimizer, variational_state=self.vstate, preconditioner=self.sr)
        self.groundstate_driver.run(n_iter=iterations,out=self.gs_logger_name)
        
        # 保存基态参数
        self.ground_state_params = self.vstate.parameters
        with open(self.gs_logger_name + '.json', 'wb') as f:
            pickle.dump(self.ground_state_params, f)
        print(f"基态参数已保存到 {self.gs_logger_name + '.json'}")
        
        # 获取基态能量
        exp_name = self.gs_logger_name
        data = json.load(open(exp_name + '.log'))
        energy_gs = data["Energy"]["Mean"]["real"]
        self.final_energy_gs = reduce(lambda x, y: x if y is None else y, energy_gs)
        print(f"\n基态能量: {self.final_energy_gs:.8f} Ha")
        print(f"与FCI误差: {abs(self.final_energy_gs - self.E_fci):.8f} Ha")
        
    def compute_1st_excited_state(self,lamada:float=2.5,iterations:int=500,file_path:str='./Data_H2/RBM'):
        self.ex1_logger_name = file_path+f'/H2_1st_excited_state_{self.bond_length:.2f}A'
        # 加载基态参数
        with open(self.gs_logger_name + '.json', 'rb') as f:
            gs_params = pickle.load(f)

        # 创建基态变分态对象
        self.ground_vstate = nk.vqs.MCState(self.sampler, self.model, n_discard_per_chain=self.n_discard_per_chain, n_samples=self.n_samples)
        self.ground_vstate.init_parameters(jax.nn.initializers.normal(stddev=0.25))
        self.ground_vstate.parameters = gs_params
        
        #print(f'基态加载完毕!基态能量={self.gs}')
        model_ex1 = self.model
        vs_ex1 = nk.vqs.MCState(self.sampler, model_ex1, n_discard_per_chain=self.n_discard_per_chain, n_samples=self.n_samples)
        shift_list = [lamada]  # 基态的惩罚权重参数
        state_list = [self.ground_vstate]  # 基态列表
        gs_ex1 = vmc_ex.VMC_ex(
            hamiltonian=self.ha,
            optimizer=self.optimizer,
            variational_state=vs_ex1,
            preconditioner=self.sr,
            state_list=state_list,
            shift_list=shift_list
        )
        
        gs_ex1.run(out=self.ex1_logger_name, n_iter=iterations)

        # 保存第一激发态参数
        self.first_excited_state_params = vs_ex1.parameters
        with open(self.ex1_logger_name + '.json', 'wb') as f:
            pickle.dump(self.first_excited_state_params, f)
        print(f"第一激发态参数已保存到 {self.ex1_logger_name + '.json'}")
        
        # 获取第一激发态能量
        data_ex1 = json.load(open(self.ex1_logger_name + '.log'))
        energy_ex1 = data_ex1["Energy"]["Mean"]["real"]
        self.final_energy_ex1 = reduce(lambda x, y: x if y is None else y, energy_ex1)
        print(f"\n第一激发态能量: {self.final_energy_ex1:.8f} Ha")
        print(f'精确的第一激发态能量: {self.E_fcis[1]:.8f} Ha')
        print(f"激发能: {self.final_energy_ex1 - self.final_energy_gs:.8f} Ha")
        
    def compute_2nd_excited_state(self,iterations:int=500,file_path:str='./Data_H2/RBM'):
        self.ex2_logger_name = file_path+f'/H2_2nd_excited_state_{self.bond_length:.2f}A'
        #加载第一激发态参数
        with open(self.ex1_logger_name + '.json', 'rb') as f:
            ex1_params = pickle.load(f)
        # 创建第一激发态变分态对象
        self.first_excited_vstate = nk.vqs.MCState(self.sampler, self.model, n_discard_per_chain=self.n_discard_per_chain, n_samples=self.n_samples)
        self.first_excited_vstate.init_parameters(jax.nn.initializers.normal(stddev=0.25))
        self.first_excited_vstate.parameters = ex1_params
        
        model_ex2 = self.model
        vs_ex2 = nk.vqs.MCState(self.sampler, model_ex2, n_discard_per_chain=self.n_discard_per_chain, n_samples=self.n_samples)
        shift_list = [1.0,1.0]  # 基态的惩罚权重参数
        state_list = [self.ground_vstate,self.first_excited_vstate]  # 基态列表
        gs_ex2 = vmc_ex.VMC_ex(
            hamiltonian=self.ha,
            optimizer=self.optimizer,
            variational_state=vs_ex2,
            preconditioner=self.sr,
            state_list=state_list,
            shift_list=shift_list
        )
    
        gs_ex2.run(out=self.ex2_logger_name, n_iter=iterations)
        
        # 保存第二激发态参数
        self.second_excited_state_params = vs_ex2.parameters
        with open(self.ex2_logger_name + '.json', 'wb') as f:
            pickle.dump(self.second_excited_state_params, f)
        print(f"第二激发态参数已保存到 {self.ex2_logger_name + '.json'}")
        
        # 获取第二激发态能量
        data_ex2 = json.load(open(exp_name_ex2 + '.log'))
        energy_ex2 = data_ex2["Energy"]["Mean"]["real"]
        self.final_energy_ex2 = reduce(lambda x, y: x if y is None else y, energy_ex2)
        print(f"\n第二激发态能量: {self.final_energy_ex2:.8f} Ha")
        print(f'精确的第二激发态能量: {self.E_fcis[2]:.8f} Ha')
        print(f"激发能: {self.final_energy_ex2 - self.final_energy_gs:.8f} Ha")
        
        
    #计算所有的基态、第一激发态、第二激发态、第三激发态
    def FCI_compute(self):
        self.cisolver = fci.FCI(self.mf)
        self.cisolver.nroots=4
        self.E_fcis, self.fcivec = self.cisolver.kernel()
        self.E_fci = self.E_fcis[0]
        for i in range(4):
            if i==0:
                print(f"FCI基态能量: {self.E_fcis[i]:.8f} Ha")
            else:
                print(f"FCI第{i}激发态能量: {self.E_fcis[i]:.8f} Ha")
            
