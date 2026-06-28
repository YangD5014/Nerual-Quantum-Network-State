

变分量子蒙特卡洛算法中,基态能量的表达式如下：

$$
E(\theta) = \int E_{loc}(R,\theta) \pi(R,\theta)dR

$$

R 是系统配置,$\theta$ 是模型参数
$\pi(R,\theta)$ 是模型在R上的密度: $\pi(R,\theta) = \frac{ \left| \psi(R,\theta) \right|^2}{\int \left| \psi(R,\theta) \right|^2 dR}$
$E_{loc}(R,\theta)$ 是本地能量: $E_{loc}=\frac{\hat{H}\psi(R,\theta) }{\psi(R,\theta)}$

但是在 Netket 官方给出的定义是:
$$
O_{\text{loc}}(s) = \frac{\langle s | \text{op} | \psi \rangle}{\langle s | \psi \rangle}
$$