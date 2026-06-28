

变分量子蒙特卡洛算法中,基态能量的表达式如下：

$$
E(\theta) = \int E_{loc}(R,\theta) \pi(R,\theta)dR

$$

R 是系统配置,$\theta$ 是模型参数
$\pi(R,\theta)$ 是模型在R上的密度: $\pi(R,\theta) = \frac{ \left| \psi(R,\theta) \right|^2}{\int \left| \psi(R,\theta) \right|^2 dR}$
$E_{loc}(R,\theta)$ 是本地能量: $E_{loc}=\frac{\hat{H}\psi(R,\theta) }{\psi(R,\theta)}$

梯度公式推导为:

$$
\frac{dE(\theta)}{d\theta}=\frac{d}{d\theta} \int E_{loc}(R,\theta) \pi(R,\theta)dR

$$

由于被积函数的上下限不包含$\theta$,根据莱布尼兹求导法则, $\frac{d}{d\theta}$可以放入积分符号内部.
且根据推论: $\frac{df(x)}{dx} = f(x)\frac{ln(f(x))}{dx}$,则:

$$
\nabla_{\theta}E = \int \frac{E_{loc}(R,\theta)}{d\theta} \pi(R,\theta) +E_{loc}(R,\theta) \frac{ln(\pi(R,\theta))}{d\theta} dR
$$
$$
\nabla_{\theta}E = \int E_{loc}(R,\theta)\pi(R,\theta) \left[ \frac{d\ln(\pi(R,\theta))}{d\theta}+\frac{dE_{loc}(R,\theta)}{d\theta} \right] dR
$$  
其中$\int{\frac{d\ln(E_{loc}(R,\theta))}{d(\theta)} \pi(R,\theta) dR} = 0$  

$$
\therefore \nabla_{\theta}E = \int E_{loc}(R,\theta)\pi(R,\theta)  \nabla_{\theta}\ln{\pi(R,\theta)} dR
$$ 



$$\nabla_{\theta}\ln(\pi(R,\theta)) = \frac{d}{d\theta}\left[\ln{\psi(R,\theta)\psi^*(R,\theta)}-\ln{\int\left| \psi(R,\theta) \right|^2 dR} \right]
$$ 
$$
令Z = \int \left| \psi(R,\theta) \right|^2 dR\\
\nabla_{\theta}\ln{Z} = \frac{1}{Z} \nabla_{\theta}Z = \frac{1}{Z} \frac{d}{d\theta} \int \left| \psi(R,\theta) \right|^2 dR\\
\frac{1}{Z} \nabla_{\theta} \int{\psi(R,\theta) \psi^*(R,\theta) dR}=\frac{1}{Z} \int{\nabla_{\theta}\psi \psi^* + \nabla_{\theta}\psi^* \psi dR}
$$
因此整合得到:  

$$
\nabla_{\theta}E = \int{E_{loc}\pi \left[\nabla\ln{\psi} + \nabla \ln{\psi^* - \frac{1}{Z}\int{\nabla \psi \psi^* +\nabla \psi^* \psi}} \right]} dR
$$

令$\mathcal{O}_{\psi} = \nabla_{\theta} \ln{\psi(R,\theta)}$,令$\mathcal{O}_{\psi^*} = \nabla_{\theta} \ln{\psi^*(R,\theta)}$，所以$\mathcal{O}_{\psi} + \mathcal{O}_{\psi^*} =\frac{\nabla \psi \psi^* + \nabla \psi^* \psi}{Z}$

$$ \therefore 
\nabla_{\theta}E = \int{E_{loc}\pi \left[（\mathcal{O}_{\psi} + \mathcal{O}_{\psi^*}） - \frac{1}{Z}\int{\nabla \psi \psi^* +\nabla \psi^* \psi}dR^{\prime} \right]} dR
$$
总之最后的梯度公式为:
$$
\boxed{
\nabla_{\theta} E = \mathbb{E}_{\pi}\left[ E_{\text{loc}} \left( \mathcal{O}_{\psi} + \mathcal{O}_{\psi^*} \right) \right] - \mathbb{E}_{\pi}\left[ E_{\text{loc}} \right] \mathbb{E}_{\pi}\left[ \mathcal{O}_{\psi} + \mathcal{O}_{\psi^*} \right]
}
$$
在代码里,将变形为:
$$
\boxed{
\nabla_{\theta}E = \mathbb{E}_{\pi}\left[ \left(E_{\text{loc}}-\mathbb{E}_{\pi}\left[ E_{\text{loc}} \right] \right) \left( \mathcal{O}_{\psi} + \mathcal{O}_{\psi^*} \right) \right]
 }
$$