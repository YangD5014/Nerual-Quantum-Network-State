netket.vqs.MCState
class netket.vqs.MCState[source]
Bases: VariationalState

Variational State for a Variational Neural Quantum State.

The state is sampled according to the provided sampler.

Inheritance
Inheritance diagram of netket.vqs.MCState
__init__(sampler, model=None, *, n_samples=None, n_samples_per_rank=None, n_discard_per_chain=None, chunk_size=None, variables=None, init_fun=None, apply_fun=None, seed=None, sampler_seed=None, mutable=False, training_kwargs={})[source]
Constructs the MCState.

Parameters
:
sampler (Sampler) – The sampler

model – (Optional) The neural quantum state ansatz, encoded into a model. This should be a flax.linen.Module instance, or any other supported neural network framework. If not provided, you must specify init_fun and apply_fun.

n_samples (int | None) – the total number of samples across chains and processes when sampling (default=1000).

n_samples_per_rank (int | None) – the total number of samples across chains on one process when sampling. Cannot be specified together with n_samples (default=None).

n_discard_per_chain (int | None) – number of discarded samples at the beginning of each monte-carlo chain (default=5, except for ‘direct’ samplers where it is 0).

seed (Union[int, Any, None]) – rng seed used to generate a set of parameters (only if parameters is not passed). Defaults to a random one.

sampler_seed (Union[int, Any, None]) – rng seed used to initialise the sampler. Defaults to a random one.

mutable (Union[bool, str, Collection[str], DenyList]) – Name or list of names of mutable arguments. Use it to specify if the model has a state that can change during evaluation, but that should not be optimised. See also flax.linen.Module.apply() documentation (default=False)

init_fun (Callable[[Any, Sequence[int], Union[None, str, type[Any], dtype, _SupportsDType]], Array] | None) – Function of the signature f(model, shape, rng_key, dtype) -> Optional_state, parameters used to initialise the parameters. Defaults to the standard flax initialiser. Only specify if your network has a non-standard init method.

variables (Any | None) – Optional initial value for the variables (parameters and model state) of the model.

apply_fun (Callable | None) – Function of the signature f(model, variables, σ) that should evaluate the model. Defaults to model.apply(variables, σ). specify only if your network has a non-standard apply method.

training_kwargs (dict) – a dict containing the optional keyword arguments to be passed to the apply_fun during training. Useful for example when you have a batchnorm layer that constructs the average/mean only during training.

chunk_size (int | None) – (Defaults to None) If specified, calculations are split into chunks where the neural network is evaluated at most on chunk_size samples at once. This does not change the mathematical results, but will trade a higher computational cost for lower memory cost.

Attributes
chain_length
Length of the markov chain used for sampling configurations.

If running under JAX sharding, the total samples will be n_devices * chain_length * n_batches.

chunk_size
Suggested maximum size of the chunks used in forward and backward evaluations of the Neural Network model.

If your inputs are smaller than the chunk size this setting is ignored.

This can be used to lower the memory required to run a computation with a very high number of samples or on a very large lattice. Notice that inputs and outputs must still fit in memory, but the intermediate computations will now require less memory.

This option comes at an increased computational cost. While this cost should be negligible for large-enough chunk sizes, don’t use it unless you are memory bound!

This option is an hint: only some operations support chunking. If you perform an operation that is not implemented with chunking support, it will fall back to no chunking. To check if this happened, set the environment variable NETKET_DEBUG=1.

hilbert
The descriptor of the Hilbert space on which this variational state is defined.

model
Returns the model definition of this variational state.

When using model frameworks that encode the parameters directly into the model, such as equinox or flax.nnx, this will return the model including the parameters.

If you want access to the raw model without the parameters that is used internally by netket, use MCState._model instead.

model_state
The optional PyTree with the mutable state of the model, which is not optimized.

n_discard_per_chain
Number of discarded samples at the beginning of the markov chain.

n_parameters
The total number of parameters in the model.

n_samples
The total number of samples generated at every sampling step.

n_samples_per_rank
The number of samples generated on every JAX device at every sampling step.

parameters
The pytree of the parameters of the model.

sampler
The Monte Carlo sampler used by this Monte Carlo variational state.

samples
Returns the set of cached samples.

The samples returned are guaranteed valid for the current state of the variational state. If no cached parameters are available, then they are sampled first and then cached.

To obtain a new set of samples either use reset() or sample().

variables
The PyTree containing the parameters and state of the model, used when evaluating it.

sampler_state: SamplerState
The current state of the sampler.

mutable: bool | str | Collection[str] | DenyList
Specifies which collections in the model_state should be treated as mutable. Largely unused.

Methods
check_mc_convergence(op, *, min_chain_length=50, max_chain_length=500, plot=False)[source]
Diagnose whether the sampler’s sweep size produces decorrelated samples and whether the chains are stationary (thermalized).

Warning

Experimental functionality. This method is subject to change without notice in future NetKet releases. If you find it useful (or not!), please let us know with a 👍 / 👎 on GitHub or on Slack.

This routine temporarily runs the sampler at unit sweep size (one raw MC step per exposed sample) and estimates the integrated autocorrelation time 
 of the local estimators of op via an online Geyer IPS estimator. It reports whether the current sweep_size is sufficient to make consecutive samples effectively independent (i.e. 
 expressed in sweep units is less than 1).

When the autocorrelation window saturates — meaning the chains are still too short to see the full ACF tail — the internal sweep size is doubled adaptively, until convergence or until max_chain_length samples per chain are reached.

The original state is never mutated: all sampling is done on a shallow copy.

Parameters
:
op (AbstractOperator) – The operator whose local estimators are used to probe correlations (typically the Hamiltonian).

min_chain_length (int) – Minimum number of samples per chain to accumulate before the convergence criterion is evaluated.

max_chain_length (int) – Hard upper limit on samples per chain. The procedure stops unconditionally once this many samples have been drawn, even if convergence has not been reached.

plot (bool) – If True, display a diagnostic figure showing the evolution of the mean, autocorrelation time, and 
 across iterations.

Returns
:
A tuple (stats, hist_data) where stats is the final OnlineStatistics accumulator and hist_data is a HistoryDict recording the evolution of key diagnostics as the number of samples is increased.

See also

netket._src.vqs.check_mc_convergence.check_mc_convergence()

expect(O)[source]
Estimates the quantum expectation value for a given operator 
 or generic observable. In the case of a pure state 
 and an operator, this is 
 otherwise for a mixed state 
, this is 
.

Parameters
:
O (AbstractOperator) – the operator or observable for which to compute the expectation value.

Return type
:
Stats

Returns
:
An estimation of the quantum expectation value 
.

expect_and_forces(O, *, mutable=None)[source]
Estimates the quantum expectation value and the corresponding force vector for a given operator O.

The force vector 
 is defined as the covariance of log-derivative of the trial wave function and the local estimators of the operator. For complex holomorphic states, this is equivalent to the expectation gradient 
 
. For real-parameter states, the gradient is given by 
 
.

Parameters
:
O (AbstractOperator) – The operator O for which expectation value and force are computed.

mutable (Union[bool, str, Collection[str], DenyList, None]) – Can be bool, str, or list. Specifies which collections in the model_state should be treated as mutable: bool: all/no collections are mutable. str: The name of a single mutable collection. list: A list of names of mutable collections. This is used to mutate the state of the model while you train it (for example to implement BatchNorm. Consult Flax’s Module.apply documentation for a more in-depth explanation).

Return type
:
tuple[Stats, Any]

Returns
:
An estimate of the quantum expectation value <O>. An estimate of the force vector 
.

expect_and_grad(O, *, mutable=None, **kwargs)[source]
Estimates the quantum expectation value and its gradient for a given operator 
.

Parameters
:
O (AbstractOperator) – The operator 
 for which expectation value and gradient are computed.

mutable (Union[bool, str, Collection[str], DenyList, None]) –

Can be bool, str, or list. Specifies which collections in the model_state should be treated as mutable: bool: all/no collections are mutable. str: The name of a single mutable collection. list: A list of names of mutable collections. This is used to mutate the state of the model while you train it (for example to implement BatchNorm. Consult Flax’s Module.apply documentation for a more in-depth explanation).

use_covariance – whether to use the covariance formula, usually reserved for hermitian operators, 

Return type
:
tuple[Stats, Any]

Returns
:
An estimate of the quantum expectation value <O>. An estimate of the gradient of the quantum expectation value <O>.

expect_to_precision(op, *, atol=None, rtol=None, max_iter=10000, max_lag=64, verbose=True)[source]
Sample until the standard error of 
 meets the requested tolerance.

Warning

Experimental functionality. This method is subject to change without notice in future NetKet releases. If you find it useful (or not!), please let us know with a 👍 / 👎 on GitHub or on Slack.

Iteratively draws new batches of samples and updates an online statistics accumulator until the estimated standard error of the mean satisfies the requested absolute and/or relative tolerance, or until max_iter iterations are exhausted. A progress bar is shown by default.

At least one of atol or rtol must be provided. If both are given, sampling continues until both are simultaneously satisfied.

Unlike expect(), this method modifies the state’s sampler in place (new samples are drawn on self directly), so the sampler state is advanced as a side effect.

Parameters
:
op (AbstractOperator) – The operator 
 whose expectation value 
 is estimated.

atol (float | None) – Desired absolute standard error of the mean. Sampling stops when error_of_mean ≤ atol.

rtol (float | None) – Desired relative standard error of the mean. Sampling stops when error_of_mean / |mean| ≤ rtol.

max_iter (int) – Maximum number of sampling iterations before stopping unconditionally.

max_lag (int) – Maximum lag used by the online autocorrelation estimator.

verbose (bool) – If True (default), display a tqdm progress bar showing the current error and tolerances.

Returns
:
The final OnlineStatistics accumulator. Call .get_stats() on it to obtain a standard Stats object with mean, variance, and error of the mean.

Raises
:
ValueError – If neither atol nor rtol is provided, or if the sampler is not a MetropolisSampler.

See also

netket._src.vqs.expect_to_precision.expect_to_precision()

grad(Ô, *, use_covariance=None, mutable=None)[source]
Estimates the gradient of the quantum expectation value of a given operator O.

Parameters
:
op (netket.operator.AbstractOperator) – the operator O.

is_hermitian – optional override for whether to use or not the hermitian logic. By default it’s automatically detected.

use_covariance (bool | None)

mutable (bool | str | Collection[str] | DenyList | None)

Returns
:
An estimation of the average gradient of the quantum expectation value <O>.

Return type
:
array

init(seed=None, dtype=None)[source]
Initialises the variational parameters of the variational state.

init_parameters(init_fun=None, *, seed=None)[source]
Re-initializes all the parameters with the provided initialization function, defaulting to the normal distribution of standard deviation 0.01.

Warning

The init function will not change the dtype of the parameters, which is determined by the model. DO NOT SPECIFY IT INSIDE THE INIT FUNCTION

Parameters
:
init_fun (Callable[[Any, Sequence[int], Union[None, str, type[Any], dtype, _SupportsDType]], Array] | None) – a jax initializer such as jax.nn.initializers.normal(). Must be a Callable taking 3 inputs, the jax PRNG key, the shape and the dtype, and outputting an array with the valid dtype and shape. If left unspecified, defaults to jax.nn.initializers.normal(stddev=0.01)

seed (Any | None) – Optional seed to be used. The seed is synced across all JAX processes. If unspecified, uses a random seed.

local_estimators(op, *, chunk_size=None)[source]
Compute the local estimators for the operator op (also known as local energies when op is the Hamiltonian) at the current configuration samples self.samples.

 
Warning

The samples differ between JAX processes, so returned the local estimators will also take different values on each process. To compute sample averages and similar quantities, you will need to perform explicit operations over all JAX devices. (Use functions like self.expect to get process-independent quantities without manual reductions.)

Parameters
:
op (AbstractOperator) – The operator.

chunk_size (int | None) – Suggested maximum size of the chunks used in forward and backward evaluations of the model. (Default: self.chunk_size)

log_value(σ)[source]
Evaluate the variational state for a batch of states and returns the logarithm of the amplitude of the quantum state.

For pure states, this is 
, whereas for mixed states this is 
, where 
 and 
 are respectively a pure state (wavefunction) and a mixed state (density matrix). For the density matrix, the left and right-acting states (row and column) are obtained as σr=σ[::,0:N] and σc=σ[::,N:].

Given a batch of inputs (Nb, N), returns a batch of outputs (Nb,).

Return type
:
Array

Parameters
:
σ (Array)

quantum_geometric_tensor(qgt_T=None)[source]
Computes an estimate of the quantum geometric tensor G_ij. This function returns a linear operator that can be used to apply G_ij to a given vector or can be converted to a full matrix.

Parameters
:
qgt_T (Callable[[VariationalState], LinearOperator] | None) – the optional type of the quantum geometric tensor. By default it’s automatically selected.

Returns
:
A linear operator representing the quantum geometric tensor.

Return type
:
nk.optimizer.LinearOperator

reset()[source]
Resets the sampled states. This method is called automatically every time that the parameters/state is updated.

sample(*, chain_length=None, n_samples=None, n_discard_per_chain=None)[source]
Sample a certain number of configurations.

If one among chain_length or n_samples is defined, that number of samples are generated. Otherwise the value set internally is used.

Parameters
:
chain_length (int | None) – The length of the markov chains.

n_samples (int | None) – The total number of samples across all JAX devices.

n_discard_per_chain (int | None) – Number of discarded samples at the beginning of the markov chain.

Return type
:
Array

to_array(normalize=True)[source]
Returns the dense-vector representation of this state.

Parameters
:
normalize (bool) – If True, the vector is normalized to have L2-norm 1.

Return type
:
Array

Returns
:
An exponentially large vector representing the state in the computational basis.

to_qobj()[source]
Convert the variational state to a qutip’s ket Qobj.

Returns
:
A qutip.Qobj object.