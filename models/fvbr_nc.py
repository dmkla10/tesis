import numpy as np
from itertools import product
from scipy import stats, optimize
from scipy.special import logsumexp, softplus, expit

class FVBR_NC():
    """
    Fully Visible Boltzmann Regression with No Covariates.
    Assumes that each theta_{(i,j)} = alpha_i + alpha_j
    """
    def __init__(self, parameters:dict,d:int):
        
        self.d = int(d)
        self.param_size = 2*self.d
        
        assert(self.d == d)
        assert(type(parameters["b"]) == dict)
        assert(type(parameters["alpha"]) == dict)
        assert(len(parameters["b"]) == self.d)
        assert(len(parameters["alpha"]) == self.d)
        
        
        
        self.b = {i:parameters["b"][i] for i in range(self.d)}
        self.b_vector = np.array(list(self.b.values()))
        
        self.alpha = {i:parameters["alpha"][i] for i in range(self.d)}
        self.alpha_vector = np.array(list(self.alpha.values()))
        
        self.param_vector = np.concatenate([self.b_vector,self.alpha_vector])
        
        self.W_keys = []
        for i in range(self.d):
            for j in range(i):
                self.W_keys.append((i,j))
        self.W_keys = sorted(self.W_keys)
        
        W = np.zeros(shape=(self.d,self.d))
        for i,j in self.W_keys:
            W[(i,j)] = self.alpha[i] + self.alpha[j]
            W[(j,i)] = self.alpha[j] + self.alpha[i]
        self.W = W
        
        self.omega = list(product([0,1], repeat=self.d))
        self.omega_list_array =[ (w,np.array(w)) for w in self.omega]
        self.omega_array = np.array(self.omega)
        self.id_to_omega = {i:elem for i, elem in enumerate(self.omega)}
        
        self.xx = {
            (i, j): self.omega_array[:, i] * self.omega_array[:, j]
            for (i, j) in self.W_keys
        }
        
        linear = self.omega_array @ self.b_vector        # shape (2^d,)
        quad = np.zeros(len(self.omega_array))
        for k, (i, j) in enumerate(self.W_keys):
            quad += self.W[i,j] * self.xx[(i, j)]

        log_scores = linear + quad
        logZ = logsumexp(log_scores)
        self.X_pmf = np.exp(log_scores - logZ)

        self.omega_id = list(self.id_to_omega.keys())

        self.join_distribution = stats.rv_discrete(name="FVBM", values=(self.omega_id, list(self.X_pmf )))
    
    def sample(self, size:int):
        sample_id = self.join_distribution.rvs(size=size)
        sample = []
        for s in sample_id:
            sample.append(self.id_to_omega[s])
        self.samples = sample
        self.samples_array = np.array(self.samples)
        self.sample_indices = [self.omega.index(x) for x in self.samples]
        
        self.D_x = np.zeros(self.d)
        self.D_xx = {key: 0.0 for key in self.W_keys}

        for x in self.samples:
            x = np.array(x)
            self.D_x += x
            for (i, j) in self.W_keys:
                self.D_xx[(i, j)] += x[i] * x[j]
        return "Done."
        
    def logl(self, params):
        """
        Compute the likelihood for a set of parameters
        """
        assert(len(params) == (len(self.b) + len(self.alpha)))
        
        b_values = params[:self.d]
        alpha_values = params[self.d:]
        
        alpha = {}
        b = {}
        for i in range(self.d):
            b[i] = b_values[i]
            alpha[i] = alpha_values[i]
        
        assert(len(alpha) == self.d)
        assert(len(b) == self.d)
        
        W = np.zeros(shape=(self.d,self.d))
        for i,j in self.W_keys:
            W[i,j] = alpha[i] + alpha[j]
            W[j,i] = alpha[j] + alpha[i]
        b_vector = np.array(list(b.values()))
        
        linear = self.omega_array @ b_vector
        quad = np.zeros(len(self.omega_array))
        for i, j in self.W_keys:
            quad += W[i,j] * self.xx[(i, j)]
        log_scores = linear + quad
        logZ = logsumexp(log_scores)
        p = np.exp(log_scores - logZ)
        
        ll = np.sum(log_scores[self.sample_indices]) - len(self.samples) * logZ
            
        return -ll
    
    def logl_jac(self, params, minimizing:bool=True):
        assert(len(params) == (len(self.b) + len(self.alpha)))
        
        b_values = params[:self.d]
        alpha_values = params[self.d:]
        
        alpha = {}
        b = {}
        for i in range(self.d):
            b[i] = b_values[i]
            alpha[i] = alpha_values[i]
        
        assert(len(alpha) == self.d)
        assert(len(b) == self.d)
        
        W = np.zeros(shape=(self.d,self.d))
        for i,j in self.W_keys:
            W[i,j] = alpha[i] + alpha[j]
            W[j,i] = alpha[j] + alpha[i]
        b_vector = np.array(list(b.values()))
        
        linear = self.omega_array @ b_vector
        quad = np.zeros(len(self.omega_array))
        for i, j in self.W_keys:
            quad += W[i,j] * self.xx[(i, j)]

        log_scores = linear + quad
        logZ = logsumexp(log_scores)
        p = np.exp(log_scores - logZ)
        
        E_x = p @ self.omega_array
        jac_b = self.D_x - len(self.samples) * E_x
                
         # --- data term for alphas: D_alpha[k] = sum_s x_sk * (row_sum_s - x_sk)
        row_sums = self.samples_array.sum(axis=1)                # shape (n,)
        D_alpha = (self.samples_array * (row_sums[:, None] - self.samples_array)).sum(axis=0)
        # D_alpha is shape (d,)

        # --- model expectations for alpha: use E_xx_full
        # weighted = p[:, None] * omega_array  -> (num_configs, d)
        weighted = p[:, None] * self.omega_array                 # shape (2^d, d)
        E_xx_full = weighted.T.dot(self.omega_array)            # shape (d, d)
        # expected derivative per alpha_k: sum_j E_xx_full[k, j] - E_x[k]
        E_alpha_term = E_xx_full.sum(axis=1) - E_x               # shape (d,)

        jac_alpha = D_alpha - len(self.samples) * E_alpha_term   # shape (d,)

        jac = np.zeros_like(params)
        jac[:self.d] = jac_b
        jac[self.d:] = jac_alpha
        return -jac

    
    def fit_logl(self, show_progress:bool=False):
        
        param0 = np.array([0]*self.param_size)
        if show_progress:
            print("Comenzando ajuste...")
        res = optimize.minimize(
            fun=self.logl,
            x0=param0,
            jac=self.logl_jac,
            method="L-BFGS-B"
            )
        if show_progress:
            print("Ajuste finalizado.")
        self.l_fitted_params = res.x
        p_dif = self.param_vector - self.l_fitted_params
        if show_progress:
            print("Modulo de la diferencia del ajuste de: " + str(np.linalg.norm(p_dif)))
            print("Diferencia promedio del ajuste de: " + str(sum(np.abs(p_dif))/self.d ))
        self.l_params_diff = np.abs(p_dif)
        
        return res
    
    def logpl(self, params):
        assert(len(params) == (len(self.b) + len(self.alpha)))
        
        b_values = params[:self.d]
        alpha_values = params[self.d:]
        
        alpha = {}
        b = {}
        for i in range(self.d):
            b[i] = b_values[i]
            alpha[i] = alpha_values[i]
        
        assert(len(alpha) == self.d)
        assert(len(b) == self.d)
        
        W = np.zeros(shape=(self.d,self.d))
        for i,j in self.W_keys:
            W[i,j] = alpha[i] + alpha[j]
            W[j,i] = alpha[j] + alpha[i]
        b_vector = np.array(list(b.values()))
        
        U = np.dot(self.samples_array, W) + b_vector
        
        lpl = (self.samples_array*U - softplus(U)).sum()
            
        return -lpl
    
    def logpl_jac(self, params):
        
        assert(len(params) == (len(self.b) + len(self.alpha)))
        
        b_values = params[:self.d]
        alpha_values = params[self.d:]
        
        alpha = {}
        b = {}
        for i in range(self.d):
            b[i] = b_values[i]
            alpha[i] = alpha_values[i]
        
        assert(len(alpha) == self.d)
        assert(len(b) == self.d)
        
        W = np.zeros(shape=(self.d,self.d))
        for i,j in self.W_keys:
            W[i,j] = alpha[i] + alpha[j]
            W[j,i] = alpha[j] + alpha[i]
        b_vector = np.array(list(b.values()))
        
        U = np.dot(self.samples_array, W) + b_vector
        resid = self.samples_array - expit(U)
        b_jac = resid.sum(axis=0)
        
        row_sums = self.samples_array.sum(axis=1)
        rsum = resid.sum(axis=1)
        
        term1 = self.samples_array.T.dot(rsum)
        term2 = np.sum(resid * (row_sums[:, None] - 2.0 * self.samples_array), axis=0)
        jac_alpha = term1 + term2
        
        jac = np.zeros_like(params, dtype=float)
        jac[:self.d] = b_jac
        jac[self.d:] = jac_alpha
        
        return -jac
        
        
    
    def fit_logpl(self, show_progress:bool=False):
        param0 = np.array([0]*self.param_size)
        if show_progress:
            print("Comenzando ajuste...")
        res = optimize.minimize(
            fun=self.logpl,
            x0=param0,
            jac=self.logpl_jac,
            method="L-BFGS-B"
            )
        if show_progress:
            print("Ajuste finalizado.")
        self.pl_fitted_params = res.x
        p_dif = self.param_vector - self.pl_fitted_params
        if show_progress:
            print("Modulo de la diferencia del ajuste de: " + str(np.linalg.norm(p_dif)))
            print("Diferencia promedio del ajuste de: " + str(sum(np.abs(p_dif))/self.d ))
        self.pl_params_diff = np.abs(p_dif)
        
        return res
        
