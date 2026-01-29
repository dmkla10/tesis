import numpy as np
from itertools import product
from scipy import stats, optimize
from scipy.special import logsumexp, softplus, expit

class FVBR_NF():
    """
    Fully Visible Boltzmann Regression with No Covariates.
    Assumes that each theta_{(i,j)} = alpha_i + alpha_j
    """
    def __init__(self, parameters:dict, nodes_features:dict, d:int, c:int, d_max_l:int=10):
        
        self.d = int(d)
        self.c = int(c)
        self.d_max_l = int(d_max_l)
        self.param_size = self.c + self.c*self.d
        
        assert(len(parameters["gamma"]) == c)
        assert(len(nodes_features) == d)
        assert(all([(len(v) == c) for v in nodes_features.values()]))
        assert(all([(len(l) == c) for l in parameters["lambd"].values()]))
        assert(self.d == d)
        
        self.V = {i:nodes_features[i] for i in range(self.d)}
        self.V_matrix = np.array((list(self.V.values())))
        
        
        self.lambd = {i:parameters["lambd"][i] for i in range(self.d)}
        self.lambd_matrix = np.array(list(self.lambd.values()))
        
        self.gamma = parameters["gamma"]
        self.gamma_vector = np.array(self.gamma)
        
        self.param_vector = np.concatenate([self.gamma_vector,self.lambd_matrix.flatten()])
        
        self.W_keys = []
        for i in range(self.d):
            for j in range(i):
                self.W_keys.append((i,j))
        self.W_keys = sorted(self.W_keys)
        
        W = np.zeros(shape=(self.d,self.d))
        for i,j in self.W_keys:
            W[(i,j)] = np.dot(self.lambd[i], self.V[j]) + np.dot(self.lambd[j], self.V[i])
            W[(j,i)] = np.dot(self.lambd[j], self.V[i]) + np.dot(self.lambd[i], self.V[j])
        self.W = W.copy()
        
        self.omega = list(product([0,1], repeat=self.d))
        self.omega_list_array =[ (w,np.array(w)) for w in self.omega]
        self.omega_array = np.array(self.omega)
        self.id_to_omega = {i:elem for i, elem in enumerate(self.omega)}
        
        self.xx = {
            (i, j): self.omega_array[:, i] * self.omega_array[:, j]
            for (i, j) in self.W_keys
        }
        
        self.b_vector = (self.V_matrix @ self.gamma_vector).copy()
        
        linear = self.omega_array @ self.b_vector
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
        self.samples_array = np.array(self.samples, dtype=float)
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
        if self.d > self.d_max_l:
            raise ValueError("d is too big for likelihood computation")
        assert(len(params) == (len(self.gamma) + self.lambd_matrix.size))
        
        gamma = np.array(params[:self.c])
        lambd_values = params[self.c:]
        
        lambd = {}
        for i in range(self.d):
            lambd[i] = lambd_values[self.c*i:self.c*(i+1)]
            
        lambd_matrix = np.array(list(lambd.values()))
        
        assert(len(gamma) == self.c)
        assert(len(lambd) == self.d)
        assert(all([(len(l) == self.c) for l in lambd.values()]))
        
        W_ = (lambd_matrix @ self.V_matrix.T) + (self.V_matrix @ lambd_matrix.T)
        W = W_ - np.diag(np.diag(W_))
        
        b_vector = self.V_matrix @ gamma
        
        linear = self.omega_array @ b_vector
        quad = np.zeros(len(self.omega_array))
        for i, j in self.W_keys:
            quad += W[i,j] * self.xx[(i, j)]
        log_scores = linear + quad
        logZ = logsumexp(log_scores)
        p = np.exp(log_scores - logZ)
        
        ll = np.sum(log_scores[self.sample_indices]) - len(self.samples) * logZ
            
        return -ll
    
    def logl_jac(self, params):
        if self.d > self.d_max_l:
            raise ValueError("d is too big for likelihood computation")
        assert(len(params) == (len(self.gamma) + self.lambd_matrix.size))
        
        gamma = np.array(params[:self.c])
        lambd_values = params[self.c:]
        n = len(self.samples)
        
        lambd = {}
        for i in range(self.d):
            lambd[i] = lambd_values[self.c*i:self.c*(i+1)]
            
        lambd_matrix = np.array(list(lambd.values()))
        
        assert(len(gamma) == self.c)
        assert(len(lambd) == self.d)
        assert(all([(len(l) == self.c) for l in lambd.values()]))
        
        W_ = (lambd_matrix @ self.V_matrix.T) + (self.V_matrix @ lambd_matrix.T)
        W = W_ - np.diag(np.diag(W_))
        
        b_vector = self.V_matrix @ gamma
        
        linear = self.omega_array @ b_vector
        quad = np.zeros(len(self.omega_array))
        for i, j in self.W_keys:
            quad += W[i,j] * self.xx[(i, j)]

        log_scores = linear + quad
        logZ = logsumexp(log_scores)
        p = np.exp(log_scores - logZ)
        
        E_x = p @ self.omega_array
        
        # E_xx_full = (p[:,None] * omega_array).T @ omega_array
        weighted = p[:, None] * self.omega_array                 # (2^d, d)
        E_xx_full = weighted.T.dot(self.omega_array)             # (d, d)

        # data sufficient stats (from samples)
        X = np.array(self.samples_array, dtype=float)            # (n,d)
        D_x = X.sum(axis=0)                                      # (d,)

        # ---- jac for gamma (c,)
        D_gamma = self.V_matrix.T.dot(D_x)                                   # (c,)
        E_gamma = self.V_matrix.T.dot(E_x)                                   # (c,)
        jac_gamma = D_gamma - n * E_gamma                        # (c,)

        # ---- jac for lambd (d, c)
        # S_matrix: row s = sum_j v_j * x_sj  -> X @ V  (n, c)
        S_matrix = X.dot(self.V_matrix)                                      # (n, c)
        term1 = X.T.dot(S_matrix)                                # (d, c)  (sum_s x_sm * S_s)
        term2 = D_x[:, None] * self.V_matrix                                 # (d, c)  (D_x[m] * v_m)
        D_lambd = term1 - term2                                  # (d, c)

        E_term1 = E_xx_full.dot(self.V_matrix)                               # (d, c)
        E_term2 = E_x[:, None] * self.V_matrix                               # (d, c)
        E_lambd = E_term1 - E_term2                              # (d, c)

        jac_lambd = D_lambd - n * E_lambd                        # (d, c)

        # assemble
        jac = np.concatenate([jac_gamma, jac_lambd.flatten(order='C')])
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
        if self.d > self.d_max_l:
            raise ValueError("d is too big for likelihood computation")
        assert(len(params) == (len(self.gamma) + self.lambd_matrix.size))
        
        gamma = np.array(params[:self.c])
        lambd_values = params[self.c:]
        
        lambd = {}
        for i in range(self.d):
            lambd[i] = lambd_values[self.c*i:self.c*(i+1)]
            
        lambd_matrix = np.array(list(lambd.values()))
        
        assert(len(gamma) == self.c)
        assert(len(lambd) == self.d)
        assert(all([(len(l) == self.c) for l in lambd.values()]))
        
        W_ = (lambd_matrix @ self.V_matrix.T) + (self.V_matrix @ lambd_matrix.T)
        W = W_ - np.diag(np.diag(W_))
        
        b_vector = self.V_matrix @ gamma
        
        U = np.dot(self.samples_array, W) + b_vector
        
        lpl = (self.samples_array*U - softplus(U)).sum()
            
        return -lpl
    
    def logpl_jac(self, params):
        
        if self.d > self.d_max_l:
            raise ValueError("d is too big for likelihood computation")
        assert(len(params) == (len(self.gamma) + self.lambd_matrix.size))
        
        gamma = np.array(params[:self.c])
        lambd_values = params[self.c:]
        
        lambd = {}
        for i in range(self.d):
            lambd[i] = lambd_values[self.c*i:self.c*(i+1)]
            
        lambd_matrix = np.array(list(lambd.values()))
        
        assert(len(gamma) == self.c)
        assert(len(lambd) == self.d)
        assert(all([(len(l) == self.c) for l in lambd.values()]))
        
        W_ = (lambd_matrix @ self.V_matrix.T) + (self.V_matrix @ lambd_matrix.T)
        W = W_ - np.diag(np.diag(W_))
        
        b_vector = self.V_matrix @ gamma
        
        U = np.dot(self.samples_array, W) + b_vector
        resid = self.samples_array - expit(U)
        
        # ---- gamma gradient
        jac_gamma = self.V_matrix.T @ resid.sum(axis=0)   # (c,)

        # ---- lambda gradient
        XV = self.samples_array @ self.V_matrix                     # (n,c)
        RV = resid @ self.V_matrix                # (n,c)

        term_diag = resid.T @ XV                  # (d,c)

        rx_sum = (resid * self.samples_array).sum(axis=0)[:, None] # (d,1)
        term_diag -= 2.0 * rx_sum * self.V_matrix       # (d,c)

        term_off = self.samples_array.T @ RV                       # (d,c)

        jac_lambd = term_diag + term_off           # (d,c)

        jac = np.concatenate([jac_gamma, jac_lambd.flatten()])
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
        
