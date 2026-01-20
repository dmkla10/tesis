import numpy as np
from itertools import product
from scipy import stats, optimize
from scipy.special import logsumexp, softplus, expit

class FVBM():
    def __init__(self, parameters:dict,d:int):
        
        self.d = int(d)
        self.param_size = int(self.d+0.5*self.d*(self.d-1))
        self.b = {i:parameters["b"][i] for i in range(self.d)}
        self.b_vector = np.array(list(self.b.values()))
        self.theta = parameters["theta"]
        self.theta_ids = sorted(list(self.theta.keys()))
        self.theta = {key:self.theta[key] for key in self.theta_ids}
        self.theta_vector = np.array(list(self.theta.values()))
        self.param_vector = np.array(list(self.b.values())+list(self.theta.values()))
        
        
        # Checkeamos que los parámetros estén en orden
        assert(self.d == d)
        assert(len(self.theta) == 0.5*self.d*(self.d-1))
        assert(len(self.b) == self.d)
        for i in range(self.d):
            for j in range(i):
                assert(((i,j) in self.theta_ids) ^ ((j,i) in self.theta_ids))
        
        W = np.zeros(shape=(self.d,self.d))
        for key in self.theta.keys():
            W[key] = self.theta[key]
            W[tuple(reversed(key))] = self.theta[key]
        self.W = W
        
        self.omega = list(product([0,1], repeat=self.d))
        self.omega_list_array =[ (w,np.array(w)) for w in self.omega]
        self.omega_array = np.array(self.omega)
        self.id_to_omega = {i:elem for i, elem in enumerate(self.omega)}
        
        self.xx = {
            (i, j): self.omega_array[:, i] * self.omega_array[:, j]
            for (i, j) in self.theta_ids
        }
        
        linear = self.omega_array @ self.b_vector        # shape (2^d,)
        quad = np.zeros(len(self.omega_array))
        for k, (i, j) in enumerate(self.theta_ids):
            quad += self.theta_vector[k] * self.xx[(i, j)]

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
        self.D_xx = {key: 0.0 for key in self.theta_ids}

        for x in self.samples:
            x = np.array(x)
            self.D_x += x
            for (i, j) in self.theta_ids:
                self.D_xx[(i, j)] += x[i] * x[j]
        return "Done."
        
    def logl(self, params, minimizing:bool=True):
        """
        Compute the likelihood for a set of parameters
        """
        assert(len(params) == (len(self.b) + len(self.theta)))
        
        b_values = params[:self.d]
        theta_values = params[self.d:]
        theta = {}
        b = {}
        for i,key in enumerate(self.theta.keys()):
            theta[key] = theta_values[i]
        for i,key in enumerate(self.b.keys()):
            b[key] = b_values[i]
        
        assert(len(theta) == len(self.theta))
        assert(len(b) == len(self.b))
        
        W = np.zeros(shape=(self.d,self.d))
        for key in theta.keys():
            W[key] = theta[key]
            W[tuple(reversed(key))] = theta[key]
        b = {i:b[i] for i in range(self.d)}
        b_vector = np.array(list(b.values()))
        
        linear = self.omega_array @ b_vector
        quad = np.zeros(len(self.omega_array))
        for k, (i, j) in enumerate(self.theta_ids):
            quad += theta_values[k] * self.xx[(i, j)]
        log_scores = linear + quad
        logZ = logsumexp(log_scores)
        p = np.exp(log_scores - logZ)
        
        ll = np.sum(log_scores[self.sample_indices]) - len(self.samples) * logZ
            
        return -ll
    
    def logl_jac(self, params, minimizing:bool=True):
        assert(len(params) == (len(self.b) + len(self.theta)))
        
        b_values = params[:self.d]
        theta_values = params[self.d:]
        theta = {}
        b = {}
        for i,key in enumerate(self.theta.keys()):
            theta[key] = theta_values[i]
        for i,key in enumerate(self.b.keys()):
            b[key] = b_values[i]
        
        assert(len(theta) == len(self.theta))
        assert(len(b) == len(self.b))
        
        W = np.zeros(shape=(self.d,self.d))
        for key in theta.keys():
            W[key] = theta[key]
            W[tuple(reversed(key))] = theta[key]
        b = {i:b[i] for i in range(self.d)}
        b_vector = np.array(list(b.values()))
        
        linear = self.omega_array @ b_vector
        quad = np.zeros(len(self.omega_array))
        for k, (i, j) in enumerate(self.theta_ids):
            quad += theta_values[k] * self.xx[(i, j)]

        log_scores = linear + quad
        logZ = logsumexp(log_scores)
        p = np.exp(log_scores - logZ)
        
        E_x = p @ self.omega_array
        jac_b = self.D_x - len(self.samples) * E_x
                
        jac_theta = np.array([
            self.D_xx[(i, j)] - len(self.samples) * np.dot(p, self.xx[(i, j)])
            for (i, j) in self.theta_ids
        ])

        jac = np.zeros_like(params)
        jac[:self.d] = jac_b
        jac[self.d:] = jac_theta
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
        assert(len(params) == (len(self.b) + len(self.theta)))
        
        b_values = params[:self.d]
        theta_values = params[self.d:]
        theta = {}
        b = {}
        for i,key in enumerate(self.theta.keys()):
            theta[key] = theta_values[i]
        for i,key in enumerate(self.b.keys()):
            b[key] = b_values[i]
        
        assert(len(theta) == len(self.theta))
        assert(len(b) == len(self.b))
        
        W = np.zeros(shape=(self.d,self.d))
        for key in theta.keys():
            W[key] = theta[key]
            W[tuple(reversed(key))] = theta[key]
        b = {i:b[i] for i in range(self.d)}
        b_vector = np.array(list(b.values()))
        
        U = np.dot(self.samples_array, W) + b_vector
        
        lpl = (self.samples_array*U - softplus(U)).sum()
            
        return -lpl
    
    def logpl_jac(self, params):
        
        assert(len(params) == (len(self.b) + len(self.theta)))
        
        b_values = params[:self.d]
        theta_values = params[self.d:]
        theta = {}
        b = {}
        for i,key in enumerate(self.theta.keys()):
            theta[key] = theta_values[i]
        for i,key in enumerate(self.b.keys()):
            b[key] = b_values[i]
        
        assert(len(theta) == len(self.theta))
        assert(len(b) == len(self.b))
        
        W = np.zeros(shape=(self.d,self.d))
        for key in theta.keys():
            W[key] = theta[key]
            W[tuple(reversed(key))] = theta[key]
        b = {i:b[i] for i in range(self.d)}
        b_vector = np.array(list(b.values()))
        
        U = np.dot(self.samples_array, W) + b_vector
        resid = self.samples_array - expit(U)
        b_jac = resid.sum(axis=0)
        
        W_jac = np.dot(self.samples_array.T,resid)
        w_jac = np.array([W_jac[i, j] for (i, j) in self.theta_ids])
        
        jac = np.concatenate([b_jac, w_jac])
        
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
        
