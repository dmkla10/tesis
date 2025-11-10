// fvbm_pseudoposterior.stan
data {
  int<lower=1> p;          // number of variables (rows)
  int<lower=1> N;          // number of samples (columns)
  array[p,N] int<lower=-1, upper=1> X;
  real<lower=0> prior_sd_w;     // prior sd for off-diagonal W entries
  real<lower=0> prior_sd_b;     // prior sd for biases
}

transformed data {
  // Size of the w parameters
  int n_off = (p * (p - 1)) / 2;

  // Indexes to create the symetric W matrix
  array[n_off] int iu1;
  array[n_off] int iu2;
  int pos = 1;
  for (i in 1:(p-1)) {
    for (j in (i+1):p) {
        iu1[pos] = i;
        iu2[pos] = j;
        pos = pos + 1;
      }
  } 
}


parameters {
  vector[n_off] w_off;  // Interactions
  vector[p] b;          // biases
}

transformed parameters {
  matrix[p,p] W;
  
  // Dejamos la diagonal como cero
  for (i in 1:p)
    for (j in 1:p)
      W[i,j] = 0.0;

  for (pos2 in 1:n_off) {
    W[iu1[pos2], iu2[pos2]] = w_off[pos2];
    W[iu2[pos2], iu1[pos2]] = w_off[pos2];
  }
}

model {
  // Priors (proper)
  w_off ~ normal(0, prior_sd_w);
  b     ~ normal(0, prior_sd_b);

  // Pseudo-likelihood: sum over samples (columns) and variables (rows)
  // Each conditional P(X[i,n] | X[-i,n]) has logit u = b[i] + sum_k W[k,i] * X[k,n]
  for (n in 1:N) {
    for (i in 1:p) {
      real u;
      u = 0.0;
      // compute dot product W[:,i] . X[:,n]  
      for (k in 1:p)
        u = u + W[k,i] * X[k,n]; // Note: W[i,i] = 0, so self-term contributes 0
      u = u + b[i];
      //target += bernoulli_logit_lpmf(X[i,n] | u); // add conditional log-prob
      target += -log1p_exp(-2.0 * X[i,n] * u);
    }
  }
}