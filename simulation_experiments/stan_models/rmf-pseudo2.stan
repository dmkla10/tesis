data {
  int<lower=0> N; // Cantidad de nodos
  int<lower=0> M; // Cantidad de enlaces observados
  array[M] vector[N] y; // Vectores de incidencia (de la matriz de incidencia)
  array[N] real w; // vector de atributos de los nodos (univariados)
  vector[M] z; // vector de atributos de los enlaces (univariados)
}
transformed data {
  vector[N] w_vec = to_vector(w);
  vector[N] onesN = rep_vector(1.0, N);
}
parameters {
  real gamma;     // Propensión de las características de los nodos
  array[N] real beta;
  vector[N] theta;
}
model {
  gamma ~ normal(0,3);
  beta ~ normal(0,3);
  theta ~ normal(0,3);
  array[N] real theta_array;
  theta_array = to_array_1d(theta);
  
  for (j in 1:M) {
    for (i in 1:N) {
      real unitary_utility = gamma*w[i] + beta[i]*z[j];
      real binary_utility = dot_product(y[j] .* (w_vec*theta_array[i]) + (w[i]*theta), onesN) - (2*w[i]*theta_array[i])*y[j][i];
      to_int(y[j][i]) ~ bernoulli_logit(unitary_utility + binary_utility);
    }
  }
}