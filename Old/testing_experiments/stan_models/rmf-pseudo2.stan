data {
  int<lower=0> N; // Cantidad de nodos
  int<lower=0> M; // Cantidad de enlaces observados
  int<lower=1> W; // Cantidad de atributos de los nodos
  int<lower=1> Z; // Cantidad de atributos de los enlaces
  array[M] vector[N] y; // Vectores de incidencia (de la matriz de incidencia)
  matrix[N,W] w; // vector de atributos de los nodos (univariados)
  array[M] vector[Z] z; // vector de atributos de los enlaces (univariados)
}
transformed data {
  vector[N] onesN = rep_vector(1.0, N);
  vector[W] onesW = rep_vector(1.0, W);
}
parameters {
  vector[W] gamma;             // Propensión de las características de los nodos
  array[N] vector[Z] beta;     // 
  matrix[N,W] theta;
}
model {
  gamma ~ normal(0,3);
  array[N] vector[W] theta_array;
  array[N] vector[W] w_array;
  
  for (i in 1:N) {
    beta[i] ~ normal(0,3);
    theta[i] ~ normal(0,3);
    theta_array[i] = to_vector(row(theta, i));
    w_array[i] = to_vector(row(w, i));
    }
  
  for (j in 1:M) {
    for (i in 1:N) {
      real unitary_utility = dot_product(gamma,w[i]) + dot_product(beta[i],z[j]);
      real binary_utility = dot_product(y[j] .* ((theta * w_array[i]) + (w*theta_array[i])), onesN) - 2 * y[j][i] * dot_product(w_array[i],theta_array[i]);
      to_int(y[j][i]) ~ bernoulli_logit(unitary_utility + binary_utility);
    }
  }
}