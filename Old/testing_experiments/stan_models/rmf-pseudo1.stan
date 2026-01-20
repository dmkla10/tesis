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
  array[N] vector[W] w_array;
  for (i in 1:N) {
    w_array[i] = to_vector(row(w, i));
  }
}
parameters {
  vector[W] gamma;             // Propensión de las características de los nodos
  array[N] vector[Z] beta;     // 
  vector[W] theta;
}
model {
  gamma ~ normal(0,3);
  theta ~ normal(0,3);
  
  array[W] real theta_array;
  theta_array = to_array_1d(theta);
  for (i in 1:N) {
    beta[i] ~ normal(0,3);
    }
  
  for (j in 1:M) {
    for (i in 1:N) {
      real unitary_utility = dot_product(gamma,w[i]) + dot_product(beta[i],z[j]);
      real binary_utility = dot_product(y[j] .* ((rep_matrix(to_row_vector(w_array[i]), N) + w) * theta), onesN) - 2 * y[j][i] * dot_product(w_array[i], theta);
      to_int(y[j][i]) ~ bernoulli_logit(unitary_utility + binary_utility);
      
    }
  }
}