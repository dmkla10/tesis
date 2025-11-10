data {
  int<lower=0> N;              // cantidad de observaciones
  int<lower=0> K;              // cantidad de covariables
  array [N] int<lower=0> y;    // Cantidad observada
  matrix[N, K] X;              // Matriz de covariables

  int<lower=0> N_new;
  matrix[N_new, K] X_new;
}
parameters {
  real beta0;
  vector[K] beta;              // coeficientes del modelo lineal
  real<lower=0> phi;           // parametro de sobredispersion
  vector<lower=1>[N] lambda;   // intensidades Poisson latentes (una por observación)
}
transformed parameters {
  vector[N] mu = exp(X * beta + beta0);     // media del modelo lineal
  vector[N] alpha = mu .* mu ./ phi; // parámetro shape de la gamma
  vector[N] beta_g = mu ./ phi;      // parámetro rate de la gamma
}
model {
  // Priors
  beta ~ normal(0, 4);
  phi ~ gamma(2, 1);

  // prior sobre gamma
  lambda ~ gamma(alpha, beta_g);

  // Likelihood: y | lambda
  y ~ poisson(lambda);
}

generated quantities {
  vector[N_new] mu_new = exp(X_new * beta + beta0);             // media del modelo lineal para nuevos datos
  vector[N_new] alpha_new = mu_new .* mu_new ./ phi;    // shape de la gamma para nuevos datos
  vector[N_new] beta_new = mu_new ./ phi;               // rate de la gamma para nuevos datos
  vector[N_new] lambda_new;
  array[N_new] int y_new;

  for (n in 1:N_new) {
    lambda_new[n] = gamma_rng(alpha_new[n], beta_new[n]);     // muestreo de lambda posterior predictivo
    y_new[n] = poisson_rng(lambda_new[n]);                    // muestreo de y nuevo
  }
}