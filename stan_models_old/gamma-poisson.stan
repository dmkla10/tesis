data {
  int<lower=0> N;              // cantidad de observaciones
  int<lower=0> K;              // cantidad de covariables
  array [N] int<lower=0> y;    // Cantidad observada
  matrix[N, K] X;              // Matriz de covariables
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