data {
  int<lower=1> D;
  int<lower=0> N;
  int<lower=1> L;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=0, upper=L> ll;
  array[N] row_vector[D] x;
}
parameters {
  array[D] real mu;
  array[D] real<lower=0> sigma;
  array[L] vector[D] beta;
}
model {
  for (d in 1:D) {
    mu[d] ~ normal(0, 100);
    for (l in 1:L)
      beta[l,d] ~ normal(mu[d], sigma[d]);
  }
  for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(x[n] * beta[ll[n]]));
}