data {
  int K;
  int N;
  int U;
  array[N] int y;
  array[N] int<lower=1, upper=U> uu;
}
parameters {
  vector[K-1] raw_alpha; 
  array[U] real u_alpha;
}
transformed parameters {
   vector[K] alpha = append_row(0, raw_alpha);
}

model {
  alpha ~ normal(0, 5);
  u_alpha ~ normal(0, 5);
  for (n in 1:N) {
    y[n] ~ categorical_logit(alpha + u_alpha[uu[n]]); // el u_alpha no hace nada xd revisar paper de skander
  }
}