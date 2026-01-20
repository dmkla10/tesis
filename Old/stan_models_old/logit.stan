data {
  int<lower=0> N;
  int<lower=0> U;                       // Cantidad de atributos de los nodos
  int<lower=0> W;                       // Cantidad de atributos de los enlaces
  int<lower=0> I;                       // Cantidad de nodos
  int<lower=0> J;                       // Cantidad de categorías de enlaces   
  int<lower=0> Q;
  array[N] vector[U] u;                        // Atributos de los nodos para cada observacion
  array[I] vector[U] x;                        // Atributos de los nodos para cada nodo
  array[N] vector[W] w;                        // Atributos de los enlaces para cada observacion
  array[N] int<lower=1, upper=I> i;     // Indice del nodo (sujeto, producto, etc)
  array[N] int<lower=1, upper=J> j;     // Indice de categoría del enlace (tipo de delito, comprador, etc)
  array[N] vector[Q] q;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[U] gamma;     // Propensión de las características de los nodos
  array[I] vector[W] beta;     // Preferencias de los nodos por los atributos de los enlaces
  vector[U] theta1;
  vector[U] theta2;
  array[I] vector[Q] lambda;
  real theta01;
  real theta02;
}
model {
  theta01 ~ normal(0,3);
  theta02 ~ normal(0,3);
  theta1 ~ normal(0,3);
  theta2 ~ normal(0,3);
  gamma ~ normal(0,3);
  
  for (ii in 1:I) {
    beta[ii] ~ normal(dot_product(theta01 + x[ii],theta1), 1);
    lambda[ii] ~ normal(dot_product(theta02 + x[ii],theta2), 1);
  }
  for (n in 1:N) {
    real node_propension   = dot_product(u[n], gamma);
    real link_preference   = dot_product(w[n], beta[i[n]]);
    real group_preference = dot_product(q[n], lambda[i[n]]);
    y[n] ~ bernoulli_logit(link_preference + node_propension + group_preference);
  }
}