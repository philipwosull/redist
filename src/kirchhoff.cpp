#include "map_calc.h"

// Both of these used to delegate to redistmetrics, which required arma matrices.
// They now use the templated versions in map_calc.h, which work directly on a
// column of an Rcpp::IntegerMatrix.
// The log_st_map no longer returns infinity if a county intersect district piece is 
// disconnected. It now just throws an error

// [[Rcpp::export]]
Rcpp::NumericVector log_st_map(const Graph &g, const Rcpp::IntegerMatrix &districts,
                               const std::vector<unsigned int> &counties, int n_distr) {
    int const N = districts.ncol();

    Rcpp::NumericVector out(N);
    for (int n = 0; n < N; n++) {
        out[n] = eval_log_st(districts.column(n), g, counties, n_distr);
    }

    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector n_removed(const Graph &g, const Rcpp::IntegerMatrix &districts, int n_distr) {
    int const N = districts.ncol();

    Rcpp::NumericVector out(N);
    for (int n = 0; n < N; n++) {
        out[n] = eval_er(districts.column(n), g);
    }

    return out;
}
