// [[Rcpp::depends(redistmetrics)]]

#include "smc_base.h"
#include <redistmetrics.h>

// [[Rcpp::export]]
NumericVector log_st_map(const Graph &g, const Eigen::MatrixXi &districts,
                         const std::vector<unsigned int> &counties, int n_distr) {
    return redistmetrics::log_st_map(g, districts, counties, n_distr);
}

// [[Rcpp::export]]
NumericVector n_removed(const Graph &g, const Eigen::MatrixXi &districts, int n_distr) {
    return redistmetrics::n_removed(g, districts, n_distr);
}
