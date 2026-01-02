#ifndef RSG_H
#define RSG_H

#include <Rcpp.h>

Rcpp::List rsg (List adj_list,
               std::vector<int> adj_length,
               std::vector<double> population,
               int Ndistrict,
               double target_pop,
               double thresh,
               int maxiter
               );

#endif
