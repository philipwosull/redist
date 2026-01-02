#pragma once
#ifndef GRAPH_OP_H
#define GRAPH_OP_H

#include <vector>
#include <Rcpp.h>
#include <Rcpp.h>
#include "redist_types.h"



// [[Rcpp::export]]
RegionMultigraphCount get_region_multigraph(
    Rcpp::List const &adj_list,
    std::vector<unsigned int> const &region_ids
);

// [[Rcpp::export]]
Eigen::MatrixXd get_region_laplacian(
    Rcpp::List const &adj_list,
    std::vector<unsigned int> const &region_ids
);





RegionMultigraphCount build_region_multigraph(
    Graph const &g, 
    PlanVector const &region_ids,
    int const num_regions
);


#endif
