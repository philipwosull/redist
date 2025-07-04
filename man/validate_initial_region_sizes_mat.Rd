% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/redist_alg_helpers.R
\name{validate_initial_region_sizes_mat}
\alias{validate_initial_region_sizes_mat}
\title{Checks that a region size matrix is valid}
\usage{
validate_initial_region_sizes_mat(
  init_dvals_mat,
  ndists,
  nsims,
  init_num_regions
)
}
\arguments{
\item{init_dvals_mat}{A ndists by nsims matrix of region sizes}

\item{ndists}{The number of final districts in the plan.}

\item{nsims}{The number of simulations being run.}

\item{init_num_regions}{The number of regions in the partial plans stored in the
\code{init_region_ids_mat}}
}
\description{
Checks that every size matrix is valid. A partial plan with \code{init_num_regions}
has a valid size matrix if it is \code{ndists} by \code{nsims} where
- For each column rows 1:\code{init_num_regions} sum to 1 and each entry is
between 1 and \code{ndists} inclusive
- For each column rows \code{init_num_regions}+1:\code{ndists} are all zero
}
\details{
Will throw an error if anything is wrong.
}
