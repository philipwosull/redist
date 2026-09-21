#pragma once
#ifndef SMC_ALG_HELPERS_H
#define SMC_ALG_HELPERS_H


#include <memory>
#include <string_view>
#include <vector>
#include <Rcpp.h>

#include "advanced_types.h"
#include "base_plan_type.h"



class TreeSplitter;
class SplittingSchedule;
class RNGState;

namespace RcppThread {
    class ThreadPool;
}


Rcpp::List maximum_input_sizes();

/*
 * Convert zero-indxed R adjacency list to Graph object (vector of vectors of ints).
 */
Graph list_to_graph(const Rcpp::List &l);

/*
 * Creates a reindexing vector for the plan with two regions merged
 * does this by making region2_id map to region1_id. Nothing else
 * changes
 */
void set_merged_region_reindex_vec(int const num_regions, std::vector<int> &region_reindex_vec,
                                   int const region1_id, int const region2_id);

/*
 *  Reorders all the plans in the vector by order a region was split
 *
 *  Takes a vector of plans and uses the vector of dummy plans to reorder
 *  each of the plans by the order a region was split.
 *
 *
 *  @title Reorders all the plans in the vector by order a region was split
 *
 *  @param pool A threadpool for multithreading
 *  @param plan_ptrs_vec A vector of pointers to plans
 *  @param dummy_plans_vec A vector of pointers to dummy plans
 *
 *  @details Modifications
 *     - Each plan in the `plans_vec` object is reordered by when the region was split
 *     - Each plan is a shallow copy of the plans in `plans_vec`
 *
 */
void reorder_all_plans(RcppThread::ThreadPool &pool,
                       std::vector<std::unique_ptr<Plan>> &plan_ptrs_vec,
                       std::vector<std::unique_ptr<Plan>> &dummy_plan_ptrs_vec);

std::vector<std::unique_ptr<TreeSplitter>>
get_tree_splitter_ptrs(MapParams const &map_params, SplittingMethodType const splitting_method,
                       SamplingSpace const sampling_space,
                       Rcpp::List const &control, int const nsims, int const num_threads);

// lightweight container for plans
class PlanEnsemble {

  public:
    // constructor for empty plans
    PlanEnsemble(MapParams const &map_params, int const total_pop, int const nsims,
                 SamplingSpace const sampling_space, RcppThread::ThreadPool &pool,
                 int const verbosity = 3);
    // constructor for non-empty starting plans
    PlanEnsemble(MapParams const &map_params, SplittingSchedule const &splitting_schedule,
                 int const num_regions, int const nsims, SamplingSpace const sampling_space,
                 Rcpp::IntegerMatrix const &plans_mat,
                 Rcpp::IntegerMatrix const &region_sizes_mat, std::vector<RNGState> &rng_states,
                 RcppThread::ThreadPool &pool, int const verbosity = 3);

    int const nsims;
    int const V;
    int const ndists;
    int const total_seats;
    SamplingSpace const sampling_space;
    std::vector<RegionID> flattened_all_plans;
    std::vector<RegionID> flattened_all_region_sizes;
    std::vector<int> flattened_all_region_pops;
    std::vector<int> flattened_all_region_order_added;
    // Empty unless sampling space is ForestSpace or LinkingEdgeSpace.
    int const num_forest_edge_bit_words_per_plan;
    std::vector<EdgeBitWord> flattened_all_forest_edge_bits;
    std::vector<std::unique_ptr<Plan>> plan_ptr_vec;

    // exports current plans to 1-indexed Rcpp matrix
    Rcpp::IntegerMatrix get_R_plans_matrix();
    // export current region sizes to Rcpp matrix
    Rcpp::IntegerMatrix get_R_sizes_matrix(RcppThread::ThreadPool &pool);
    // export current region populations to Rcpp matrix
    Rcpp::IntegerMatrix get_region_pops_matrix(RcppThread::ThreadPool &pool);
    // counts the number of unique plans in the ensemble
    int count_unique_plans(RcppThread::ThreadPool &pool) const;

    // debugging methods
    // checks all plans are valid. 
    void check_all_plans_valid(
        MapParams const &map_params,
        std::string_view where
    );
};

PlanEnsemble get_plan_ensemble(
    MapParams const &map_params, SplittingSchedule const &splitting_schedule,
    int const num_regions, int const nsims, SamplingSpace const sampling_space,
    Rcpp::IntegerMatrix const &plans_mat, Rcpp::IntegerMatrix const &region_sizes_mat,
    std::vector<RNGState> &rng_states, RcppThread::ThreadPool &pool, int const verbosity);

std::unique_ptr<PlanEnsemble> get_plan_ensemble_ptr(
    MapParams const &map_params, SplittingSchedule const &splitting_schedule,
    int const num_regions, int const nsims, SamplingSpace const sampling_space,
    Rcpp::IntegerMatrix const &plans_mat, Rcpp::IntegerMatrix const &region_sizes_mat,
    std::vector<RNGState> &rng_states, RcppThread::ThreadPool &pool, int const verbosity);


// converts trees to compact edge list form 
// This only supports undirected trees 
std::vector<bool> vector_tree_to_edge_vector(
    GraphEdgeIndex const &edge_index,
    Tree const &tree
);

// Converts a graph edge index to an R deciperable list
// where its a list of length edge_index.num_edges
// and each element is the pair (u,v) of the vertices in the 
// edge it represents 
Rcpp::List graph_edge_index_to_list(
    GraphEdgeIndex const &edge_index
);


#endif
