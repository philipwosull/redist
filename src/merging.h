#pragma once
#ifndef MERGING_H
#define MERGING_H

#include "advanced_types.h"
#include "base_plan_type.h"
#include "splitting_schedule_types.h"

#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <memory>

class Plan;
class PlanMultigraph;
class ScoringFunction;
class SplittingSchedule;
class ScheduleContext;
class RNGState;
class USTSampler;
class TreeSplitter;
class WeightCache;

// Simple struct for tracking granular time
struct GranularMCMCTimes {
    double wilson_time = 0.0;
    double backfill_wilson_time = 0.0;
    double get_valid_mergepairs = 0.0;
    double selecting_merge_pair = 0.0;
    double hard_constraint_time = 0.0;
    double eff_boundary_length = 0.0;
    double region_scores = 0.0;
    double plan_scores = 0.0;
    double tau_terms = 0.0;
    double plan_copying = 0.0;
};


// Essentially a wrapper for all everything merge-split keeps per plan, 
// for either the current or the proposed plan
struct PlanState {
    PlanState(MapParams const &map_params, bool need_to_compute_multigraph_taus)
        : multigraph(map_params, need_to_compute_multigraph_taus) {}

    PlanMultigraph multigraph;
    ScheduleContext schedule_ctx;
    std::vector<std::pair<RegionID, RegionID>> merge_pairs;
    std::vector<double> pair_wgts;  // unnormalized, aligned with merge_pairs
    // log number of linking edges of the plan (its log multigraph tau). Only
    // computed in linking edge space; NaN until compute_log_multigraph_tau
    // sets it, and reset to NaN whenever build_merge_pairs rebuilds the
    // multigraph, so a stale read shows up as a non finite MH ratio
    double log_multigraph_tau = std::numeric_limits<double>::quiet_NaN();

    // Builds schedule_ctx for `plan`, then the multigraph and merge_pairs.
    // Returns false if the multigraph could not be built (the plan is not
    // hierarchically valid), in which case merge_pairs should not be used.
    // Time goes to granular_times.get_valid_mergepairs.
    bool build_merge_pairs(Plan const &plan, SplittingSchedule const &splitting_schedule,
                           ScoringFunction const &scoring_function,
                           GranularMCMCTimes &granular_times);

    // Unnormalized selection weight of each pair in merge_pairs, so
    // build_merge_pairs must come first. Time goes to
    // granular_times.selecting_merge_pair.
    void compute_pair_wgts(Plan const &plan, std::string const &merge_prob_type,
                           GranularMCMCTimes &granular_times);

    // Sets log_multigraph_tau from the multigraph, so build_merge_pairs must
    // come first. Does nothing unless multigraph.need_to_compute_multigraph_taus
    // (linking edge space). Not timed here, since callers time it with the
    // other effective boundary terms.
    void compute_log_multigraph_tau(Plan const &plan, ScoringFunction const &scoring_function);
};


// Holds PlanState for both the current and proposed plan 
// and allows for easy memoization of certain values 
class MergeSplitStates {
  public:
    MergeSplitStates(MapParams const &map_params, bool need_to_compute_multigraph_taus)
        : current(std::make_unique<PlanState>(map_params, need_to_compute_multigraph_taus)),
          proposed(std::make_unique<PlanState>(map_params, need_to_compute_multigraph_taus)) {}

    std::unique_ptr<PlanState> current; 
    std::unique_ptr<PlanState> proposed;

    // swaps the current and proposed pointer upon acceptance
    void accept() { std::swap(current, proposed); }

    // build_context + attempt_to_get_valid_mergesplit_pairs +
    // get_adj_pair_unnormalized_weights in one, for the current plan, plus its
    // log multigraph tau in linking edge space (timed as eff_boundary_length).
    // Throws if the multigraph could not be built or there are no merge pair
    // weights, since both should always succeed for the current plan
    void build_current_plan_context(Plan const &plan, SplittingSchedule const &splitting_schedule,
                                    ScoringFunction const &scoring_function,
                                    std::string const &merge_prob_type,
                                    GranularMCMCTimes &granular_times);

};


std::vector<double> get_adj_pair_unnormalized_weights(
    Plan const &plan, std::vector<std::pair<RegionID, RegionID>> const &valid_region_adj_pairs,
    std::string const &selection_type);

// `states.current` must be built for `plan` (build_current_plan_context).
// `states.proposed` is rebuilt for the proposed plan, and states.accept() is
// called if the proposal is accepted so `states.current` always matches `plan`.
std::tuple<bool, bool, double, int> attempt_mergesplit_step(
    MapParams const &map_params, const SplittingSchedule &splitting_schedule,
    ScoringFunction const &scoring_function, RNGState &rng_state,
    SamplingSpace const sampling_space, Plan &plan, Plan &new_plan, USTSampler &ust_sampler,
    TreeSplitter &tree_splitter, MergeSplitStates &states, std::string const merge_prob_type,
    bool save_edge_selection_prob, double const rho, bool const do_mh,
    bool const using_caching, WeightCache *weight_cache,
    GranularMCMCTimes &granular_times);

int run_merge_split_steps(MapParams const &map_params,
                          const SplittingSchedule &splitting_schedule,
                          ScoringFunction const &scoring_function, RNGState &rng_state,
                          SamplingSpace const sampling_space, Plan &plan, Plan &dummy_plan,
                          USTSampler &ust_sampler, TreeSplitter &tree_splitter,
                          MergeSplitStates &states,
                          std::string const merge_prob_type, double const rho,
                          int num_steps_to_run,
                          std::vector<int> &tree_sizes, std::vector<int> &successful_tree_sizes,
                          bool const using_caching, WeightCache *weight_cache, 
                          GranularMCMCTimes &granular_times);

#endif
