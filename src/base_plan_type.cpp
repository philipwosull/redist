/********************************************************
 * Author: Philip O'Sullivan
 * Institution: Harvard University
 * Date Created: 2025/01
 * Purpose: Implements base Plan (ie graph partition) class
 ********************************************************/


#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>

#include <Rcpp.h>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base_plan_type.h"


#include "R_ext/Print.h"
#include "advanced_types.h"
#include "redist_constants.h"
#include "redist_types.h"
#include "sparse_logdet.h"
#include "scoring.h"
#include "splitting_schedule_types.h"
#include "tree_op.h"
#include "utils.h"
#include "random.h"
#include "wilson.h"

bool constexpr DEBUG_BASE_PLANS_VERBOSE = false;
bool constexpr DEBUG_LOG_LINK_EDGE_VERBOSE = false;


bool Plan::check_region_pop_valid(MapParams const &map_params, int const region_id) const {
    auto region_pop = region_pops[region_id];
    auto region_size = region_sizes[region_id];
    auto region_pop_lb = map_params.lower * region_size;
    auto region_pop_ub = map_params.upper * region_size;

    return region_pop_lb <= region_pop && region_pop <= region_pop_ub;
}

std::pair<bool, std::vector<int>>
Plan::all_region_pops_valid(MapParams const &map_params) const {

    bool all_valid = true;
    std::vector<int> bad_regions;

    for (int region_id = 0; region_id < num_regions; region_id++) {
        if (!check_region_pop_valid(map_params, region_id)) {
            all_valid = false;
            bad_regions.push_back(region_id);
        }
    }

    return std::make_pair(all_valid, bad_regions);
}

std::pair<bool, std::vector<int>>
Plan::all_regions_connected(Graph const &g, CircularQueue<int> &vertex_queue,
                            std::vector<bool> &vertices_visited,
                            std::vector<bool> &regions_visited) {
    std::fill(regions_visited.begin(), regions_visited.end(), false);
    std::fill(vertices_visited.begin(), vertices_visited.end(), false);
    vertex_queue.clear();
    int const V = g.size();

    std::set<int> disconnected_regions;

    for (int v = 0; v < V; v++) {
        // skip if we've already visited this
        if (vertices_visited[v]) {
            continue;
        }
        auto v_region = region_ids[v];
        // If we've already visited this region but not a vertex
        // then the region is disconnected
        if (regions_visited[v_region] && !vertices_visited[v]) {
            disconnected_regions.insert(v_region);
        }
        // mark this vertex as visited
        vertices_visited[v] = true;
        // mark this region as visited
        regions_visited[v_region] = true;
        vertex_queue.push(v);
        // now visit all its neighbors
        while (!vertex_queue.empty()) {
            auto u = vertex_queue.pop();
            // mark this as visited
            vertices_visited[u] = true;

            // add unvisited neighbors
            for (auto const &u_nbor : g[u]) {
                if (vertices_visited[u_nbor])
                    continue;
                auto u_nbor_region = region_ids[u_nbor];
                if (u_nbor_region == v_region) {
                    vertex_queue.push(u_nbor);
                    vertices_visited[u_nbor] = true;
                }
            }
        }
    }

    std::vector<int> disconnected_region_output(disconnected_regions.begin(),
                                                disconnected_regions.end());
    bool any_disconnected = disconnected_region_output.size() > 0;

    return std::make_pair(any_disconnected, disconnected_region_output);
}

// checks the inputted plan has the number of regions it claims it does
// checks the sizes and that the labels make sense.
// if makes sense then it counts the number of districts and multidistricts
// NOTE: This needs to be updated for multimember districts 
void Plan::check_inputted_region_sizes(int ndists, bool split_district_only) const {
    int total_size_implied_by_sizes_mat = 0;

    for (int i = 0; i < num_regions; ++i) {
        int const size_i = static_cast<int>(region_sizes[i]);

        if (size_i <= 0) {
            std::ostringstream oss;
            oss << "Region size input for active region is 0 or less.\n";
            oss << "region_id=" << i << "\n";
            oss << "region_sizes[" << i << "]=" << size_i << "\n";
            oss << "num_regions=" << num_regions << "\n";
            oss << "ndists=" << ndists << "\n";
            oss << debug_string(true);

            throw std::runtime_error(oss.str());
        }

        total_size_implied_by_sizes_mat += size_i;

        if (split_district_only && i != num_regions - 1 && size_i != 1) {
            std::ostringstream oss;
            oss << "For partial plan, all non-remainder regions must have size 1.\n";
            oss << "region_id=" << i << "\n";
            oss << "region_sizes[" << i << "]=" << size_i << "\n";
            oss << "num_regions=" << num_regions << "\n";
            oss << "remainder_region_id=" << (num_regions - 1) << "\n";
            oss << debug_string(true);

            throw std::runtime_error(oss.str());
        }
    }

    if (total_size_implied_by_sizes_mat != ndists) {
        std::ostringstream oss;
        oss << "Sum of region sizes does not equal ndists.\n";
        oss << "sum_active_region_sizes=" << total_size_implied_by_sizes_mat << "\n";
        oss << "ndists=" << ndists << "\n";
        oss << "num_regions=" << num_regions << "\n";
        oss << debug_string(true);

        throw std::runtime_error(oss.str());
    }

    for (int i = num_regions; i < ndists; ++i) {
        int const size_i = static_cast<int>(region_sizes[i]);

        if (size_i != 0) {
            std::ostringstream oss;
            oss << "Inactive region has nonzero size.\n";
            oss << "region_id=" << i << "\n";
            oss << "region_sizes[" << i << "]=" << size_i << "\n";
            oss << "num_regions=" << num_regions << "\n";
            oss << "ndists=" << ndists << "\n";
            oss << debug_string(true);

            throw std::runtime_error(oss.str());
        }
    }
}

void Plan::check_inputted_region_ids(int ndists) const {
    std::unordered_set<int> unique_ids;
    unique_ids.reserve(num_regions);

    for (std::size_t i = 0; i < region_ids.size(); ++i) {
        int const rid = static_cast<int>(region_ids[i]);

        if (rid < 0 || rid >= num_regions) {
            std::ostringstream oss;
            oss << "Invalid region id in inputted plan.\n";
            oss << "vertex=" << i << "\n";
            oss << "region_ids[" << i << "]=" << rid << "\n";
            oss << "num_regions=" << num_regions << "\n";
            oss << "ndists=" << ndists << "\n";
            oss << debug_string(true);

            throw std::runtime_error(oss.str());
        }

        unique_ids.insert(rid);
    }

    int const actual_num_regions = static_cast<int>(unique_ids.size());

    if (actual_num_regions != num_regions) {
        std::ostringstream oss;
        oss << "Plan did not have expected number of unique region ids.\n";
        oss << "expected num_regions=" << num_regions << "\n";
        oss << "actual_num_regions=" << actual_num_regions << "\n";
        oss << debug_string(true);

        throw std::runtime_error(oss.str());
    }

    std::vector<int> sorted_labels(unique_ids.begin(), unique_ids.end());
    std::sort(sorted_labels.begin(), sorted_labels.end());

    for (int i = 0; i < num_regions; ++i) {
        if (sorted_labels.at(i) != i) {
            std::ostringstream oss;
            oss << "Plan did not have expected consecutive region ids.\n";
            oss << "expected label=" << i << "\n";
            oss << "actual sorted label=" << sorted_labels.at(i) << "\n";
            oss << "All sorted labels: ";
            for (int const label : sorted_labels) {
                oss << label << " ";
            }
            oss << "\n";
            oss << debug_string(true);

            throw std::runtime_error(oss.str());
        }
    }
}
// Constructs existing parital plan
Plan::Plan(int const num_regions, const std::vector<unsigned int> &pop, PlanVector &this_plan_region_ids,
           RegionSizes &this_plan_region_sizes, IntPlanAttribute &this_plan_region_pops,
           IntPlanAttribute &this_plan_order_added, PlanEdgeBits &this_plan_forest_edge_bits)
    : forest_edges(this_plan_forest_edge_bits),
      region_ids(this_plan_region_ids), region_sizes(this_plan_region_sizes),
      region_pops(this_plan_region_pops), region_added_order(this_plan_order_added),
      num_regions(num_regions), region_order_max(num_regions + 2) {

    // Create other region-level information
    // fill first num_regions entries with 1,...,num_regions
    std::iota(std::begin(region_added_order), std::begin(region_added_order) + num_regions, 1);

    // // compute the population for each of the regions
    for (size_t v = 0; v < region_ids.size(); v++) {
        region_pops[region_ids[v]] += pop[v];
    }
};

// assumes that the inputted attributes are all zero
Plan::Plan(int const total_seats, int const total_pop, PlanVector &this_plan_region_ids,
           RegionSizes &this_plan_region_sizes, IntPlanAttribute &this_plan_region_pops,
           IntPlanAttribute &this_plan_order_added, PlanEdgeBits &this_plan_forest_edge_bits)
    : forest_edges(this_plan_forest_edge_bits),
      region_ids(this_plan_region_ids), region_sizes(this_plan_region_sizes),
      region_pops(this_plan_region_pops), region_added_order(this_plan_order_added),
      num_regions(1), region_order_max(total_seats + 1) {
    region_pops[0] = total_pop;
    region_sizes[0] = total_seats;
    region_added_order[0] = 1;
};

// returns the region ids of the two most recently split regions
// DO NOT CALL THIS WHEN THERE IS ONLY 1 REGION!
std::pair<int, int> Plan::get_most_recently_split_regions() const {
    int largest_index = -1, second_largest_index = -1;
    int largest_value = -1 * num_regions, second_largest_value = -1 * num_regions;

    // Iterate through the vector
    for (std::size_t i = 0; i < num_regions; ++i) {
        if (region_added_order[i] > largest_value) {
            // Update second-largest before updating largest
            second_largest_value = largest_value;
            second_largest_index = largest_index;

            largest_value = region_added_order[i];
            largest_index = i;
        } else if (region_added_order[i] > second_largest_value) {
            // Update second-largest only
            second_largest_value = region_added_order[i];
            second_largest_index = i;
        }
    }

    // always make region1 the region with the smaller id
    int region1_id = std::min(largest_index, second_largest_index);
    int region2_id = std::max(largest_index, second_largest_index);

    return std::make_pair(region1_id, region2_id);
}

// Takes a plan and reorders the regions according to the order the regions
// were split
// IT IS VERY IMPORTANT THAT THE TWO PLANS NOT POINT TO THE SAME arma::umat or everything breaks
// This breaks the dummy plan and does not copy it to be the original
void Plan::reorder_plan_by_oldest_split(Plan &dummy_plan) {

    // Recall that the region_added_order attribute is a vector that stores info
    // on the relative split order. So if entry i is greater than entry j that
    // means region i was split more recently than region j

    // For example if you had a plan with 6 regions and a region_added_order vector set to
    // {4,1,7,3,9,2} then that means
    //  - Region 0 was added 3rd
    //  - Region 1 was added 1st
    //  - Region 2 was added 5th
    //  - Region 3 was added 4rd
    //  - Region 4 was added 6th
    //  - Region 5 was added 2nd

    // want to return [3, 0, 4, 2, 5, 1]
    // std::vector<int> rel_order = {4,1,7,3,9,2};

    // get the relative order vector of the n regions
    std::vector<int> rel_order(this->region_added_order.begin(),
                               this->region_added_order.begin() + this->num_regions);

    // Create a vector of indices [0, 1, 2, ..., n-1]
    std::vector<int> indices(rel_order.size());
    std::iota(indices.begin(), indices.end(), 0);

    /*
    Sort the indices based on the values in rel_order so that means
    indices[i] < indices[j] iff rel_order[i] < rel_order[j] so that means
    indices[0] is the old (ie not updated) label of the oldest split region,
    In general indices[i] == k  means that the region with the old label of
    k will have a new id of i

    For our example with rel_order = {4,1,7,3,9,2} then that means the indices
    would be sorted to be {1, 5, 3, 0, 2, 4} so we know old region 1 is now zero,
    old region 5 is now 1, etc. Alternatively interpret as indices[i] == k means
    new region i was old region k
    */
    std::sort(indices.begin(), indices.end(),
              [&rel_order](int a, int b) { return rel_order[a] < rel_order[b]; });

    /*
    Create a vector mapping old region id to the new ordered region id. So
    `old_region_id_to_new_vec[i]` is the new region id of the region with old
    id i. In other words it means region i should be relabeled as old_region_id_to_new_vec[i]
    */

    std::vector<int> old_region_id_to_new_vec(rel_order.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        old_region_id_to_new_vec[indices[i]] = i;
    }

    dummy_plan.region_ids.copy(region_ids);
    // First we relabel all the region vertex ids
    for (size_t i = 0; i < this->region_ids.size(); i++) {
        // Send region id i to old_region_id_to_new_vec[i]
        this->region_ids[i] = old_region_id_to_new_vec.at(dummy_plan.region_ids[i]);
    }

    // Make some dummy plan attributes a shallow copy of the plan
    dummy_plan.region_sizes.copy(region_sizes);
    dummy_plan.region_pops.copy(region_pops);

    // Now we reorder the region dvals and population
    for (size_t i = 0; i < this->num_regions; i++) {
        // Recall indices[i] == k means the old region with id k now has id i
        // so we want to set the value at the new region id `i` to the value it
        // had in the old one which is `indices[i]`
        int old_region_id = indices[i];
        int new_region_id = i;

        this->region_sizes[new_region_id] = dummy_plan.region_sizes[old_region_id];
        this->region_pops[new_region_id] = dummy_plan.region_pops[old_region_id];
        // Since the regions are in order of added reset the order added to just be 1,..., n
        this->region_added_order[i] = i + 1;
    }

    // reset the max region counter
    this->region_order_max = this->num_regions + 6;
}

// Returns the number of districts and multidistricts in the plan
std::pair<int, int> Plan::get_num_district_and_multidistricts() const {
    int num_districts = 0;
    int num_multidistricts = 0;
    for (size_t region_id = 0; region_id < num_regions; region_id++) {
        if (region_sizes[region_id] == 1)
            num_districts++;
        if (region_sizes[region_id] > 1)
            num_multidistricts++;
    }

    return std::make_pair(num_districts, num_multidistricts);
}

std::string Plan::debug_string(bool print_region_ids) const {
    auto region_counts = get_num_district_and_multidistricts();
    int num_districts = region_counts.first;
    int num_multidistricts = region_counts.second;
    std::ostringstream oss;

    oss << "Plan with "
        << num_regions << " regions, "
        << num_districts << " districts, "
        << num_multidistricts << " multidistricts and "
        << num_districts << " total seats and "
        << region_ids.size() << " Vertices.\n";

    oss << "Region Level Values:[";
    for (int region_id = 0; region_id < num_regions; ++region_id) {
        oss << "(Region " << region_id
            << ", Size=" << static_cast<int>(region_sizes[region_id])
            << ", pop= " << static_cast<int>(region_pops[region_id])
            << "), ";
    }
    oss << "]\n";

    if (print_region_ids) {
        oss << "Plan IDs: c(";
        for (std::size_t i = 0; i < region_ids.size(); ++i) {
            oss << static_cast<int>(region_ids[i]);
            if (i + 1 < region_ids.size()) {
                oss << ", ";
            }
        }

        oss << ")\nPlan Sizes: c(";
        for (int i = 0; i < num_regions; ++i) {
            oss << static_cast<int>(region_sizes[i]);
            if (i + 1 < num_regions) {
                oss << ", ";
            }
        }
        oss << ")\n";
    }

    return oss.str();
}

// Prints our object using Rcout. Should be used in Rcpp call
void Plan::Rprint(bool verbose) const {
    auto print_str = debug_string(verbose);
    Rcpp::Rcerr << print_str;
    return;

    auto region_counts = get_num_district_and_multidistricts();
    int num_districts = region_counts.first;
    int num_multidistricts = region_counts.second;
    Rcpp::Rcout << "Plan with " << num_regions << " regions, " << num_districts
                << " districts, " << num_multidistricts << " multidistricts and "
                << std::accumulate(region_sizes.begin(), region_sizes.end(), 0)
                << " total seats and " << region_ids.size() << " Vertices.\n";

    Rcpp::Rcout << "Region Level Values:[";
    for (int region_id = 0; region_id < num_regions; region_id++) {
        Rcpp::Rcout << "(Region " << region_id
                    << ", Size=" << static_cast<int>(region_sizes[region_id])
                    << ", pop= " << region_pops[region_id] << "), ";
    }
    Rcpp::Rcout << "]\n";

    if (verbose) {
        REprintf("Plan IDs: c(");
        for (size_t i = 0; i < region_ids.size(); ++i) {
            REprintf("%u", region_ids[i]);
            if (i + 1 != region_ids.size())
                REprintf(", ");
        }
        REprintf(")\n Plan Sizes: c(");
        for (size_t i = 0; i < num_regions; ++i) {
            REprintf("%u", region_sizes[i]);
            if (i + 1 != num_regions)
                REprintf(", ");
        }
        REprintf(")\n");
    }
}

// this shallow copies two plans of the same size
void Plan::shallow_copy(Plan const &plan_to_copy) {
    // copy the scalars
    num_regions = plan_to_copy.num_regions;
    region_order_max = plan_to_copy.region_order_max;
    // copy plan vector
    region_ids.copy(plan_to_copy.region_ids);
    // copy the region sizes vector
    region_sizes.copy(plan_to_copy.region_sizes);
    // copy population
    region_pops.copy(plan_to_copy.region_pops);
    // copy order added tracker
    region_added_order.copy(plan_to_copy.region_added_order);
    // It should be 
    if (!forest_edges.empty()){
        forest_edges.copy(plan_to_copy.forest_edges);
    }

    // we copy linking edges as if there's none then 
    // its an empty vector and thats ok
    linking_edges = plan_to_copy.linking_edges;
    return;
}


// Selects a valid multidistrict to split uniformly at random
//
// Given a plan object with at least one multidistrict this function randomly
// selects a valid multidistrict to split with uniform probability (ie it is
// one over the number of valid multidistricts to split in the plan.) and
// returns the log of the probability that region was chosen.
//
//
// @param region_to_split an integer that will be updated by reference with the
// id number of the region selected to split
// @param splitting_schedule The splitting schedule deciding which regions can be split
// @param schedule_ctx The schedule context built for this plan
//
// @details `region_id_to_split` is set to the id of the region selected
//
// @return the region id that was chosen to be split
//
int Plan::choose_multidistrict_to_split(SplittingSchedule const &splitting_schedule,
                                        ScheduleContext const &schedule_ctx,
                                        RNGState &rng_state,
                                        double const selection_alpha) const {

    // make vectors with cumulative d value and region label for later
    std::vector<int> valid_region_ids, associated_region_sizes;



    for (int region_id = 0; region_id < num_regions; region_id++) {
        auto region_size = region_sizes[region_id];
        if constexpr(perf_config::supposedly_safe_input_checks){
            if (region_size < 0 || region_size > splitting_schedule.total_seats) {
                std::ostringstream oss;
                oss << "Region size out of bounds in choose_multidistrict_to_split.\n";
                oss << "region_id=" << region_id << "\n";
                oss << "region_size=" << region_size << "\n";
                oss << "total_seats=" << splitting_schedule.total_seats << "\n";
                oss << debug_string(true);
                throw std::runtime_error(oss.str());
            }
        }
        // if valid then add id to vector
        if (splitting_schedule.region_is_splittable(*this, schedule_ctx, region_id)) {
            // add the count and label to vector
            valid_region_ids.push_back(region_id);
            associated_region_sizes.push_back(region_size);
        }
    }
    auto num_candidates = valid_region_ids.size();

    // If one just return that
    if (num_candidates == 1){
        return valid_region_ids[0];
    }else if (num_candidates == 0) {
        std::ostringstream oss;
        oss << "Plan::choose_multidistrict_to_split found no valid regions to split.\n";
        oss << "num_regions=" << num_regions << "\n";

        oss << "Region sizes: ";
        for (int region_id = 0; region_id < num_regions; ++region_id) {
            oss << static_cast<int>(region_sizes[region_id]) << " ";
        }
        oss << "\n";

        oss << debug_string(true);

        throw std::runtime_error(oss.str());
    }
        

    std::vector<double> region_wgts(valid_region_ids.size());

    for (size_t i = 0; i < valid_region_ids.size(); i++) {
        region_wgts[i] = std::pow(associated_region_sizes[i], selection_alpha);
    }
    int idx = rng_state.r_int_unnormalized_wgt(region_wgts);
    int region_id_to_split = valid_region_ids.at(idx);

    return region_id_to_split;
}

// Attempts to draw a spanning tree on a region using Wilson's algorithm
//
// Attempts to draw a spanning tree on a specific region of a plan using
// Wilson's algorithm. The function will try `attempts_to_make` times to
// draw a tree and if all those fail then it will return false. If its
// successful true will be returned and the tree will be copied into the spanning forest.
//
// @title Attempt to Find a Valid Spanning Tree Edge to Cut into Two New Regions
//
// @param map_params Map parameters (adj graph, population, etc.)
// @param ust_sampler Uniform tree sampler object
// @param region_to_draw_tree_on The id of the region to draw the tree on
// @param rng_state thread safe random number generation object
// @param attempts_to_make How many attempts at calling wilson's algorithm to make
// before giving up.
//
// @details Modifications
//    - `ust_sampler` is modified in place to store the new tree is sampling is
//         successful
//
// @return `true` if tree was successfully draw, `false` otherwise.
//
// @keyword internal
// @noRd
std::pair<bool, int>
Plan::draw_tree_on_region(USTSampler &ust_sampler, const int region_to_draw_tree_on,
                          RNGState &rng_state, int const attempts_to_make) {

    int num_attempts = 0;
    bool tree_drawn = false;
    USTDrawResult ust_draw_result;
    for (size_t attempt_num = 0; attempt_num < attempts_to_make; attempt_num++) {
        ++num_attempts;
        // attempt to draw a tree
        ust_draw_result = ust_sampler.attempt_to_draw_tree_on_region(rng_state, *this, region_to_draw_tree_on);
        tree_drawn = ust_draw_result.successful;
        // if successful return
        if (tree_drawn)
            break;
    }

    // return immediately if false
    if (!tree_drawn)
        return std::make_pair(false, num_attempts);

    // update root region id
    // and its forest vertices
    // clear the tree in the packed forest  
    forest_edges.clear_region_tree(
        region_ids, region_to_draw_tree_on,
        ust_sampler.map_params.graph_edge_index
    );

    // make a queue of vertex, parent
    std::queue<std::pair<int, int>> vertex_queue;
    // add roots children to queue
    // for (auto const &child_vertex : ust_sampler.ust.neighbors(ust_draw_result.root)) {
    for (auto const &child_vertex : ust_sampler.ust[ust_draw_result.root]) {
        vertex_queue.push({child_vertex, ust_draw_result.root});
        // Add this edge to the packed forest 
        forest_edges.set_edge(ust_draw_result.root, child_vertex, ust_sampler.map_params.graph_edge_index);
    }

    // update all the children
    while (!vertex_queue.empty()) {
        // get and remove head of queue
        auto queue_pair = vertex_queue.front();
        int vertex = queue_pair.first;
        int parent_vertex = queue_pair.second;
        vertex_queue.pop();
        // Add this edge to the packed forest 
        forest_edges.set_edge(vertex, parent_vertex, ust_sampler.map_params.graph_edge_index);

        // for (auto const &child_vertex : ust_sampler.ust.neighbors(vertex)) {
        for (auto const &child_vertex : ust_sampler.ust[vertex]) {
            // add children to queue
            vertex_queue.push({child_vertex, vertex});
            // add this edge from vertex to its children
            forest_edges.set_edge(vertex, child_vertex, ust_sampler.map_params.graph_edge_index);
        }
    }

    // print_tree(ust);
    // // forest_edges.print(map_params.graph_edge_index);

    return std::make_pair(true, num_attempts);
}

// Updates the region level info
// Creates new regions and updates the `Plan` object using a cut tree
//
// Takes a cut spanning tree `ust` and variables on the two new regions
// induced by the cuts and creates space/updates the information on those
// two new regions in the `plan` object. This function increases the number
// of regions aspect by 1 and updates the region level information and all
// other variables changed by adding a new region.
//
// It also sets `plan.remainder_region` equal to `new_region2_id` if
// split_district_only is true.
//
//
// @title Create and update new plan regions from cut tree
//
// @param ust A cut (ie has two partition pieces) directed spanning tree
// passed by reference
// @param plan A plan object
// @param split_district_only Whether or not this was split according to a
// one district split scheme (as in does the remainder need to be updated)
// @param old_split_region_id The id of the region that was split into the two
// new ones. Region1 will be set to this id
// @param new_region_id The id that region2 will be set
// @param new_region1_tree_root The vertex of the root of one piece of the cut
// tree. This always corresponds to the region with the smaller dval (allowing
// for the possiblity the dvals are equal).
// @param new_region1_dval The dval associated with the new region 1
// @param new_region1_pop The population associated with the new region 1
// @param new_region2_tree_root The vertex of the root of other piece of the cut
// tree. This always corresponds to the region with the bigger dval (allowing
// for the possiblity the dvals are equal).
// @param new_region2_dval The dval associated with the new region 2
// @param new_region2_pop The population associated with the new region 2
// @param new_region1_id The id the new region 1 was assigned in the plan
// @param new_region2_id The id the new region 2 was assigned in the plan
//
// @details Modifications
//    - `plan` is updated in place with the two new regions
//    - `new_region1_id` is set to the id new region1 was assigned
//    which is just the `old_split_region_id`
//    - `new_region2_id` is set to the id new region2 was assigned
//    which is just `plan.num_regions-1`
//
void Plan::update_region_info_from_cut(EdgeCut cut_edge, const int split_region1_id,
                                       const int split_region2_id, bool const add_region) {
    // Get information on the two new regions cut
    int split_region1_tree_root, split_region2_tree_root;
    int split_region1_size, split_region2_size;
    int split_region1_pop, split_region2_pop;

    cut_edge.get_split_regions_info(split_region1_tree_root, split_region1_size,
                                    split_region1_pop, split_region2_tree_root,
                                    split_region2_size, split_region2_pop);

    // update plan with new regions if needed
    if (add_region)
        num_regions++; // increase region count by 1

    // make the first new region get max plus one
    region_order_max++;
    int new_region1_order_added_num = region_order_max;
    // Now make the second new region max plus one again
    region_order_max++;
    int new_region2_order_added_num = region_order_max;

    // Now update the region level information
    // updates the new region 1
    region_sizes[split_region1_id] = split_region1_size;
    region_added_order[split_region1_id] = new_region1_order_added_num;
    region_pops[split_region1_id] = split_region1_pop;

    // updates the new region 2
    // New region 2's id is the highest id number so push back
    region_sizes[split_region2_id] = split_region2_size;
    region_added_order[split_region2_id] = new_region2_order_added_num;
    region_pops[split_region2_id] = split_region2_pop;
}

void Plan::update_plan_ids_and_forest_from_cut(TreeSplitter const &tree_splitter, 
        USTSampler &ust_sampler, EdgeCut const cut_edge,
        const int split_region1_id, const int split_region2_id, bool const add_region){
    // Get the root of the tree associated with region 1 and 2
    int split_region1_tree_root, split_region2_tree_root;
    int split_region1_size, split_region2_size;
    int split_region1_pop, split_region2_pop;

    cut_edge.get_split_regions_info(split_region1_tree_root, split_region1_size,
                                    split_region1_pop, split_region2_tree_root,
                                    split_region2_size, split_region2_pop);


    if constexpr(perf_config::object_integrity_checking){
        // Checks the tree starting at `split_region1_tree_root` is actually a directed tree
        ust_sampler.check_tree_integrity(
            ust_sampler.get_vertex_tree(),
            "Before calling `assign_region_id_and_forest_from_tree` on split_region1_id\n",
            split_region1_tree_root,
            0,
            false
        );
    }
    // update the vertex labels and the tree for region 1
    assign_region_id_and_forest_from_tree(ust_sampler.ust, region_ids, 
                                          forest_edges,
                                          split_region1_tree_root, split_region1_id,
                                          ust_sampler.map_params.graph_edge_index,
                                          ust_sampler.vertex_queue);

    if constexpr(perf_config::object_integrity_checking){
        ust_sampler.check_tree_integrity(
            ust_sampler.get_vertex_tree(),
            "Before calling `assign_region_id_and_forest_from_tree` on split_region2_id\n",
            split_region2_tree_root,
            0,
            false
        );
    }
    assign_region_id_and_forest_from_tree(ust_sampler.ust, region_ids, 
                                          forest_edges,
                                          split_region2_tree_root, split_region2_id,
                                          ust_sampler.map_params.graph_edge_index,
                                          ust_sampler.vertex_queue);

}


void Plan::update_from_successful_split(TreeSplitter const &tree_splitter,
                                        USTSampler &ust_sampler, EdgeCut const &cut_edge,
                                        int const new_region1_id, int const new_region2_id,
                                        bool const add_region) {
    // now update the region level information from the edge cut
    update_region_info_from_cut(cut_edge, new_region1_id, new_region2_id, add_region);

    // Now update the vertex level information
    update_vertex_and_plan_specific_info_from_cut(tree_splitter, ust_sampler, cut_edge,
                                                  new_region1_id, new_region2_id, add_region);
}

void Plan::fill_in_skipped_subtrees(
    USTSampler &ust_sampler,
    RNGState &rng_state,
    int const max_tries_multiple
) {
    // only do if we have forest edges
    if (!forest_edges.empty()){
        ust_sampler.fill_in_skipped_subtrees(
            forest_edges,
            rng_state,
            max_tries_multiple
        );
    }
    return;
}

// Attempts Gets valid pairs to merge for MCMC
// Returns false if multigraph was not successfully built
std::pair<bool, std::vector<std::pair<RegionID, RegionID>>>
Plan::attempt_to_get_valid_mergesplit_pairs(PlanMultigraph &plan_multigraph,
                                            SplittingSchedule const &splitting_schedule, ScheduleContext &schedule_ctx,
                                            ScoringFunction const &scoring_function) const {
    bool const result = plan_multigraph.build_plan_multigraph(region_ids, num_regions);
    // return false if not successful
    if (!result)
        return std::make_pair(false, std::vector<std::pair<RegionID, RegionID>>{});
    // plan_multigraph.Rprint_detailed(*this);
    // else remove all the invalid size and mergesplit pairs and invalid hard constraint
    plan_multigraph.remove_invalid_size_pairs(*this, splitting_schedule, schedule_ctx);
    plan_multigraph.remove_invalid_hard_constraint_pairs(*this, scoring_function);
    plan_multigraph.remove_invalid_mergesplit_pairs(*this);
    // plan_multigraph.Rprint_detailed(*this);

    return std::make_pair(true, plan_multigraph.pair_map.hashed_pairs);
}

std::vector<std::pair<RegionID, RegionID>> Plan::get_valid_smc_merge_regions(
    PlanMultigraph &plan_multigraph, SplittingSchedule const &splitting_schedule, ScheduleContext &schedule_ctx,
    ScoringFunction const &scoring_function) const {
    // build the multigraph
    plan_multigraph.build_plan_multigraph(region_ids, num_regions);


    std::vector<std::pair<RegionID, RegionID>> output_pairs;
    output_pairs.reserve(plan_multigraph.pair_map.num_hier_smc_merge_valid_pairs);

    // go through and only add pairs the schedule says are ok to merge
    for (auto const a_pair : plan_multigraph.pair_map.hashed_pairs) {
        if (!splitting_schedule.is_valid_merge_pair(*this, schedule_ctx, a_pair.first,
                                                    a_pair.second))
            continue;

        // now check hierarchical merge is ok
        auto result = plan_multigraph.pair_map.get_value(a_pair.first, a_pair.second);
        if (!result.second.merge_is_hier_valid)
            continue;

        // now check hard constraints are satisfied
        bool const hard_constr_result =
            scoring_function.merged_plan_ok(*this, a_pair.first, a_pair.second);

        if (!hard_constr_result)
            continue;

        // we've passed all checks so now we can add
        output_pairs.emplace_back(a_pair);
    }

    return output_pairs;
}

// Plan debugging functions 
void Plan::check_forest_equality(
    Tree const &ust1,
    Tree const &ust2,
    GraphEdgeIndex const &graph_edge_index,
    std::string_view msg
) const {
    check_forest_integrity(graph_edge_index, msg);

    std::ostringstream oss;
    bool failed = false;

    if (ust1.size() != ust2.size()) {
        failed = true;

        oss << msg << "\n";
        oss << "Tree 1 has size " << ust1.size()
            << " and Tree 2 has size " << ust2.size() << "!\n";

        oss << "Printing Tree 1\n";
        oss << tree_to_string(ust1);

        oss << "Printing Tree 2\n";
        oss << tree_to_string(ust2);

        oss << debug_string(true);

        throw std::runtime_error(oss.str());
    }

    for (std::size_t v = 0; v < ust1.size(); ++v) {
        if (ust1[v].size() != ust2[v].size()) {
            failed = true;

            oss << msg << "\n";
            oss << "Tree 1 has size " << ust1[v].size()
                << " and Tree 2 has size " << ust2[v].size()
                << " at vertex " << v << "!\n";

            oss << "Printing Tree 1\n";
            oss << tree_to_string(ust1);

            oss << "Printing Tree 2\n";
            oss << tree_to_string(ust2);

            oss << debug_string(true);

            throw std::runtime_error(oss.str());
        }

        std::unordered_multiset<int> s1(ust1[v].begin(), ust1[v].end());
        std::unordered_multiset<int> s2(ust2[v].begin(), ust2[v].end());

        if (s1 != s2) {
            failed = true;

            std::vector<int> a(ust1[v].begin(), ust1[v].end());
            std::vector<int> b(ust2[v].begin(), ust2[v].end());

            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());

            oss << msg << "\n";
            oss << "Vertex " << v << " does not have the same neighbors!\n";

            oss << "forest_graph[" << v << "]: ";
            for (int const u : a) {
                oss << u << " ";
            }
            oss << "\n";

            oss << "packed_forest[" << v << "]: ";
            for (int const u : b) {
                oss << u << " ";
            }
            oss << "\n";

            oss << "In forest_graph but not packed_forest: ";
            std::size_t i = 0;
            std::size_t j = 0;

            while (i < a.size()) {
                if (j >= b.size() || a[i] < b[j]) {
                    oss << "(" << v << ", " << a[i] << ") ";
                    ++i;
                } else if (b[j] < a[i]) {
                    ++j;
                } else {
                    ++i;
                    ++j;
                }
            }
            oss << "\n";

            oss << "In packed_forest but not forest_graph: ";
            i = 0;
            j = 0;

            while (j < b.size()) {
                if (i >= a.size() || b[j] < a[i]) {
                    oss << "(" << v << ", " << b[j] << ") ";
                    ++j;
                } else if (a[i] < b[j]) {
                    ++i;
                } else {
                    ++i;
                    ++j;
                }
            }
            oss << "\n";

            oss << "Printing Tree 1\n";
            oss << tree_to_string(ust1);

            oss << "Printing Tree 2\n";
            oss << tree_to_string(ust2);

            oss << debug_string(true);

            throw std::runtime_error(oss.str());
        }
    }
}

// checks the forests have no edges crossing regions
void Plan::check_forest_integrity(
    GraphEdgeIndex const &graph_edge_index,
    std::string_view msg
) const {
    if (forest_edges.empty()) return;
    std::ostringstream oss;
    bool failed = false;

    int graph_forest_cross_region_edge_count = 0;
    int packed_forest_cross_region_edge_count = 0;

    for (int v = 0; v < graph_edge_index.V; ++v) {
        int const v_region = static_cast<int>(region_ids[v]);

        if (v_region < 0 || v_region >= num_regions) {
            failed = true;
            oss << "Invalid region id at vertex " << v
                << ": region_ids[" << v << "]=" << v_region
                << ", num_regions=" << num_regions << "\n";
            continue;
        }

        for (auto const &incident_edge : graph_edge_index.incident_edges[v]) {
            int const u = static_cast<int>(incident_edge.neighbor);

            if (u < 0 || u >= graph_edge_index.V) {
                failed = true;
                oss << "GRAPH EDGE INDEX: Invalid incident neighbor "
                    << u << " found for vertex " << v << "\n";
                continue;
            }

            if (v > u) {
                continue;
            }

            int const u_region = static_cast<int>(region_ids[u]);

            if (v_region != u_region &&
                forest_edges.test_edge_id(incident_edge.edge_id)) {
                failed = true;
                ++packed_forest_cross_region_edge_count;

                oss << "PACKED FOREST CROSS-REGION EDGE: pair ("
                    << v << ", " << u << "), v-region="
                    << v_region << ", u-region=" << u_region
                    << ", edge_id="
                    << static_cast<unsigned int>(incident_edge.edge_id)
                    << "\n";
            }
        }
    }

    if (failed) {
        oss << msg << "\n";
        oss << "Forest integrity check failed after scanning whole graph.\n";
        oss << "Graph forest cross-region edge count: "
            << graph_forest_cross_region_edge_count << "\n";
        oss << "Packed forest cross-region edge count: "
            << packed_forest_cross_region_edge_count << "\n";
        oss << debug_string(true);

        throw std::runtime_error(oss.str());
    }
}

bool Plan::forest_graph_equals_order_insensitive(
    Tree const &other,
    GraphEdgeIndex const &graph_edge_index, 
    std::string &out
) const {
    std::ostringstream oss;

    auto forest_graph = forest_edges.get_graph_tree(graph_edge_index);

    if (forest_graph.size() != other.size()) {
        oss << "forest_graph sizes differ: "
            << forest_graph.size() << " vs " << other.size() << "\n";

        out = oss.str();
        return false;
    }

    std::vector<int> a;
    std::vector<int> b;

    for (std::size_t v = 0; v < forest_graph.size(); ++v) {
        a.assign(forest_graph[v].begin(), forest_graph[v].end());
        b.assign(other[v].begin(), other[v].end());

        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());

        if (a != b) {
            oss << "Vertex " << v << " does not have the same neighbors!\n";

            oss << "forest_graph[" << v << "]: ";
            for (int const u : a) {
                oss << u << " ";
            }
            oss << "\n";

            oss << "other[" << v << "]: ";
            for (int const u : b) {
                oss << u << " ";
            }
            oss << "\n";

            oss << "In forest_graph but not other: ";
            std::size_t i = 0;
            std::size_t j = 0;

            while (i < a.size()) {
                if (j >= b.size() || a[i] < b[j]) {
                    oss << "(" << v << ", " << a[i] << ") ";
                    ++i;
                } else if (b[j] < a[i]) {
                    ++j;
                } else {
                    ++i;
                    ++j;
                }
            }
            oss << "\n";

            oss << "In other but not forest_graph: ";
            i = 0;
            j = 0;

            while (j < b.size()) {
                if (i >= a.size() || b[j] < a[i]) {
                    oss << "(" << v << ", " << b[j] << ") ";
                    ++j;
                } else if (a[i] < b[j]) {
                    ++i;
                } else {
                    ++i;
                    ++j;
                }
            }
            oss << "\n";
        }
    }

    out = oss.str();
    return out.empty();
}



std::string component_counts_to_string(
    std::vector<int> const &component_region_counts,
    int const num_components
) {
    std::ostringstream oss;

    oss << "Component region counts:\n";
    for (int i = 0; i < num_components; ++i) {
        oss << "  Component " << i
            << " - " << component_region_counts[i]
            << " Regions\n";
    }

    return oss.str();
}

std::string sorted_pairs_to_string(
    std::vector<std::pair<std::pair<RegionID, RegionID>, PairHashData>> const &all_pairs,
    std::vector<int> const &county_component,
    std::string_view title
) {
    std::ostringstream oss;

    oss << title << "\n";
    for (auto const &val : all_pairs) {
        int const r1 = static_cast<int>(val.first.first);
        int const r2 = static_cast<int>(val.first.second);

        oss << "Regions (" << r1 << ", " << r2 << ")";

        if (r1 >= 0 && r1 < static_cast<int>(county_component.size()) &&
            r2 >= 0 && r2 < static_cast<int>(county_component.size())) {
            oss << " | Components ("
                << county_component[r1] << ", "
                << county_component[r2] << ")";
        } else {
            oss << " | Components (invalid region id)";
        }

        oss << " | Shared Status "
            << (val.second.same_admin_component ? "SHARED" : "NOT SHARED")
            << " | Admin Adjacent "
            << (val.second.admin_adjacent ? "YES" : "NOT")
            << " | Hier Merge Valid "
            << (val.second.merge_is_hier_valid ? "YES" : "NOT")
            << " | within_county_edges=" << val.second.within_county_edges
            << " | across_county_edges=" << val.second.across_county_edges
            << "\n";
    }

    return oss.str();
}



void RegionPairHash::count_graph_edge(RegionID region1_id, RegionID region2_id, 
    int vertex1,
    int vertex2,
    bool const same_county,
    int const county) {
    // swap if region2_id < region1_id
    // so its always true that 
    // region1_id < region2_id
    // vertex1 belongs to region1_id
    // vertex2 belongs to region2_id
    if (region2_id < region1_id){
        std::swap(region1_id, region2_id);
        std::swap(vertex1, vertex2);
    }
        
    // get the index
    auto hash_index = pair_hash(region1_id, region2_id);
    /*
    * If this is the first map edge encountered between these regions,
    * create the pair and save this edge as the initial representative.
    */
    if (!hashed[hash_index]) {
        hashed[hash_index] = true;
        hashed_pairs.push_back({region1_id, region2_id});
        ++num_hashed_pairs;
        values[hash_index].representative_vertex1 = vertex1;
        values[hash_index].representative_vertex2 = vertex2;
    }

    if (same_county) {
        /*
        * IMPORTANT FOR HIERARCHICAL SAMPLING:
        *
        * If this is the FIRST within-county edge encountered for this
        * pair, replace the existing representative edge with this one.
        *
        * Why?
        *
        * If a pair has at least one within-county edge, admin_adjacent
        * becomes true, and ForestPlan later ignores across-county edges
        * for this pair. Therefore our representative edge must also be
        * one of the countable within-county edges.
        *
        * admin_adjacent is still false exactly when this is the first
        * within-county edge encountered.
        */
        if (!values[hash_index].admin_adjacent) {
            values[hash_index].representative_vertex1 = vertex1;
            values[hash_index].representative_vertex2 = vertex2;
        }
        // if edges in the same county increase within county count
        // and mark as administratively adjacent
        values[hash_index].admin_adjacent = true;
        values[hash_index].same_admin_component = true;
        values[hash_index].shared_county = county;
        values[hash_index].within_county_edges++;
    } else {
        // else increase count of across county stuff
        values[hash_index].across_county_edges++;
    }
    return;
}


// Creates helpful debug string 
std::string RegionPairHash::debug_string(
    std::vector<int> const &county_component
) const {
    std::ostringstream oss;

    oss << "Pair Map has " << num_hashed_pairs << " Elements:\n";

    for (auto const &a_pair : hashed_pairs) {
        auto val = get_value(a_pair.first, a_pair.second).second;

        int const r1 = static_cast<int>(a_pair.first);
        int const r2 = static_cast<int>(a_pair.second);

        oss << "    Regions: (" << r1 << ", " << r2 << ")";

        if (r1 >= 0 && r1 < static_cast<int>(county_component.size()) &&
            r2 >= 0 && r2 < static_cast<int>(county_component.size())) {
            oss << " | Components ("
                << county_component[r1] << ", "
                << county_component[r2] << ") "
                << (county_component[r1] == county_component[r2]
                        ? "SAME(check now)"
                        : "NOT SAME(check now)");
        } else {
            oss << " | Components invalid";
        }

        oss << " | "
            << (val.same_admin_component ? "SAME(attr)" : "NOT SAME(attr)")
            << " | "
            << (val.merge_is_hier_valid ? "YES" : "NOT")
            << " Hier Merge Valid"
            << " | "
            << (val.admin_adjacent ? "YES" : "NOT")
            << " Admin Adjacent"
            << " | " << val.within_county_edges << " within county edges"
            << " | " << val.across_county_edges << " across county edges"
            << "\n";
        oss << " | representative_edge=("
            << val.representative_vertex1
            << ", "
            << val.representative_vertex2
            << ")";
    }

    return oss.str();
}


// Prints relevant info - for debugging
void RegionPairHash::Rprint(std::vector<int> const &county_component) const {
    Rcpp::Rcerr << debug_string(county_component);
};

double PlanMultigraph::compute_non_hierarchical_log_multigraph_tau(
    int const num_regions, ScoringFunction const &scoring_function) {
    // if two regions then its just the log(boundary length) between
    // Since any minor of laplacian is just the degree of vertex 0 or 1
    if (num_regions == 2) {
        // Get the value
        auto val = pair_map.get_value(0, 1).second;
        if (val.admin_adjacent) {
            return std::log(static_cast<double>(val.within_county_edges));
        } else {
            return std::log(static_cast<double>(val.across_county_edges));
        }
    }

    // else go through and build the laplacian for all `num_regions` regions
    // so 0 ... `num_regions-1`

    // reset the laplacian where the number of vertices
    // is num_regions and the dropped vertex is the largest 
    // region id so num_regions-1
    laplacian.reset(num_regions, num_regions-1);

    // Now we iterate through the pairs
    for (auto const a_pair : pair_map.hashed_pairs) {
        // Get the value
        auto val = pair_map.get_value(a_pair.first, a_pair.second).second;
        // Since we're assuming non hierarchical there only one
        // component so all edges are within county edges

        int const edges = val.within_county_edges;
        if constexpr (perf_config::supposedly_safe_input_checks) {
        // sanity check its not a loop
        if (a_pair.second == a_pair.first) {
            std::ostringstream oss;
            oss << "ERROR in compute_non_hierarchical_log_multigraph_tau: "
                << "unexpected equal region indices while constructing Laplacian minor.\n";
            oss << "a_pair=("
                << static_cast<int>(a_pair.first) << ", "
                << static_cast<int>(a_pair.second) << ")\n";
            oss << "num_regions=" << num_regions << "\n";
            oss << "edges=" << edges << "\n";
            oss << debug_string();

            throw std::runtime_error(oss.str());
            }
        }

        // Now for both pairs we need to
        // 1. Update the degree of the vertex
        // 2. Increase the count of edges between the regions
        // The laplacian builder handles that and skips adding the degree
        // and edge count for the dropped vertex `num_regions-1`
        laplacian.add_edge(a_pair.first, a_pair.second, edges);
    }

    return laplacian.log_det();
};

double PlanMultigraph::compute_non_hierarchical_merged_log_multigraph_tau(
    int const num_regions, RegionID const region1_id, RegionID const region2_id,
    ScoringFunction const &scoring_function) {
    // If 2 regions then merged is 1 region and there's exactly 1 way to draw
    // linking edges (since its empty set) so log(1) = 0
    if (num_regions == 2)
        return 0.0;

    /*
    we make merged reindex which essentially just reindexes all the regions to
    the two merged regions are the new largest value and everything else gets
    potentially shifted down. For example imagine 5 regions
    and we merge 1 and 3 so the reindex vector is [0,3,1,3,2]
     */
    int const merged_reindex = num_regions - 2;
    for (int current_reindex = 0, i = 0; i < num_regions; i++) {
        if (i == region1_id || i == region2_id) {
            region_reindex_vec[i] = merged_reindex;
        } else {
            region_reindex_vec[i] = current_reindex;
            ++current_reindex;
        }
    }
    int const num_merged_regions = num_regions - 1;

    /*
    Recall the graph laplacian for a multigraph with V vertices is a VxV matric
    where
        - v_ii is the degree of the vertex (ie the total number of edges where
        v_ii is a vertex)
        - v_ij is -m_ij where m_ij is the number of edges between vertex i and j

    Kirchenoff's theorem says the determinant of any minor of laplcacian is number
    of spanning trees. A minor is laplacian with any row and column deleted.

    So for the merged plan it has num_regions-1 vertices and since we only compute
    determinant of the minor we actually have a num_regions-2 x num_regions-2 matrix
    where we delete the row and column corresponding to the merged region.
     */

    // reset the laplacian where the number of vertices
    // is num_merged_regions and the dropped vertex is the merged_reindex
    laplacian.reset(num_merged_regions, merged_reindex);

    // Now we iterate through the pairs
    for (auto const a_pair : pair_map.hashed_pairs) {
        // Get the value
        auto val = pair_map.get_value(a_pair.first, a_pair.second).second;

        auto reshuffled_pair1 = region_reindex_vec[a_pair.first];
        auto reshuffled_pair2 = region_reindex_vec[a_pair.second];

        // If its (region1, region2) then skip because merged
        if (reshuffled_pair1 == reshuffled_pair2)
            continue;

        // since non hierarchcial every merge pair is valid and all edges are
        // within county edges

        int edges = val.within_county_edges;
        // Now for both pairs we need to
        // 1. Update the degree of the vertex
        // 2. Increase the count of edges between the regions
        // The laplacian builder handles that and skips adding the degree
        // and edge count for the dropped vertex `merged_reindex`
        laplacian.add_edge(reshuffled_pair1, reshuffled_pair2, edges);
    }

    return laplacian.log_det();
}

double PlanMultigraph::compute_log_multigraph_tau(int const num_regions,
                                                  ScoringFunction const &scoring_function) {
    if (counties_on) {
        return compute_hierarchical_log_multigraph_tau(num_regions, scoring_function);
    } else {
        return compute_non_hierarchical_log_multigraph_tau(num_regions, scoring_function);
    }
}

double PlanMultigraph::compute_merged_log_multigraph_tau(
    int const num_regions, RegionID const region1_id, RegionID const region2_id,
    ScoringFunction const &scoring_function) {
    if (counties_on) {
        return compute_hierarchical_merged_log_multigraph_tau(
            num_regions, region1_id, region2_id, scoring_function);
    } else {
        return compute_non_hierarchical_merged_log_multigraph_tau(num_regions, region1_id,
                                                                  region2_id, scoring_function);
    }
}

double PlanMultigraph::compute_hierarchical_log_multigraph_tau(
    int const num_regions, ScoringFunction const &scoring_function) {
    // if two regions then its just the log(boundary length) between
    // Since any minor of laplacian is just the degree of vertex 0 or 1
    if (num_regions == 2) {
        // Get the value
        auto val = pair_map.get_value(0, 1).second;
        if (val.admin_adjacent) {
            return std::log(static_cast<double>(val.within_county_edges));
        } else {
            return std::log(static_cast<double>(val.across_county_edges));
        }
    }

    // We need to compute this hierarchically. First we compute within each
    // connected component then we compute across all of them
    // First we need to sort all the pairs

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        Rprint();
    }

    auto all_pairs = pair_map.get_all_values(true);

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("%d Components | Pre-Sorted Pairs: \n", num_county_connected_components);
        for (auto const &val : all_pairs) {
            REprintf("Regions (%d, %d) | Components (%d, %d) | Shared Status %s\n",
                     (int)val.first.first, (int)val.first.second,
                     county_component[val.first.first], county_component[val.first.second],
                     (val.second.same_admin_component ? "SHARED" : "NOT SHARED"));
        }
    }

    // Sort the pairs to
    //  1. All pairs that are in different components are at the end
    //  2. Among pairs in the same component sort them by their component
    std::sort(all_pairs.begin(), all_pairs.end(), [&](const auto &a, const auto &b) {
        const auto &[region_1a, region_2a] = a.first;
        const PairHashData &data_a = a.second;

        const auto &[region_1b, region_2b] = b.first;
        const PairHashData &data_b = b.second;

        // Rule 1: same_admin_component == true goes to the front
        if (data_a.same_admin_component != data_b.same_admin_component) {
            // recall true =1, false = 0 so this says if data_a and data_b
            // different then a is greater iff its true
            return data_a.same_admin_component > data_b.same_admin_component;
        }

        // Rule 2: for matching entries, sort by county_component[region1]
        if (data_a.same_admin_component) {
            return county_component[region_1a] < county_component[region_1b];
        }

        return false; // otherwise, keep existing relative order
    });

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("%d Components | NOW SORTED Pairs: \n", num_county_connected_components);
        for (auto const &val : all_pairs) {
            REprintf("Regions (%d, %d) | Components (%d, %d) | Shared Status %s\n",
                     (int)val.first.first, (int)val.first.second,
                     county_component[val.first.first], county_component[val.first.second],
                     (val.second.same_admin_component ? "SHARED" : "NOT SHARED"));
        }
    }

    double log_tau = 0.0;

    // current index through all_pairs
    int curr_index = 0;

    // Now we go through each component and compute log spanning trees
    for (int component_id = 0; component_id < num_county_connected_components; component_id++) {

        // stop loop if we've reached pairs across components
        if (curr_index >= all_pairs.size() ||
            !all_pairs[curr_index].second.same_admin_component)
            break;

        int num_component_regions = component_region_counts[component_id];

        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("Component %d which has %d regions!\n", component_id,
                     num_component_regions);
        }

        // If 1 component then there only 1 tree so log(1) = 0
        if (num_component_regions <= 1) {
            // do nothing
            // REprintf("1 Region in Component! Log tau unchanged at %f\n", log_tau);
            // do nothing
        } else if (num_component_regions == 2) {
            // If its two then its just the degree of either vertex which is the boundary length
            // Since its the same component its just within component edges
            log_tau +=
                std::log(static_cast<double>(all_pairs[curr_index].second.within_county_edges));
            // REprintf("2 Regions in Component! Log tau now at %f\n", log_tau);
            // now increment index
            ++curr_index;
        } else {
            if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                REprintf("3 or more regions in component!\n");
            }
            // else more than two regions so we need to reindex the regions
            // This ensures each region in the component is indexed from
            // 0, ... num regions in the component
            for (int current_reindex = 0, a_region_id = 0; a_region_id < num_regions;
                 a_region_id++) {
                // if in the component then reindex that region
                if (county_component[a_region_id] == component_id) {
                    region_reindex_vec[a_region_id] = current_reindex;
                    ++current_reindex;
                    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                        REprintf("Mapping %d to %d!\n", a_region_id,
                                 region_reindex_vec[a_region_id]);
                    }
                }
            }


            // reset the laplacian where the number of vertices
            // is num_component_regions and the dropped vertex is the num_component_regions-1
            laplacian.reset(num_component_regions, num_component_regions - 1);

            // now we iterate through all pairs where both are in this component
            while (curr_index < all_pairs.size() &&
                   county_component[all_pairs[curr_index].first.first] == component_id &&
                   county_component[all_pairs[curr_index].first.second] == component_id) {

                const auto &[pair_region1, pair_region2] = all_pairs[curr_index].first;
                const PairHashData &pair_val = all_pairs[curr_index].second;

                // Because we filter out hier merge invalid we don't need to check

                double edges = 0;

                // else valid merge so get edges we count
                if (pair_val.admin_adjacent) {
                    edges =
                        static_cast<double>(all_pairs[curr_index].second.within_county_edges);
                } else {
                    std::ostringstream oss;

                    oss << "Error in compute_hierarchical_log_multigraph_tau.\n";
                    oss << "Region pair is not administratively adjacent but was treated as hier-merge invalid.\n";
                    oss << "num_county_connected_components=" << num_county_connected_components << "\n";
                    oss << "pair_region1=" << pair_region1 << "\n";
                    oss << "pair_region2=" << pair_region2 << "\n";

                    if (pair_region1 >= 0 && pair_region1 < static_cast<int>(county_component.size()) &&
                        pair_region2 >= 0 && pair_region2 < static_cast<int>(county_component.size())) {
                        oss << "county_component[pair_region1]=" << county_component[pair_region1] << "\n";
                        oss << "county_component[pair_region2]=" << county_component[pair_region2] << "\n";
                    }

                    oss << debug_string();

                    throw std::runtime_error(oss.str());
                }

                if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                    REprintf("Region Pair (%d, %d) | %d | %g edges!\n ", pair_region1,
                             pair_region2, pair_val.admin_adjacent, edges);
                }

                auto reindexed_pair_region1 = region_reindex_vec[pair_region1];
                auto reindexed_pair_region2 = region_reindex_vec[pair_region2];

                if constexpr (perf_config::supposedly_safe_input_checks) {
                if (reindexed_pair_region1 == reindexed_pair_region2){
                    std::ostringstream oss;
                    oss << "ERROR in compute_hierarchical_log_multigraph_tau: "
                        << "equal reindexed pair inside component Laplacian.\n";
                    oss << "reindexed_pair_region1=" << reindexed_pair_region1 << "\n";
                    oss << "reindexed_pair_region2=" << reindexed_pair_region2 << "\n";
                    oss << "component_id=" << component_id << "\n";
                    oss << "num_component_regions=" << num_component_regions << "\n";
                    oss << "edges=" << edges << "\n";
                    oss << debug_string();

                    throw std::runtime_error(oss.str());
                }
                }

                // Now for both pairs we need to
                // 1. Update the degree of the vertex
                // 2. Increase the count of edges between the regions
                laplacian.add_edge(reindexed_pair_region1,
                                    reindexed_pair_region2, 
                                    edges);

                // increase current index by 1
                ++curr_index;
            }
            if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                REprintf("Printing Laplacian Minor!\n");
                // laplacian_minor.print();
            }
            // Now add log det
            log_tau += laplacian.log_det();
        }
    }

    if (curr_index >= all_pairs.size() && num_county_connected_components > 1) {
        std::ostringstream oss;

        oss << "ERROR in compute_hierarchical_log_multigraph_tau: "
            << "curr_index ran past all_pairs before processing all admin components.\n";
        oss << "all_pairs.size()=" << all_pairs.size() << "\n";
        oss << "curr_index=" << curr_index << "\n";
        oss << "num_county_connected_components="
            << num_county_connected_components << "\n";

        oss << sorted_pairs_to_string(
            all_pairs,
            county_component,
            "NOW SORTED Pairs:"
        );

        oss << component_counts_to_string(
            component_region_counts,
            num_county_connected_components
        );

        oss << debug_string();

        throw std::runtime_error(oss.str());
    }

    // If only one connected component then we stop
    if (num_county_connected_components == 1) {
        return log_tau;
    } else if (num_county_connected_components == 2) {
        int across_component_edges = 0;
        // If two then we just sum the edges across and take log
        for (int i = curr_index; i < all_pairs.size(); i++) {
            across_component_edges += all_pairs[i].second.across_county_edges;
        }
        // add
        log_tau += std::log(static_cast<double>(across_component_edges));
        return log_tau;
    }

    // now reset laplacian for size `num_county_connected_components`
    // with the ignore vertex being the last one so
    // `num_county_connected_components - 1`
    laplacian.reset(num_county_connected_components, 
        num_county_connected_components - 1);

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("Current index now %d with %zu pairs\n", curr_index, all_pairs.size());
    }

    // Iterate over the remaining pairs that are across components
    for (int i = curr_index; i < all_pairs.size(); i++) {
        const auto &[pair_region1, pair_region2] = all_pairs[i].first;
        const PairHashData &pair_val = all_pairs[i].second;

        int component1_id = county_component[pair_region1];
        int component2_id = county_component[pair_region2];

        // pairs from curr_index on were sorted to be across components
        if (pair_val.same_admin_component) {
            std::ostringstream oss;
            oss << "ERROR in compute_hierarchical_log_multigraph_tau: region pair ("
                << static_cast<int>(pair_region1) << ", " << static_cast<int>(pair_region2)
                << ") at sorted index " << i << " (first across component index "
                << curr_index << ") is marked as in the same administrative component "
                << "but should be across components " << component1_id << " and "
                << component2_id << ".\n"
                << debug_string();
            throw std::runtime_error(oss.str());
        }

        double edges = static_cast<double>(pair_val.across_county_edges);
        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("Components (%d,%d) - %g edges across!\n", component1_id, component2_id,
                     edges);
        }

        if constexpr (perf_config::supposedly_safe_input_checks) {
        if (component1_id == component2_id){
            std::ostringstream oss;
            oss << "ERROR in compute_hierarchical_log_multigraph_tau: "
                << "equal component ids while constructing component Laplacian.\n";
            oss << "component1_id=" << component1_id << "\n";
            oss << "component2_id=" << component2_id << "\n";
            oss << "num_county_connected_components=" << num_county_connected_components << "\n";
            oss << "edges=" << edges << "\n";
            oss << debug_string();

            throw std::runtime_error(oss.str());
        }
        }

        // Now for both pairs we need to
        // 1. Update the degree of the vertex
        // 2. Increase the count of edges between the regions
        laplacian.add_edge(component1_id,
                            component2_id, 
                            edges);
    }

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("Printing Component Laplacian Minor\n");
    }

    // Now add log det
    log_tau += laplacian.log_det();

    return log_tau;
}

// Need to take care because some previously inelgible hier merge pairs now
// become ok to merge
double PlanMultigraph::compute_hierarchical_merged_log_multigraph_tau(
    int const num_regions, RegionID const region1_id, RegionID const region2_id,
    ScoringFunction const &scoring_function) {
    // if two regions then their merge is just 1 region which has no linking
    // edges. Size of empty set is 0 so return log(1) = 0
    if (num_regions <= 2) {
        return 0.0;
    }

    // Determine if the merge is joining two admin connected components
    auto region1_component = county_component[region1_id];
    auto region2_component = county_component[region2_id];
    bool const two_components_merged = region1_component != region2_component;
    int merged_num_admin_connected_components =
        num_county_connected_components - two_components_merged;

    // We need to compute this hierarchically. First we compute within each
    // connected component then we compute across all of them
    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("We are merging %u and %u! in unmereged components( %d, %d) | There are %d "
                 "Unmerged Components and %d merged components | \n",
                 region1_id, region2_id, region1_component, region2_component,
                 num_county_connected_components, merged_num_admin_connected_components);
        Rprint();
    }

    // First get all pairs ignoring invalid merges
    auto all_pairs = pair_map.get_all_values(true);

    // We'll use the reindex vector to first reindex component ids
    // Since there can't be more components than regions
    // this will just map component two to component 1
    for (int a_region_id = 0; a_region_id < num_regions; a_region_id++) {
        auto const this_region_component = county_component[a_region_id];
        // all components but region 2's get mapped to themselves
        if (this_region_component == region2_component) {
            county_component_reindex[a_region_id] = region1_component;
        } else {
            county_component_reindex[a_region_id] = this_region_component;
        }
        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("Region %d maps to component %d now! (was %d) \n", a_region_id,
                     county_component_reindex[a_region_id], county_component[a_region_id]);
        }
    }

    if (two_components_merged) {
        // If we're joining two components then we need to filter out pairs
        // that are not admin adjacent but were previously in different components
        all_pairs.erase(
            std::remove_if(
                all_pairs.begin(), all_pairs.end(),
                [&](const std::pair<std::pair<RegionID, RegionID>, PairHashData> &an_el) {
                    const auto &[region_1a, region_2a] = an_el.first;
                    const PairHashData &data_a = an_el.second;

                    // We remove if both are same component and not admin adjacent
                    bool const invalid_merge_now = (county_component_reindex[region_1a] ==
                                                        county_component_reindex[region_2a] &&
                                                    !data_a.admin_adjacent);

                    // remove if any size is invalid split size or if merged size can't be split
                    return invalid_merge_now;
                }),
            all_pairs.end());
    } else {
        // We're merging two regions in the same admin connected component so just
        // remove that pair
        all_pairs.erase(std::remove_if(all_pairs.begin(), all_pairs.end(),
                                       [&](auto an_el) {
                                           const auto &[region_1a, region_2a] = an_el.first;
                                           return region_1a == region1_id &&
                                                  region_2a == region2_id;
                                       }),
                        all_pairs.end());
    }

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("%d Components | Pre-Sorted Pairs: \n", merged_num_admin_connected_components);
        for (auto const &val : all_pairs) {
            REprintf("Regions (%d, %d) | Components (%d, %d) | Shared Status %s | Edges %d\n",
                     val.first.first, val.first.second,
                     county_component_reindex[val.first.first],
                     county_component_reindex[val.first.second],
                     (val.second.same_admin_component ? "SHARED" : "NOT SHARED"),
                     (val.second.same_admin_component ? val.second.within_county_edges
                                                      : val.second.across_county_edges));
        }
    }

    // Sort the pairs to
    //  1. All pairs that are in different components are at the end
    //  2. Among pairs in the same component sort them by their component
    // need to account for one less component in sorting
    std::sort(all_pairs.begin(), all_pairs.end(), [&](const auto &a, const auto &b) {
        const auto &[region_1a, region_2a] = a.first;

        bool const data_a_same_component =
            (county_component_reindex[region_1a] == county_component_reindex[region_2a]);

        const auto &[region_1b, region_2b] = b.first;

        bool const data_b_same_component =
            (county_component_reindex[region_1b] == county_component_reindex[region_2b]);

        // Rule 1: same_admin_component == true goes to the front
        if (data_a_same_component != data_b_same_component) {
            // recall true =1, false = 0 so this says if data_a and data_b
            // different then a is greater iff its true
            return data_a_same_component > data_b_same_component;
        }

        // Rule 2: for matching entries, sort by reindexed county_component[region1]
        if (data_a_same_component) {
            return county_component_reindex[region_1a] < county_component_reindex[region_1b];
        }

        return false; // otherwise, keep existing relative order
    });

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("%d Components | NOW SORTED Pairs: \n", merged_num_admin_connected_components);
        for (auto const &val : all_pairs) {
            REprintf("Regions (%d, %d) | Components (%d, %d) | Shared Status %s | Edges %d\n",
                     val.first.first, val.first.second,
                     county_component_reindex[val.first.first],
                     county_component_reindex[val.first.second],
                     (val.second.same_admin_component ? "SHARED" : "NOT SHARED"),
                     (val.second.same_admin_component ? val.second.within_county_edges
                                                      : val.second.across_county_edges));
        }
    }

    double log_tau = 0.0;

    // current index through all_pairs
    int curr_index = 0;

    // Now we go through each component and compute log spanning trees
    for (int component_id = 0; component_id < num_county_connected_components; component_id++) {
        // Number of components in the region is same if not component 1
        // If component 1 then we add up the two regions
        int num_component_regions;
        if (two_components_merged) {
            if (component_id == region2_component) {
                // if two components merged then we ignore the second one
                continue;
            } else if (component_id == region1_component) {
                // number of components is the sum of the regions
                // minus 1 because merged
                num_component_regions = component_region_counts[component_id] +
                                        component_region_counts[region2_component] - 1;
            } else {
                num_component_regions = component_region_counts[component_id];
            }
        } else {
            num_component_regions = component_region_counts[component_id];
            // If merged component subtract a region
            if (component_id == region1_component) {
                --num_component_regions;
            }
        }
        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("Component %d which has %d regions!\n", component_id,
                     num_component_regions);
        }

        // stop loop if we've reached pairs across components
        if (curr_index >= all_pairs.size() ||
            (county_component_reindex[all_pairs[curr_index].first.first] !=
             county_component_reindex[all_pairs[curr_index].first.second]))
            break;

        // If 1 component then there only 1 tree so log(1) = 0
        if (num_component_regions <= 1) {
            // do nothing
            if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                REprintf("1 Region in Component! Log tau unchanged at %f\n", log_tau);
            }
            // do nothing
        } else if (num_component_regions == 2) {
            // If its two then its just the degree of either vertex which is the boundary length
            // Since its the same component its just within component edges

            // If this is the merged component then there are potentially more than 2 pairs
            if (component_id == region1_component) {
                int merged_boundary = 0;
                while (curr_index < all_pairs.size() &&
                       county_component_reindex[all_pairs[curr_index].first.first] ==
                           component_id &&
                       county_component_reindex[all_pairs[curr_index].first.second] ==
                           component_id) {
                    merged_boundary += all_pairs[curr_index].second.within_county_edges;
                    ++curr_index;
                }
                log_tau += std::log(static_cast<double>(merged_boundary));
                if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                    REprintf("2 Regions in Merged Component! | Boundary length %d |Log tau now "
                             "at %f\n",
                             merged_boundary, log_tau);
                }
            } else {
                // else if not the merged component there is actually only one pair
                log_tau += std::log(
                    static_cast<double>(all_pairs[curr_index].second.within_county_edges));
                if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                    REprintf("2 Regions in Non-merged Component! | Boundary length %d |Log tau "
                             "now at %f\n",
                             all_pairs[curr_index].second.within_county_edges, log_tau);
                }
                // now increment index
                ++curr_index;
            }
        } else {
            if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                REprintf("3 or more regions in component!\n");
            }
            if (component_id == region1_component) {
                // We will reindex the merged region to the last index in the component
                for (int current_reindex = 0, a_region_id = 0; a_region_id < num_regions;
                     a_region_id++) {
                    if (a_region_id == region1_id || a_region_id == region2_id) {
                        // the regions we're merging get reindex to the last region in the
                        // component
                        region_reindex_vec[a_region_id] = num_component_regions - 1;
                    } else if (county_component_reindex[a_region_id] == component_id) {
                        region_reindex_vec[a_region_id] = current_reindex;
                        ++current_reindex;
                    }
                    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                        REprintf("Mapping %d to %d!\n", a_region_id,
                                 region_reindex_vec[a_region_id]);
                    }
                }
            } else {
                // else more than two regions so we need to reindex the regions
                // This ensures each region in the component is indexed from
                // 0, ... num regions in the component
                for (int current_reindex = 0, a_region_id = 0; a_region_id < num_regions;
                     a_region_id++) {
                    // if in the component then reindex that region
                    if (county_component_reindex[a_region_id] == component_id) {
                        region_reindex_vec[a_region_id] = current_reindex;
                        ++current_reindex;
                        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                            REprintf("Mapping %d to %d!\n", a_region_id,
                                     region_reindex_vec[a_region_id]);
                        }
                    }
                }
            }

            // reset the laplacian where the number of vertices is num_component_regions
            // and the dropped vertex is num_component_regions - 1. In the merged
            // component both merged regions were reindexed to num_component_regions - 1,
            // so the merged region is the dropped vertex
            laplacian.reset(num_component_regions, num_component_regions - 1);

            // now we iterate through all pairs where both are in this component
            while (
                curr_index < all_pairs.size() &&
                county_component_reindex[all_pairs[curr_index].first.first] == component_id &&
                county_component_reindex[all_pairs[curr_index].first.second] == component_id) {

                const auto &[pair_region1, pair_region2] = all_pairs[curr_index].first;
                const PairHashData &pair_val = all_pairs[curr_index].second;

                // Because we filter out hier merge invalid we don't need to check

                double edges = 0;
                // else valid merge so get edges we count
                if (pair_val.admin_adjacent) {
                    edges =
                        static_cast<double>(all_pairs[curr_index].second.within_county_edges);
                } else {
                        std::ostringstream oss;

                        oss << "Error in compute_hierarchical_merged_log_multigraph_tau.\n";
                        oss << "Region pair is not administratively adjacent but was treated as hier-merge invalid.\n";
                        oss << "merged_num_admin_connected_components="
                            << merged_num_admin_connected_components << "\n";
                        oss << "pair_region1=" << pair_region1 << "\n";
                        oss << "pair_region2=" << pair_region2 << "\n";

                        if (pair_region1 >= 0 && pair_region1 < static_cast<int>(county_component.size()) &&
                            pair_region2 >= 0 && pair_region2 < static_cast<int>(county_component.size())) {
                            oss << "county_component[pair_region1]=" << county_component[pair_region1] << "\n";
                            oss << "county_component[pair_region2]=" << county_component[pair_region2] << "\n";
                        }

                        oss << debug_string();

                        throw std::runtime_error(oss.str());
                    }

                if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                    REprintf("Region Pair (%d, %d) | %d | %g edges!\n ", pair_region1,
                             pair_region2, pair_val.admin_adjacent, edges);
                }

                auto reindexed_pair_region1 = region_reindex_vec[pair_region1];
                auto reindexed_pair_region2 = region_reindex_vec[pair_region2];

                // Distinct regions never share a reindex except the two merged ones,
                // which both map to the dropped vertex num_component_regions - 1.
                // add_edge skips that pair as a self loop, which is right since its
                // edges are inside the merged region.
                if constexpr (perf_config::supposedly_safe_input_checks) {
                    if (reindexed_pair_region1 == reindexed_pair_region2 &&
                        reindexed_pair_region1 != num_component_regions - 1) {
                        std::ostringstream oss;
                        oss << "ERROR in compute_hierarchical_merged_log_multigraph_tau: "
                            << "equal reindexed pair inside component Laplacian.\n";
                        oss << "reindexed_pair_region1=" << reindexed_pair_region1 << "\n";
                        oss << "reindexed_pair_region2=" << reindexed_pair_region2 << "\n";
                        oss << "component_id=" << component_id << "\n";
                        oss << "num_component_regions=" << num_component_regions << "\n";
                        oss << "edges=" << edges << "\n";
                        oss << debug_string();

                        throw std::runtime_error(oss.str());
                    }
                }

                // Now for both pairs we need to
                // 1. Update the degree of the vertex
                // 2. Increase the count of edges between the regions
                // The laplacian builder handles that and skips adding the degree
                // and edge count for the dropped vertex num_component_regions - 1
                laplacian.add_edge(reindexed_pair_region1, reindexed_pair_region2, edges);

                // increase current index by 1
                ++curr_index;
            }
            if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                REprintf("Printing Laplacian Minor!\n");
            }
            // Now add log det
            log_tau += laplacian.log_det();
        }
    }

    if (curr_index >= all_pairs.size() && merged_num_admin_connected_components > 1) {
        std::ostringstream oss;

        oss << "ERROR in compute_hierarchical_merged_log_multigraph_tau: "
            << "curr_index ran past all_pairs before processing all merged admin components.\n";
        oss << "all_pairs.size()=" << all_pairs.size() << "\n";
        oss << "curr_index=" << curr_index << "\n";
        oss << "merged_num_admin_connected_components="
            << merged_num_admin_connected_components << "\n";

        oss << sorted_pairs_to_string(
            all_pairs,
            county_component,
            "NOW SORTED Pairs:"
        );

        oss << component_counts_to_string(
            component_region_counts,
            merged_num_admin_connected_components
        );

        oss << debug_string();

        throw std::runtime_error(oss.str());
    }

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("Remaining Across Merged Component Pairs: %zu\n",
                 all_pairs.size() - curr_index - 1);
        for (size_t i = curr_index; i < all_pairs.size(); i++) {
            auto val = all_pairs[i];
            REprintf("Regions (%d, %d) | Components (%d, %d) | Shared Status %s | Edges %d\n",
                     val.first.first, val.first.second,
                     county_component_reindex[val.first.first],
                     county_component_reindex[val.first.second],
                     (val.second.same_admin_component ? "SHARED" : "NOT SHARED"),
                     (val.second.same_admin_component ? val.second.within_county_edges
                                                      : val.second.across_county_edges));
        }
    }

    // If only one connected component then we stop
    if (merged_num_admin_connected_components == 1) {
        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("1 Component in entire Merged Plan! Log tau unchanged at %f\n", log_tau);
        }
        return log_tau;
    } else if (merged_num_admin_connected_components == 2) {
        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("2 Components in entire Merged Plan!\n");
        }
        int across_component_edges = 0;
        // If two then we just sum the edges across and take log
        for (int i = curr_index; i < all_pairs.size(); i++) {
            across_component_edges += all_pairs[i].second.across_county_edges;
        }
        // add
        log_tau += std::log(static_cast<double>(across_component_edges));
        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("Adding log of %d, Log tau now at %f\n", across_component_edges, log_tau);
        }
        return log_tau;
    }

    // now reset laplacian for size `merged_num_admin_connected_components`
    // with the ignore vertex being the last one, which the reindexing below
    // assigns to the merged component
    laplacian.reset(merged_num_admin_connected_components,
                    merged_num_admin_connected_components - 1);

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("Current index now %d with %zu pairs\n", curr_index, all_pairs.size());
    }

    // Now we don't care about specific component id so reindex things
    // to make the merged regions
    /*
    we make merged reindex which essentially just reindexes all the regions to
    the two merged regions are the new largest value and everything else gets
    potentially shifted down. For example imagine 5 regions
    and we merge 1 and 3 so the reindex vector is [0,3,1,3,2]
     */
    int const merged_component = merged_num_admin_connected_components - 1;
    for (int current_reindex = 0, a_component_id = 0;
         a_component_id < num_county_connected_components; a_component_id++) {
        // all components but region 2's get mapped to themselves
        if (a_component_id == region2_component || a_component_id == region1_component) {
            county_component_reindex[a_component_id] = merged_component;
        } else {
            county_component_reindex[a_component_id] = current_reindex;
            ++current_reindex;
        }
        // REprintf("Mapping %d to %d!\n", i, merge_index_reshuffle[i]);
    }

    // Iterate over the remaining pairs that are across components
    for (int i = curr_index; i < all_pairs.size(); i++) {
        const auto &[pair_region1, pair_region2] = all_pairs[i].first;
        const PairHashData &pair_val = all_pairs[i].second;

        int component1_id = county_component_reindex[county_component[pair_region1]];
        int component2_id = county_component_reindex[county_component[pair_region2]];

        // pairs from curr_index on were sorted to be across components
        if (pair_val.same_admin_component) {
            std::ostringstream oss;
            oss << "ERROR in compute_hierarchical_merged_log_multigraph_tau: region pair ("
                << static_cast<int>(pair_region1) << ", " << static_cast<int>(pair_region2)
                << ") at sorted index " << i << " (first across component index "
                << curr_index << ") is marked as in the same administrative component "
                << "but should be across merged components " << component1_id << " and "
                << component2_id << ".\n"
                << debug_string();
            throw std::runtime_error(oss.str());
        }

        double edges = static_cast<double>(pair_val.across_county_edges);
        if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
            REprintf("Components (%d,%d) - %g edges across!\n", component1_id, component2_id,
                     edges);
        }
        // Distinct components never share a reindex except the ones folded into
        // the merged component, the dropped vertex, which add_edge skips as a
        // self loop (same as the old code, which threw only when the ids were
        // equal and not the merged component)
        if constexpr (perf_config::supposedly_safe_input_checks) {
            if (component1_id == component2_id && component1_id != merged_component) {
                std::ostringstream oss;
                oss << "ERROR in compute_hierarchical_merged_log_multigraph_tau: "
                    << "equal component ids while constructing merged component Laplacian.\n";
                oss << "component1_id=" << component1_id << "\n";
                oss << "component2_id=" << component2_id << "\n";
                oss << "merged_num_admin_connected_components="
                    << merged_num_admin_connected_components << "\n";
                oss << "edges=" << edges << "\n";
                oss << debug_string();

                throw std::runtime_error(oss.str());
            }
        }

        // Now for both pairs we need to
        // 1. Update the degree of the vertex
        // 2. Increase the count of edges between the regions
        // The laplacian builder handles that and skips adding the degree
        // and edge count for the dropped merged component
        laplacian.add_edge(component1_id, component2_id, edges);
    }

    if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
        REprintf("Printing Component Laplacian Minor\n");
    }

    // Now add log det
    log_tau += laplacian.log_det();

    return log_tau;
}

PlanMultigraph::PlanMultigraph(MapParams const &map_params,
                               bool const need_to_compute_multigraph_taus
                            )
    : map_params(map_params), 
      // we only say counties are on if more than 1 county and we're not forcing 
      // non-hierarchicalness
      counties_on(map_params.num_counties > 1 && !map_params.plans_may_be_non_hierarchical),
      vertices_visited(map_params.V), county_component(map_params.ndists, 0),
      component_split_counts(counties_on ? map_params.ndists : 0, 0),
      component_region_counts(counties_on ? map_params.ndists : 0, 0),
      region_overlap_counties(counties_on ? map_params.ndists : 0),
      num_county_region_components(0), num_county_connected_components(0),
      // allocated on first use by the traversal or is_hierarchically_connected,
      // so a multigraph that only uses the single-pass build never pays for them
      other_counties_vertices(0),
      current_county_diff_region_vertices(0),
      current_county_region_vertices(0),
      mg_seen(counties_on ? map_params.ndists : 0, 0),
      mg_region_parent(counties_on ? map_params.ndists : 0, 0),
      mg_county_incidences(counties_on ? map_params.num_counties : 0, 0),
      mg_comp_of_root(counties_on ? map_params.ndists : 0, -1),
      mg_comp_counties(counties_on ? map_params.ndists : 0, 0),
      mg_comp_incidences(counties_on ? map_params.ndists : 0, 0),
      pair_map(map_params.ndists),
      need_to_compute_multigraph_taus(need_to_compute_multigraph_taus),
      county_component_reindex(need_to_compute_multigraph_taus ? map_params.ndists : 0, 0),
      region_reindex_vec(need_to_compute_multigraph_taus ? map_params.ndists : 0, 0),
      laplacian(need_to_compute_multigraph_taus)
      {

      };



std::string PlanMultigraph::debug_string() const {
    std::ostringstream oss;

    oss << (counties_on ? "YES" : "NO")
        << " Counties | Pair Map has "
        << pair_map.num_hashed_pairs
        << " Elements and "
        << num_county_connected_components
        << " components:\n";

    oss << pair_map.debug_string(county_component);

    return oss.str();
}

void PlanMultigraph::Rprint() const {
    Rcpp::Rcerr << debug_string();
}

std::string PlanMultigraph::debug_string_detailed(Plan const &plan) const {
    std::ostringstream oss;

    oss << (counties_on ? "YES" : "NO")
        << " Counties | Pair Map has "
        << pair_map.num_hashed_pairs
        << " Elements:\n";

    for (auto const &a_pair : pair_map.hashed_pairs) {
        auto val = pair_map.get_value(a_pair.first, a_pair.second).second;

        int const r1 = static_cast<int>(a_pair.first);
        int const r2 = static_cast<int>(a_pair.second);

        oss << "    Regions: (" << r1 << ", " << r2 << ")";

        if (r1 >= 0 && r1 < plan.num_regions &&
            r2 >= 0 && r2 < plan.num_regions) {
            oss << " | Sizes ("
                << static_cast<int>(plan.region_sizes[r1]) << ", "
                << static_cast<int>(plan.region_sizes[r2]) << ")";
        } else {
            oss << " | Sizes invalid";
        }

        oss << " | "
            << (val.admin_adjacent ? "YES" : "NOT")
            << " Hierarchically Adjacent"
            << " | " << val.within_county_edges << " within county edges"
            << " | " << val.across_county_edges << " across county edges"
            << "\n";
    }

    return oss.str();
}

void PlanMultigraph::Rprint_detailed(Plan const &plan) {
    Rcpp::Rcerr << debug_string_detailed(plan);
}

void PlanMultigraph::check_representative_edges(
    PlanVector const &region_ids
) const {
    int const V = map_params.V;

    for (auto const &[region1_id, region2_id] :
         pair_map.hashed_pairs) {

        auto const hash_index =
            pair_map.pair_hash(
                region1_id,
                region2_id
            );

        PairHashData const &pair_data =
            pair_map.values[hash_index];

        int const v1 =
            pair_data.representative_vertex1;

        int const v2 =
            pair_data.representative_vertex2;

        /*
         * Every hashed pair represents at least one actual map edge,
         * so it must have a representative edge.
         */
        if (
            v1 < 0 || v1 >= V ||
            v2 < 0 || v2 >= V
        ) {
            std::ostringstream oss;

            oss << "Invalid representative edge in PlanMultigraph.\n";
            oss << "regions=("
                << static_cast<int>(region1_id)
                << ", "
                << static_cast<int>(region2_id)
                << ")\n";
            oss << "representative_edge=("
                << v1
                << ", "
                << v2
                << ")\n";
            oss << "V=" << V << "\n";

            throw std::runtime_error(oss.str());
        }

        /*
         * Representative edges are stored in canonical region order:
         *
         *     v1 belongs to region1_id
         *     v2 belongs to region2_id
         *
         * where region1_id < region2_id.
         */
        if (
            region_ids[v1] != region1_id ||
            region_ids[v2] != region2_id
        ) {
            std::ostringstream oss;

            oss << "Representative edge has incorrect region orientation.\n";
            oss << "stored region pair=("
                << static_cast<int>(region1_id)
                << ", "
                << static_cast<int>(region2_id)
                << ")\n";
            oss << "representative_edge=("
                << v1
                << ", "
                << v2
                << ")\n";
            oss << "actual endpoint regions=("
                << static_cast<int>(region_ids[v1])
                << ", "
                << static_cast<int>(region_ids[v2])
                << ")\n";

            throw std::runtime_error(oss.str());
        }

        /*
         * Check that (v1, v2) is genuinely an edge of the map graph.
         *
         * This linear search through the adjacency list is fine here
         * because object_integrity_checking is explicitly expensive.
         */
        auto const &v1_nbors =
            map_params.g[v1];

        bool const is_map_edge =
            std::find(
                v1_nbors.begin(),
                v1_nbors.end(),
                v2
            ) != v1_nbors.end();

        if (!is_map_edge) {
            std::ostringstream oss;

            oss << "Representative region edge is not a map edge.\n";
            oss << "regions=("
                << static_cast<int>(region1_id)
                << ", "
                << static_cast<int>(region2_id)
                << ")\n";
            oss << "representative_edge=("
                << v1
                << ", "
                << v2
                << ")\n";

            throw std::runtime_error(oss.str());
        }

        /*
         * A hashed pair must have at least one boundary edge.
         */
        if (
            pair_data.within_county_edges +
                pair_data.across_county_edges <=
            0
        ) {
            std::ostringstream oss;

            oss << "Hashed region pair has no map boundary edges.\n";
            oss << "regions=("
                << static_cast<int>(region1_id)
                << ", "
                << static_cast<int>(region2_id)
                << ")\n";

            throw std::runtime_error(oss.str());
        }

        if (counties_on) {
            bool const same_county =
                map_params.counties[v1] ==
                map_params.counties[v2];

            /*
             * If the pair is administratively adjacent, our representative
             * must be one of its within-county boundary edges.
             */
            if (
                pair_data.admin_adjacent &&
                !same_county
            ) {
                std::ostringstream oss;

                oss << "Administratively adjacent region pair has an "
                       "across-county representative edge.\n";
                oss << "regions=("
                    << static_cast<int>(region1_id)
                    << ", "
                    << static_cast<int>(region2_id)
                    << ")\n";
                oss << "representative_edge=("
                    << v1
                    << ", "
                    << v2
                    << ")\n";

                throw std::runtime_error(oss.str());
            }

            /*
             * Conversely, if there is no within-county adjacency, the
             * representative should be across a county boundary.
             */
            if (
                !pair_data.admin_adjacent &&
                same_county
            ) {
                std::ostringstream oss;

                oss << "Non-admin-adjacent region pair has a "
                       "within-county representative edge.\n";
                oss << "regions=("
                    << static_cast<int>(region1_id)
                    << ", "
                    << static_cast<int>(region2_id)
                    << ")\n";
                oss << "representative_edge=("
                    << v1
                    << ", "
                    << v2
                    << ")\n";

                throw std::runtime_error(oss.str());
            }
        }
    }
}

void PlanMultigraph::check_matches_other_build(PlanMultigraph const &other,
                                               bool const this_result,
                                               bool const other_result,
                                               PlanVector const &region_ids,
                                               int const num_regions) const {
    std::ostringstream problems;
    int num_problems = 0;
    constexpr int max_reported = 25;
    auto report = [&](std::string const &problem) {
        if (num_problems < max_reported) problems << "  " << problem << "\n";
        ++num_problems;
    };
    auto compare = [&](std::string const &what, int const mine, int const theirs) {
        if (mine != theirs) {
            report(what + ": " + std::to_string(mine) + " here vs " + std::to_string(theirs));
        }
    };

    if (this_result != other_result) {
        report(std::string("return value: ") + (this_result ? "valid" : "invalid") +
               " here vs " + (other_result ? "valid" : "invalid"));
    } else if (this_result) {
        // the rest is only defined when the plan is valid
        compare("num_county_connected_components", num_county_connected_components,
                other.num_county_connected_components);
        compare("num_county_region_components", num_county_region_components,
                other.num_county_region_components);
        compare("num_hashed_pairs", pair_map.num_hashed_pairs, other.pair_map.num_hashed_pairs);
        compare("num_hier_smc_merge_valid_pairs", pair_map.num_hier_smc_merge_valid_pairs,
                other.pair_map.num_hier_smc_merge_valid_pairs);
        for (int r = 0; r < num_regions; ++r) {
            compare("county_component[" + std::to_string(r) + "]", county_component[r],
                    other.county_component[r]);
        }
        int const shared_components =
            std::min(num_county_connected_components, other.num_county_connected_components);
        for (int k = 0; k < shared_components; ++k) {
            compare("component_split_counts[" + std::to_string(k) + "]",
                    component_split_counts[k], other.component_split_counts[k]);
            compare("component_region_counts[" + std::to_string(k) + "]",
                    component_region_counts[k], other.component_region_counts[k]);
        }
        // every pair here must be in the other build with the same counts and
        // flags; with equal pair counts (checked above) the two sets are equal
        for (auto const &a_pair : pair_map.hashed_pairs) {
            std::string const name = "pair (" + std::to_string(a_pair.first) + ", " +
                                     std::to_string(a_pair.second) + ")";
            auto const theirs_lookup = other.pair_map.get_value(a_pair.first, a_pair.second);
            if (!theirs_lookup.first) {
                report(name + " is missing from the other build");
                continue;
            }
            PairHashData const mine = pair_map.get_value(a_pair.first, a_pair.second).second;
            PairHashData const &theirs = theirs_lookup.second;
            compare(name + " within_county_edges", mine.within_county_edges,
                    theirs.within_county_edges);
            compare(name + " across_county_edges", mine.across_county_edges,
                    theirs.across_county_edges);
            compare(name + " admin_adjacent", mine.admin_adjacent, theirs.admin_adjacent);
            compare(name + " same_admin_component", mine.same_admin_component,
                    theirs.same_admin_component);
            compare(name + " merge_is_hier_valid", mine.merge_is_hier_valid,
                    theirs.merge_is_hier_valid);
            compare(name + " shared_county", mine.shared_county, theirs.shared_county);
        }
    }

    if (num_problems > 0) {
        std::ostringstream oss;
        oss << "PlanMultigraph::check_matches_other_build: the two hierarchical multigraph "
            << "builds disagree on a plan with " << num_regions << " regions ("
            << num_problems << " differences";
        if (num_problems > max_reported) oss << ", first " << max_reported << " shown";
        oss << "):\n" << problems.str();
        oss << "Region ids: c(";
        for (int v = 0; v < map_params.V; ++v) {
            oss << static_cast<int>(region_ids[v]) << (v + 1 < map_params.V ? "," : "");
        }
        oss << ")\n";
        throw std::runtime_error(oss.str());
    }
}

// indexing (i,j) in a matrix
// assuming the length of a row is row_length
// i is the row index
// j is the column index so j < row_length
inline size_t get_county_component_lookup_index(int const county_id, RegionID const region_id,
                                                int const num_regions) {
    return county_id * num_regions + region_id;
}

// checks if a plan is hierarchically connected and if it is then it
// returns the number of county region components
std::pair<bool, int>
PlanMultigraph::is_hierarchically_connected(Plan const &plan,
                                            std::vector<bool> component_lookup) {
    return is_hierarchically_connected(
        plan.region_ids, plan.num_regions, std::move(component_lookup)
    );
}

std::pair<bool, int>
PlanMultigraph::is_hierarchically_connected(PlanVector const &region_ids,
                                            int const num_regions,
                                            std::vector<bool> component_lookup) {
    // if no counties then its just one giant component
    if (!counties_on)
        return std::make_pair(true, 1);
    // reset the vertices visited vector
    std::fill(vertices_visited.begin(), vertices_visited.end(), false);
    // allocate the queue on first use
    current_county_region_vertices.ensure_capacity(
        static_cast<std::size_t>(map_params.num_edges) + 1);
    // component_lookup should be length num_countes * num_regions
    // fill this vector with all false
    std::fill(component_lookup.begin(), component_lookup.end(), false);
    // reset the number of county region components
    num_county_region_components = 0;

    // loop through the graph
    for (auto v = 0; v < map_params.V; v++) {
        // skip if we've already visited this vertex
        if (vertices_visited[v])
            continue;

        // get current county
        int current_county = map_params.counties[v] - 1;
        auto current_region = region_ids[v];
        // for lookup we pretend table is num_counties x num_regions
        auto current_component_lookup_index =
            get_county_component_lookup_index(current_county, current_region, num_regions);

        // check if we've already declared this component
        // if default value then we haven't encountered this component yet
        if (!component_lookup[current_component_lookup_index]) {
            // increase number of connected components
            ++num_county_region_components;
            // mark as true
            component_lookup[current_component_lookup_index] = true;
        } else {
            // else we have seen this before in which case its not hierarchically connected so
            // immediately return false
            return std::make_pair(false, -1);
        }

        // Now we traverse all neighbors in the same component
        current_county_region_vertices.clear();
        current_county_region_vertices.push(v);

        while (!current_county_region_vertices.empty()) {
            int u = current_county_region_vertices.pop();
            int u_region = region_ids[u];
            int u_county = map_params.counties[u];

            // mark this as visited
            vertices_visited[u] = true;
            // now visit each of the vertices children
            for (auto const child_vertex : map_params.g[u]) {
                // ignore if we already visited
                if (vertices_visited[child_vertex])
                    continue;
                // Now only add if region and county are the same
                if (u_region == region_ids[child_vertex] &&
                    u_county == map_params.counties[child_vertex]) {
                    // if same then same component mark as visited to avoid being added later
                    vertices_visited[child_vertex] = true;
                    current_county_region_vertices.push(child_vertex);
                }
            }
        }
    }

    // if we've passed through the whole graph then now return true
    return std::make_pair(true, num_county_region_components);
}

void PlanMultigraph::build_plan_non_hierarchical_multigraph(PlanVector const &region_ids) {
    // reset the hash map
    pair_map.reset();

    for (int v = 0; v < map_params.V; v++) {
        auto v_region = region_ids[v];

        for (auto const u : map_params.map_graph.neighbors(v)) {
            auto u_region = region_ids[u];
            if (v_region < u_region) {
                // if true then count
                pair_map.count_graph_edge(v_region, u_region, v, u, true, 0);
            }
        }
    }

    pair_map.num_hier_smc_merge_valid_pairs = pair_map.num_hashed_pairs;

    return;
}

bool PlanMultigraph::build_plan_hierarchical_multigraph_traversal(PlanVector const &region_ids,
                                                                  int const num_regions) {
    // reset the hash map
    pair_map.reset();
    // reset the vertices visited vector
    std::fill(vertices_visited.begin(), vertices_visited.end(), false);
    // reset the region component assignments and countes
    std::fill(county_component.begin(), county_component.end(), 0);
    std::fill(component_split_counts.begin(), component_split_counts.end(), 0);
    std::fill(component_region_counts.begin(), component_region_counts.end(), 0);

    // reset the number of county region components
    num_county_region_components = 0;
    // counter for labelling components of administratively connected
    // quotient graph. reset that
    num_county_connected_components = 0;

    // allocate the queues on first use, then clear them
    std::size_t const queue_capacity = static_cast<std::size_t>(map_params.num_edges) + 1;
    other_counties_vertices.ensure_capacity(queue_capacity);
    current_county_diff_region_vertices.ensure_capacity(queue_capacity);
    current_county_region_vertices.ensure_capacity(queue_capacity);
    other_counties_vertices.clear();
    current_county_diff_region_vertices.clear();
    current_county_region_vertices.clear();

    // Now we walk through the graph
    for (int w = 0; w < map_params.V; w++) {
        // skip if already visited
        if (vertices_visited[w])
            continue;

        // else we've encountered a new component
        int num_current_county_region_components = 1;
        int num_current_counties = 1;

        // add this vertex and mark as visited
        current_county_region_vertices.push(w);
        // assign this region to the component
        county_component[region_ids[w]] = num_county_connected_components;

        while (!current_county_region_vertices.empty() ||
               !current_county_diff_region_vertices.empty() ||
               !other_counties_vertices.empty()) {

            int v = -1;

            // first see if anything in this region and county
            if (!current_county_region_vertices.empty()) {
                v = current_county_region_vertices.pop();
            } else if (!current_county_diff_region_vertices.empty()) {
                // else we've got a vertex in a different region in the same county
                v = current_county_diff_region_vertices.pop();
                // if we haven't visited this vertex yet that means we're reaching
                // this component for the first time
                if (!vertices_visited[v]) {
                    // then change the current region and increase the count
                    num_current_county_region_components++;
                    // increase global count
                    num_county_region_components++;
                    // mark the region as being in current component
                    county_component[region_ids[v]] = num_county_connected_components;
                    // REprintf("Assigned Region %u to component %d\n",
                    //     plan.region_ids[v]+1,
                    // num_county_connected_components);
                }
            } else if (!other_counties_vertices.empty()) {
                // else we've got a vertex in a different county
                v = other_counties_vertices.pop();
                // if we haven't visited this vertex yet that means we're reaching
                // this county for the first time
                if (!vertices_visited[v]) {
                    // then change the current region and increase the count
                    num_current_county_region_components++;
                    num_current_counties++;
                    // increase global count
                    num_county_region_components++;
                    // mark the region as being in current component
                    county_component[region_ids[v]] = num_county_connected_components;
                    // REprintf("Assigned Region %u to component %d\n",
                    //     plan.region_ids[v]+1,
                    // num_county_connected_components);
                }
            }

            // skip if we already visited
            if (vertices_visited[v])
                continue;
            // mark as visited now
            vertices_visited[v] = true;

            auto v_region = region_ids[v];
            auto v_county = map_params.counties[v];

            for (int u : map_params.map_graph.neighbors(v)) {
                // get region and county
                auto u_region = region_ids[u];
                auto u_county = map_params.counties[u];
                bool same_county = u_county == v_county;

                // if v_region < u_region then count the edge
                if (v_region < u_region) {
                    pair_map.count_graph_edge(v_region, u_region, v, u, same_county, u_county - 1);
                }
                // if already visited don't add to the queue again
                if (vertices_visited[u])
                    continue;
                // else add to queue
                if (u_county == v_county) {
                    if (u_region == v_region) {
                        // if same region and county add to
                        // queue of same region and county vertices
                        current_county_region_vertices.push(u);
                    } else {
                        // else if same county different region add to that
                        // queue
                        current_county_diff_region_vertices.push(u);
                    }
                } else if (u_region == v_region) {
                    // if differerent counties then only add if same region
                    // we do not explore edges across both regions and counties
                    other_counties_vertices.push(u);
                }
            }
        }

        // now we've visited every vertex in this county adj component so get the count
        component_split_counts[num_county_connected_components] =
            num_current_county_region_components - num_current_counties;

        if constexpr (DEBUG_BASE_PLANS_VERBOSE) {
            REprintf("Component %d has %d components, %d counties so %d splits!\n",
                     num_county_connected_components, num_current_county_region_components,
                     num_current_counties,
                     num_current_county_region_components - num_current_counties);
        }

        // now increment the counter by 1
        num_county_connected_components++;

        // if too many global splits then auto reject
        if (num_county_region_components - map_params.num_counties >= num_regions) {
            if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                REprintf("EARLY RETURN A: total county-region components=%d, num_counties=%d, "
                         "num_regions=%d\n",
                         num_county_region_components, map_params.num_counties, num_regions);
                Rprint();
            }
            return false;
        }
    }

    // now for each component add up the number of regions in the component
    for (size_t region_id = 0; region_id < num_regions; region_id++) {
        ++component_region_counts[county_component[region_id]];
    }

    // now check for each component the number of splits is the number of regions
    // minus 1
    for (size_t component_id = 0; component_id < num_county_connected_components;
         component_id++) {
        if constexpr (DEBUG_BASE_PLANS_VERBOSE) {
            REprintf("Component %zu has %d splits and %d regions!\n", component_id,
                     component_split_counts[component_id],
                     component_region_counts[component_id]);
        }

        // error check, this shouldn't be possible
        if (component_split_counts[component_id] < component_region_counts[component_id] - 1) {
            REprintf(
                "CODE ERROR!!!! Somehow county adj component has %d regions but %d splits!\n",
                component_region_counts[component_id], component_split_counts[component_id]);
        } else if (component_split_counts[component_id] >=
                   component_region_counts[component_id]) {
            if constexpr (DEBUG_LOG_LINK_EDGE_VERBOSE) {
                std::ostringstream oss;

                oss << "EARLY RETURN B in hierarchical merge-valid check.\n";
                oss << "component_id=" << component_id << "\n";
                oss << "component_split_counts[component_id]="
                    << component_split_counts[component_id] << "\n";
                oss << "component_region_counts[component_id]="
                    << component_region_counts[component_id] << "\n";

                oss << "Region ids: c(";
                for (std::size_t k = 0; k < region_ids.size(); ++k) {
                    oss << static_cast<int>(region_ids[k]);
                    if (k + 1 < region_ids.size()) {
                        oss << ",";
                    }
                }
                oss << ")\n";

                oss << debug_string();

                throw std::runtime_error(oss.str());
            }
            return false;
        }
    }

    pair_map.num_hier_smc_merge_valid_pairs = pair_map.num_hashed_pairs;

    // Now mark which pairs are hierarchically ok to merge
    for (auto const a_pair : pair_map.hashed_pairs) {
        // if different components then its ok. Since merge ok by default then continue
        if (county_component[a_pair.first] != county_component[a_pair.second])
            continue;

        // else if they're the same component check if admin adjacent
        auto hash_index = pair_map.pair_hash(a_pair.first, a_pair.second);
        // update to say same componet
        pair_map.values[hash_index].same_admin_component = true;
        // if admin adjacent do nothing
        if (pair_map.values[hash_index].admin_adjacent)
            continue;
        // else reset values mark merge as false
        pair_map.values[hash_index].merge_is_hier_valid = false;
        --pair_map.num_hier_smc_merge_valid_pairs;
    }

    return true;
}

/*
 * Builds the hierarchical plan multigraph from one pass over the map's edges,
 * for plans where every region intersect county is a single connected piece (see
 * the declaration). It gives the same pair counts, county components and their
 * numbering, split and region counts, counters and validity as the traversal.
 * Only the order of pair_map.hashed_pairs and the representative edges can differ.
 *
 * It relies on four facts, using that counties are contiguous (R splits
 * non-contiguous counties before they reach C++):
 *   1. On these plans the number of county region pieces is the number of
 *      (region, county) incidences: pairs where the region has a vertex in the
 *      county.
 *   2. In a county holding two or more regions, every region present has a
 *      within-county edge to a different region, so the regions seen on the
 *      county's within-county boundary edges are exactly the regions in it. A
 *      county with no such edge lies inside one region.
 *   3. Regions in the same county are joined inside it by within-county edges
 *      between different regions, and a county component is a group of regions
 *      linked through shared counties, so a union-find over regions on those
 *      edges finds the components.
 *   4. Every component is made of whole counties, so numbering components in
 *      order of their counties' smallest vertices reproduces the traversal's
 *      numbering (by smallest vertex).
 *
 * Steps:
 *   0. reset the outputs and the scratch
 *   1. within-county edges, one county block at a time: count the pair data,
 *      mark each region present in the county (mg_seen) and unite the two
 *      regions (mg_region_parent)
 *   2. cross-county edges: count the pair data
 *   3. number the components, count the counties, regions and incidences in
 *      each, set the counters and check validity
 *   4. mark which pairs can merge hierarchically (the same as the traversal)
 */
bool PlanMultigraph::build_plan_hierarchical_multigraph_single_pass(PlanVector const &region_ids,
                                                                    int const num_regions) {
    int const num_counties = map_params.num_counties;
    std::vector<MapEdge> const &edges = map_params.county_sorted_edges.edges;
    // county c's within-county edges are edges[block[c]] up to edges[block[c + 1] - 1],
    // and the cross-county edges run from edges[block[num_counties]] to the end
    std::vector<int> const &block = map_params.county_sorted_edges.county_offsets;
    int const n_edges = static_cast<int>(edges.size());

    // union-find root of region r, halving the path on the way up
    auto find_root = [this](int r) {
        // while the parent of a region is not itself keep moving up
        while (mg_region_parent[r] != r) {
            mg_region_parent[r] = mg_region_parent[mg_region_parent[r]];
            r = mg_region_parent[r];
        }
        return r;
    };

    // ---- 0. reset ----
    pair_map.reset();
    std::fill(county_component.begin(), county_component.end(), 0);
    std::fill(component_split_counts.begin(), component_split_counts.end(), 0);
    std::fill(component_region_counts.begin(), component_region_counts.end(), 0);
    for (int r = 0; r < num_regions; ++r) {
        mg_seen[r] = 0;          // not seen in any county yet
        mg_region_parent[r] = r; // every region its own group
    }

    // ---- 1. within-county edges, one county block at a time ----
    int total_pieces = 0; // incidences summed over counties
    for (int c = 0; c < num_counties; ++c) {
        int const mark = c + 1; // mg_seen value meaning "seen in county c"
        int regions_in_county = 0;
        for (int e = block[c]; e < block[c + 1]; ++e) {
            int const u = edges[e].u;
            int const v = edges[e].v;
            auto const u_region = region_ids[u];
            auto const v_region = region_ids[v];
            if (u_region == v_region) continue; // inside one region: nothing to record
            // two regions will overlap in at most 1 admin unit so this check is ok
            pair_map.count_graph_edge(u_region, v_region, u, v, true, c);
            if (mg_seen[u_region] != mark) {
                mg_seen[u_region] = mark;
                ++regions_in_county;
            }
            if (mg_seen[v_region] != mark) {
                mg_seen[v_region] = mark;
                ++regions_in_county;
            }
            int const u_root = find_root(u_region);
            int const v_root = find_root(v_region);
            if (u_root != v_root) mg_region_parent[v_root] = u_root;
        }
        // no boundary edge inside the county, so it lies inside one region
        if (regions_in_county == 0) regions_in_county = 1;
        mg_county_incidences[c] = regions_in_county;
        total_pieces += regions_in_county;
    }

    // ---- 2. cross-county edges ----
    for (int e = block[num_counties]; e < n_edges; ++e) {
        int const u = edges[e].u;
        int const v = edges[e].v;
        auto const u_region = region_ids[u];
        auto const v_region = region_ids[v];
        if (u_region != v_region) {
            pair_map.count_graph_edge(u_region, v_region, u, v, false, -1);
        }
    }

    // ---- 3. number the components, count, and check validity ----
    std::fill(mg_comp_of_root.begin(), mg_comp_of_root.begin() + num_regions, -1);
    int num_components = 0;
    for (int const c : map_params.counties_by_min_vertex) {
        // every region in the county is in the same component, so take the one
        // holding the county's smallest vertex
        int const root = find_root(region_ids[map_params.county_vertices[c][0]]);
        if (mg_comp_of_root[root] < 0) { // first county of a new component
            mg_comp_of_root[root] = num_components;
            mg_comp_counties[num_components] = 0;
            mg_comp_incidences[num_components] = 0;
            ++num_components;
        }
        int const component = mg_comp_of_root[root];
        ++mg_comp_counties[component];
        mg_comp_incidences[component] += mg_county_incidences[c];
    }
    for (int r = 0; r < num_regions; ++r) {
        int const component = mg_comp_of_root[find_root(r)];
        county_component[r] = component;
        ++component_region_counts[component];
    }
    num_county_connected_components = num_components;
    // the traversal's meaning: pieces beyond the first one in each component
    num_county_region_components = total_pieces - num_components;

    for (int component = 0; component < num_components; ++component) {
        component_split_counts[component] =
            mg_comp_incidences[component] - mg_comp_counties[component];
        if (component_split_counts[component] < component_region_counts[component] - 1) {
            // impossible: a component's region-county graph is connected
            REprintf(
                "CODE ERROR!!!! Somehow county adj component has %d regions but %d splits!\n",
                component_region_counts[component], component_split_counts[component]);
        } else if (component_split_counts[component] >= component_region_counts[component]) {
            // the component's region-county graph has a cycle as there's too many components 
            return false;
        }
    }

    // ---- 4. pairs in the same component can only merge if admin adjacent ----
    pair_map.num_hier_smc_merge_valid_pairs = pair_map.num_hashed_pairs;
    for (auto const a_pair : pair_map.hashed_pairs) {
        if (county_component[a_pair.first] != county_component[a_pair.second])
            continue;
        auto hash_index = pair_map.pair_hash(a_pair.first, a_pair.second);
        pair_map.values[hash_index].same_admin_component = true;
        if (pair_map.values[hash_index].admin_adjacent)
            continue;
        pair_map.values[hash_index].merge_is_hier_valid = false;
        --pair_map.num_hier_smc_merge_valid_pairs;
    }

    return true;
}

bool PlanMultigraph::build_plan_multigraph(PlanVector const &region_ids,
                                           int const num_regions) {
    bool result;
    if (counties_on) {
        result = build_plan_hierarchical_multigraph_single_pass(region_ids, num_regions);
    } else {
        build_plan_non_hierarchical_multigraph(region_ids);
        result = true;
    }

    // Debug builds check the representative edges and, with counties on, also
    // run the traversal on a scratch multigraph and check the two builds agree
    if constexpr (perf_config::object_integrity_checking) {
        if (result) check_representative_edges(region_ids);
        if (counties_on) {
            PlanMultigraph traversal(map_params);
            bool const traversal_result =
                traversal.build_plan_hierarchical_multigraph_traversal(region_ids, num_regions);
            check_matches_other_build(traversal, result, traversal_result, region_ids,
                                      num_regions);
        }
    }

    return result;
}

bool PlanMultigraph::build_plan_multigraph_traversal(PlanVector const &region_ids,
                                                     int const num_regions) {
    bool successful;
    if (counties_on) {
        successful = build_plan_hierarchical_multigraph_traversal(region_ids, num_regions);
    } else {
        build_plan_non_hierarchical_multigraph(region_ids);
        successful = true;
    }
    /*
     * Only check after a successful complete construction.
     *
     * A failed hierarchical build may have returned before pair_map
     * represents a complete plan multigraph.
     */
    if constexpr (
        perf_config::object_integrity_checking
    ) {
        if (successful) {
            check_representative_edges(
                region_ids
            );
        }
    }

    return successful;
}

void PlanMultigraph::remove_invalid_size_pairs(Plan const &plan,
                                               SplittingSchedule const &splitting_schedule,
                                               ScheduleContext const &schedule_ctx) {

    // remove invalid sizes
    pair_map.hashed_pairs.erase(
        std::remove_if(pair_map.hashed_pairs.begin(), pair_map.hashed_pairs.end(),
                       [&](std::pair<RegionID, RegionID> a_pair) {
                           bool invalid_sizing = !splitting_schedule.is_valid_merge_pair(
                               plan, schedule_ctx, a_pair.first, a_pair.second);
                           // if invalid then reset data in pair map
                           if (invalid_sizing) {
                               auto hash_index =
                                   pair_map.pair_hash(a_pair.first, a_pair.second);
                               // else reset values and decrease hash table count
                               --pair_map.num_hashed_pairs;
                               pair_map.hashed[hash_index] = false;
                               pair_map.values[hash_index] = PairHashData();
                           }

                           // remove if any size is invalid split size or if merged size can't
                           // be split
                           return invalid_sizing;
                       }),
        pair_map.hashed_pairs.end());
}

void PlanMultigraph::remove_invalid_hard_constraint_pairs(
    Plan const &plan, ScoringFunction const &scoring_function) {
    // do nothing if no hard constraints
    if (!scoring_function.any_hard_constraints)
        return;

    pair_map.hashed_pairs.erase(
        std::remove_if(pair_map.hashed_pairs.begin(), pair_map.hashed_pairs.end(),
                       [&](std::pair<RegionID, RegionID> a_pair) {
                           // check if merging the region invalidates hard constraints
                           bool failed_constraint = !scoring_function.merged_plan_ok(
                               plan, a_pair.first, a_pair.second);

                           // if invalid then reset data in pair map
                           if (failed_constraint) {
                               auto hash_index =
                                   pair_map.pair_hash(a_pair.first, a_pair.second);
                               // else reset values and decrease hash table count
                               --pair_map.num_hashed_pairs;
                               pair_map.hashed[hash_index] = false;
                               pair_map.values[hash_index] = PairHashData();
                           }

                           // remove if doesn't satisfy hard constraint
                           return failed_constraint;
                       }),
        pair_map.hashed_pairs.end());
}

void PlanMultigraph::remove_invalid_hierarchical_merge_pairs(Plan const &plan) {
    if (!counties_on)
        return;

    // Now remove the pairs that are in the same component but not administratively adjacent
    // remove invalid sizes
    pair_map.hashed_pairs.erase(
        std::remove_if(pair_map.hashed_pairs.begin(), pair_map.hashed_pairs.end(),
                       [&](std::pair<RegionID, RegionID> a_pair) {
                           auto hash_index = pair_map.pair_hash(a_pair.first, a_pair.second);
                           if (pair_map.values[hash_index].merge_is_hier_valid)
                               return false;

                           // else reset values and decrease hash table count
                           --pair_map.num_hashed_pairs;
                           pair_map.hashed[hash_index] = false;
                           pair_map.values[hash_index] = PairHashData();
                           return true;
                       }),
        pair_map.hashed_pairs.end());

    return;
}

void PlanMultigraph::remove_invalid_mergesplit_pairs(Plan const &plan) {
    if (!counties_on)
        return;
    // reset the sets
    for (size_t i = 0; i < region_overlap_counties.size(); i++) {
        region_overlap_counties[i].clear();
    }
    // iterate through the pairs and note the counties they overlap in
    for (auto const &key_val_pair : pair_map.get_all_values()) {
        if (key_val_pair.second.admin_adjacent) {
            // if they overlap then add that county to both of the regions
            region_overlap_counties[key_val_pair.first.first].insert(
                key_val_pair.second.shared_county);
            region_overlap_counties[key_val_pair.first.second].insert(
                key_val_pair.second.shared_county);
        }
    }

    // Now we go through and delete and adjacent regions where
    // - they ARE NOT admin adjacent and they are both contain the same county

    pair_map.hashed_pairs.erase(
        std::remove_if(
            pair_map.hashed_pairs.begin(), pair_map.hashed_pairs.end(),
            [&](std::pair<RegionID, RegionID> a_pair) {
                // if different components then its ok
                if (county_component[a_pair.first] != county_component[a_pair.second])
                    return false;
                // else if they're the same component check if admin adjacent
                auto hash_index = pair_map.pair_hash(a_pair.first, a_pair.second);
                // if admin adjacent do nothing because thats ok
                if (pair_map.values[hash_index].admin_adjacent)
                    return false;

                bool invalid_merge = false;
                // now check if they overlap in any of the two counties
                if (region_overlap_counties[a_pair.first].size() <
                    region_overlap_counties[a_pair.second].size()) {
                    // check if any element of the smaller set is in the larger set
                    for (auto const a_county : region_overlap_counties[a_pair.first]) {
                        if (region_overlap_counties[a_pair.second].find(a_county) !=
                            region_overlap_counties[a_pair.second].end()) {
                            invalid_merge = true;
                            break;
                        }
                    }
                } else {
                    for (auto const a_county : region_overlap_counties[a_pair.second]) {
                        if (region_overlap_counties[a_pair.first].find(a_county) !=
                            region_overlap_counties[a_pair.first].end()) {
                            invalid_merge = true;
                            break;
                        }
                    }
                }
                if (invalid_merge) {
                    // if invalid merge deteced then remove
                    --pair_map.num_hashed_pairs;
                    pair_map.hashed[hash_index] = false;
                    pair_map.values[hash_index] = PairHashData();
                }
                return invalid_merge;
            }),
        pair_map.hashed_pairs.end());

    return;
}

bool PlanMultigraph::is_hierarchically_valid(Plan const &plan,
                                             std::vector<bool> component_lookup) {
    return is_hierarchically_valid(
        plan.region_ids, plan.num_regions, std::move(component_lookup)
    );
}

bool PlanMultigraph::is_hierarchically_valid(PlanVector const &region_ids,
                                             int const num_regions,
                                             std::vector<bool> component_lookup) {
    // if no counties then always valid
    if (!counties_on)
        return true;
    // first check if hierarchically connected
    auto result = is_hierarchically_connected(region_ids, num_regions, component_lookup);
    // if not immediately return false
    if (!result.first)
        return false;
    // if number of splits is greater than number of regions minus 1 reject
    if (result.second - map_params.num_counties >= num_regions)
        return false;

    // Now we know its hierarchically connected and has at most num_regions-1
    // splits so we just need to check for cycles in the administratively
    // adjacent quotient graph
    return build_plan_hierarchical_multigraph_traversal(region_ids, num_regions);
}
