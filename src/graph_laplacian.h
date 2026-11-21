#pragma once
#ifndef GRAPH_LAPLACIAN_H
#define GRAPH_LAPLACIAN_H

#include <vector>

#include "advanced_types.h" // MapParams, PlanVector
#include "sparse_logdet.h"  // LaplacianBuilder

class GraphLaplacian {
  public:
    // Sizes the scratch buffers for `map_params`, which must outlive this
    // object. One GraphLaplacian per thread, since calls reuse the buffers.
    //
    // With `allocate` false nothing is allocated, for runs that never compute
    // tau (eg compactness 1 in graph space), and calling either public method
    // is an error: it throws with perf_config::bounds_checking on, otherwise
    // it writes past the empty buffers.
    GraphLaplacian(MapParams const &map_params, bool allocate);

    // Log number of spanning trees of region `region_id` in the plan with
    // vertex labels `region_ids`. Hierarchical (sum over region intersect
    // county pieces plus the county level multigraph) when counties_on.
    double log_region_spanning_trees(PlanVector const &region_ids, int region_id);

    // The same for the region formed by merging regions region1_id and
    // region2_id, without building the merged plan.
    double log_merged_region_spanning_trees(PlanVector const &region_ids,
                                            int region1_id, int region2_id);

  private:
    // `in_region(v)` says whether vertex v is in the region. The public methods
    // build it as a lambda over the plan's region ids, so nothing here touches
    // the plan directly.
    template <typename InRegion>
    double log_spanning_trees(InRegion in_region);

    // One pass over the map for a non hierarchical region. Numbers the region's
    // vertices in ascending order: region_vertices[i] is the i-th one and
    // local_index[region_vertices[i]] = i. Entries past the returned count, and
    // local_index of vertices outside the region, are left as they were.
    // Returns the number of vertices in the region.
    template <typename InRegion>
    int scan_non_hierarchical_region(InRegion in_region);

    // One pass over the map for a hierarchical region. First zeroes
    // county_count for the previous call's touched_counties and clears it, then
    // for each region vertex v, in ascending order, in county c:
    //   - region_vertices[i] = v, for i = 0, ..., n - 1
    //   - local_index[v] = v's position within region intersect c (ascending)
    //   - county_count[c] = number of region vertices in c (zero if untouched)
    //   - touched_counties lists every county the region touches, in first
    //     seen order, and county_local[c] is c's position in it
    // Returns n, the number of vertices in the region.
    template <typename InRegion>
    int scan_hierarchical_region(InRegion in_region);

    // Groups the n region vertices from scan_hierarchical_region by county with
    // an O(n) counting sort: county c's piece is piece_vertices[county_offset[c]]
    // up to (not including) piece_vertices[county_offset[c] + county_count[c]],
    // in ascending order. Only needed when the region touches more than one
    // county; with one county the piece is just region_vertices.
    void group_region_by_county(int n);

    // Log number of spanning trees of the county level multigraph of the n
    // vertex region from scan_hierarchical_region. It has one node per touched
    // county c (node county_local[c]), and two nodes are joined by one unit of
    // edge weight per map edge between those counties inside the region. Zero
    // when the region touches only one county.
    template <typename InRegion>
    double county_level_log_spanning_trees(int n, InRegion in_region);

    // Log number of spanning trees of one connected piece of a region.
    //
    // The piece is vertices[start], ..., vertices[start + n - 1], in ascending
    // order (region_vertices for a whole region or a single county region, or
    // one county's block of piece_vertices), and local_index must already map
    // each of them to its position 0, ..., n - 1.
    // Edges come from `graph`, keeping those whose other end is in the region,
    // so every in region neighbor `graph` gives a piece vertex must itself be in
    // the piece:
    //   - map_graph, when the piece is a whole (non hierarchical) region
    //   - county_restricted_flat_graph, when the piece is region intersect county
    // Same vertex order and dropped vertex (n - 1) as build_whole_unit_log_taus,
    // so a whole county gives exactly its cached value.
    template <typename InRegion>
    double piece_log_spanning_trees(std::vector<int> const &vertices, int start, int n,
                                    FlatGraph const &graph, InRegion in_region);

    MapParams const &map_params;
    // Whether the buffers below were allocated. Declared before them since
    // their sizes depend on it.
    bool const allocated;
    // Whether tau is hierarchical: more than one county and plans assumed
    // hierarchically valid, the same rule as PlanMultigraph::counties_on.
    // Declared before the buffers since their sizes depend on it.
    bool const counties_on;

    std::vector<int> local_index;      // (V) a vertex's position within its piece
    std::vector<int> region_vertices;  // (V) region vertices in ascending order

    // only sized when allocated and counties_on, empty otherwise
    std::vector<int> piece_vertices;   // (V) region vertices grouped by county
    std::vector<int> county_count;     // (C) region vertices per county, zeroed for the
                                       //     previous region when a scan starts
    std::vector<int> county_offset;    // (C) start of each county's block in piece_vertices
    std::vector<int> county_local;     // (C) a touched county's node in the county multigraph
    std::vector<int> touched_counties; // (C capacity) counties the region touches, in first
                                       //     seen order

    LaplacianBuilder laplacian;
};

#endif