// graph_laplacian.cpp
#include "graph_laplacian.h"

#include <stdexcept>


GraphLaplacian::GraphLaplacian(MapParams const &map_params, bool const allocate)
    : map_params(map_params),
      allocated(allocate),
      counties_on(map_params.num_counties > 1 && !map_params.plans_may_be_non_hierarchical),
      local_index(allocated ? map_params.V : 0),
      region_vertices(allocated ? map_params.V : 0),
      piece_vertices(allocated && counties_on ? map_params.V : 0),
      county_count(allocated && counties_on ? map_params.num_counties : 0, 0),
      county_offset(allocated && counties_on ? map_params.num_counties : 0),
      county_local(allocated && counties_on ? map_params.num_counties : 0),
      // skips the builder's dense buffer too when not allocating
      laplacian(allocated) {
    if (allocated && counties_on) touched_counties.reserve(map_params.num_counties);
}


template <typename InRegion>
int GraphLaplacian::scan_non_hierarchical_region(InRegion in_region) {
    int n = 0;
    for (int v = 0; v < map_params.V; ++v) {
        if (in_region(v)) {
            region_vertices[n] = v;
            local_index[v] = n;
            ++n;
        }
    }
    return n;
}


template <typename InRegion>
int GraphLaplacian::scan_hierarchical_region(InRegion in_region) {
    // clear the previous call's counts here rather than at the end of a call,
    // so a call that threw partway through cannot leave stale counts behind
    for (int const c : touched_counties) county_count[c] = 0;
    touched_counties.clear();

    int n = 0;
    for (int v = 0; v < map_params.V; ++v) {
        if (in_region(v)) {
            int const c = static_cast<int>(map_params.counties[v]) - 1;
            // first vertex seen in this county
            if (county_count[c] == 0) {
                county_local[c] = static_cast<int>(touched_counties.size());
                touched_counties.push_back(c);
            }
            local_index[v] = county_count[c]++;
            region_vertices[n++] = v;
        }
    }
    return n;
}


void GraphLaplacian::group_region_by_county(int const n) {
    // each touched county's block starts where the previous one ended
    int offset = 0;
    for (int const c : touched_counties) {
        county_offset[c] = offset;
        offset += county_count[c];
    }
    // local_index already gives each vertex its position inside its county's
    // block, so every block comes out in ascending order
    for (int i = 0; i < n; ++i) {
        int const v = region_vertices[i];
        int const c = static_cast<int>(map_params.counties[v]) - 1;
        piece_vertices[county_offset[c] + local_index[v]] = v;
    }
}


template <typename InRegion>
double GraphLaplacian::piece_log_spanning_trees(std::vector<int> const &vertices,
                                                int const start, int const n,
                                                FlatGraph const &graph,
                                                InRegion in_region) {
    if constexpr (perf_config::bounds_checking) {
        if (start < 0 || n < 0 || start + n > static_cast<int>(vertices.size())) {
            throw std::out_of_range(
                "GraphLaplacian::piece_log_spanning_trees: the piece runs past the end "
                "of `vertices`.");
        }
    }
    // a single vertex has exactly one spanning tree
    if (n <= 1) return 0.0;

    laplacian.reset(n, n - 1);
    for (int i = 0; i < n; ++i) {
        for (int const nbor : graph.neighbors(vertices[start + i])) {
            // in_region must come first, since local_index is stale for
            // vertices outside the region. Adding only when the neighbor comes
            // later in the piece adds each edge once
            if (in_region(nbor) && local_index[nbor] > i) {
                laplacian.add_edge(i, local_index[nbor]);
            }
        }
    }
    return laplacian.log_det();
}


template <typename InRegion>
double GraphLaplacian::county_level_log_spanning_trees(int const n, InRegion in_region) {
    int const k = static_cast<int>(touched_counties.size());
    // a one node multigraph has exactly one spanning tree
    if (k <= 1) return 0.0;

    laplacian.reset(k, k - 1);
    for (int i = 0; i < n; ++i) {
        int const v = region_vertices[i];
        int const v_node = county_local[static_cast<int>(map_params.counties[v]) - 1];
        // only v's neighbors in other counties, so every edge seen here is
        // between two different county nodes
        for (int const nbor : map_params.cross_county_flat_graph.neighbors(v)) {
            // in_region must come first, since county_local is stale for
            // counties the region does not touch
            if (!in_region(nbor)) continue;
            int const nbor_node =
                county_local[static_cast<int>(map_params.counties[nbor]) - 1];
            // each edge is seen from both ends, so add it once, from the side
            // whose county node comes first. Repeated edges add up to the count
            if (v_node < nbor_node) {
                laplacian.add_edge(v_node, nbor_node);
            }
        }
    }
    return laplacian.log_det();
}


template <typename InRegion>
double GraphLaplacian::log_spanning_trees(InRegion in_region) {
    // both public methods come through here, so this one check covers them.
    // Without it an unallocated object would write past its empty buffers
    if constexpr (perf_config::bounds_checking) {
        if (!allocated) {
            throw std::runtime_error(
                "GraphLaplacian: tau requested from an object constructed with "
                "allocate = false.");
        }
    }

    // counties off: the whole region is one piece on the full map graph
    if (!counties_on) {
        int const n = scan_non_hierarchical_region(in_region);
        return piece_log_spanning_trees(region_vertices, 0, n, map_params.map_graph,
                                        in_region);
    }

    int const n = scan_hierarchical_region(in_region);
    // log tau of each whole county, used whenever the region contains all of it
    std::vector<double> const &whole_county_log_tau = map_params.map_log_taus.by_level[1];

    // the region lies inside one county, so it is a single piece and the
    // county level multigraph has one node (log tau 0)
    if (touched_counties.size() == 1) {
        int const c = touched_counties[0];
        if (n == static_cast<int>(map_params.county_vertices[c].size())) {
            return whole_county_log_tau[c];
        }
        return piece_log_spanning_trees(region_vertices, 0, n,
                                        map_params.county_restricted_flat_graph, in_region);
    }

    // the region spans several counties: the county level term plus one term
    // per region intersect county piece
    double log_tau = county_level_log_spanning_trees(n, in_region);
    // grouping only gives partial counties their vertices as one contiguous
    // run, so it waits for the first partial county and is skipped entirely
    // when every touched county is whole
    bool grouped = false;
    for (int const c : touched_counties) {
        int const count = county_count[c];
        // if the entire county is in the region just use the precomputed value
        if (count == static_cast<int>(map_params.county_vertices[c].size())) {
            log_tau += whole_county_log_tau[c];
        } else {
            if (!grouped) {
                group_region_by_county(n);
                grouped = true;
            }
            log_tau += piece_log_spanning_trees(piece_vertices, county_offset[c], count,
                                                map_params.county_restricted_flat_graph,
                                                in_region);
        }
    }
    return log_tau;
}


double GraphLaplacian::log_region_spanning_trees(PlanVector const &region_ids,
                                                 int const region_id) {
    return log_spanning_trees(
        [&region_ids, region_id](int const v) { return region_ids[v] == region_id; });
}


double GraphLaplacian::log_merged_region_spanning_trees(PlanVector const &region_ids,
                                                        int const region1_id,
                                                        int const region2_id) {
    return log_spanning_trees([&region_ids, region1_id, region2_id](int const v) {
        return region_ids[v] == region1_id || region_ids[v] == region2_id;
    });
}
