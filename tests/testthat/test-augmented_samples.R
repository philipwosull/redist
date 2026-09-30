# Tests for `return_augmented_samples` in `redist_smc`, which stores the
# spanning forests and linking edges of the final plans for restarting sampling.

iowa_map <- suppressMessages(
    redist_map(iowa, existing_plan = cd_2010, pop_tol = 0.05)
)
iowa_edges <- graph_edge_list(get_adj(iowa_map))

# checks a column of the forest matrix is a spanning forest of `plan`
expect_spanning_forest <- function(forest, plan, edges) {
    V <- length(plan)
    nregions <- length(unique(plan))
    forest_edges <- edges[forest, , drop = FALSE]
    expect_equal(nrow(forest_edges), V - nregions)
    expect_true(all(plan[forest_edges[, 1]] == plan[forest_edges[, 2]]))
    comps <- igraph::components(igraph::make_graph(
        t(forest_edges), n = V, directed = FALSE
    ))$membership
    expect_equal(max(comps), nregions)
    expect_true(all(tapply(comps, plan, dplyr::n_distinct) == 1))
}

# checks a plan's linking edges join its regions into a tree
expect_linking_edge_tree <- function(linking_edges, plan, edges) {
    nregions <- length(unique(plan))
    region1 <- plan[linking_edges$vertex1 + 1]
    region2 <- plan[linking_edges$vertex2 + 1]
    expect_equal(nrow(linking_edges), nregions - 1)
    expect_true(all(region1 != region2))
    expect_equal(
        unname(edges[linking_edges$edge_id, , drop = FALSE]),
        unname(cbind(
            pmin(linking_edges$vertex1, linking_edges$vertex2),
            pmax(linking_edges$vertex1, linking_edges$vertex2)
        ) + 1L)
    )
    expect_true(igraph::is_connected(
        igraph::make_graph(rbind(region1, region2), n = nregions, directed = FALSE)
    ))
}

test_that("augmented samples are not stored by default", {
    plans <- redist_smc(iowa_map, 10, sampling_space = "spanning_forest", silent = TRUE)
    expect_null(attr(plans, "augmented_samples"))
})

test_that("augmented samples are not available in graph space", {
    expect_error(
        redist_smc(iowa_map, 10, sampling_space = "graph_plan", silent = TRUE,
                   control = list(return_augmented_samples = TRUE)),
        "not available for graph space"
    )
})

test_that("spanning forest space returns forests aligned with the plans", {
    skip_if_not_installed("igraph")
    plans <- redist_smc(
        iowa_map, 20, runs = 2, sampling_space = "spanning_forest", silent = TRUE,
        control = list(return_augmented_samples = TRUE)
    )
    aug <- attr(plans, "augmented_samples")

    expect_equal(aug$edge_list, iowa_edges)
    expect_null(aug$linking_edges)
    expect_true(is.logical(aug$spanning_forests))
    # indexed by the sampled draws, so the reference plan has no column
    expect_equal(ncol(get_plans_matrix(plans)), 41)
    expect_equal(dim(aug$spanning_forests), c(nrow(iowa_edges), 40))

    sampled <- subset_sampled(plans)
    sampled_aug <- attr(sampled, "augmented_samples")
    expect_equal(sampled_aug, aug)
    plans_m <- get_plans_matrix(sampled)
    for (i in seq_len(ncol(plans_m))) {
        expect_spanning_forest(sampled_aug$spanning_forests[, i], plans_m[, i], iowa_edges)
    }
})

test_that("linking edge space returns linking edges aligned with the plans", {
    skip_if_not_installed("igraph")
    plans <- redist_smc(
        iowa_map, 20, runs = 2, sampling_space = "linking_edge", silent = TRUE,
        ms_params = list(frequency = 1, mh_accept_per_smc = 2),
        control = list(return_augmented_samples = TRUE)
    )
    sampled <- subset_sampled(plans)
    aug <- attr(sampled, "augmented_samples")
    plans_m <- get_plans_matrix(sampled)

    expect_equal(dim(aug$spanning_forests), c(nrow(iowa_edges), 40))
    expect_named(aug$linking_edges, c("draw", "edge_id", "vertex1", "vertex2", "log_prob"))
    expect_equal(sort(unique(aug$linking_edges$draw)), seq_len(40))
    for (i in seq_len(ncol(plans_m))) {
        expect_spanning_forest(aug$spanning_forests[, i], plans_m[, i], iowa_edges)
        expect_linking_edge_tree(
            aug$linking_edges[aug$linking_edges$draw == i, ], plans_m[, i], iowa_edges
        )
    }
})

test_that("augmented samples stay aligned when plans are filtered or combined", {
    plans <- redist_smc(
        iowa_map, 10, sampling_space = "linking_edge", silent = TRUE,
        control = list(return_augmented_samples = TRUE)
    )
    aug <- attr(plans, "augmented_samples")

    # the reference plan and sampled draws 3 and 6
    filtered <- dplyr::filter(plans, draw %in% c("cd_2010", "3", "6"))
    filtered_aug <- attr(filtered, "augmented_samples")
    expect_equal(ncol(get_plans_matrix(filtered)), 3)
    expect_equal(filtered_aug$spanning_forests, aug$spanning_forests[, c(3, 6)])
    expected_linking_edges <- aug$linking_edges[aug$linking_edges$draw %in% c(3, 6), ]
    expected_linking_edges$draw <- match(expected_linking_edges$draw, c(3, 6))
    rownames(expected_linking_edges) <- NULL
    expect_equal(filtered_aug$linking_edges, expected_linking_edges)

    # the reference plan alone has no sampled draws
    ref_aug <- attr(subset_ref(plans), "augmented_samples")
    expect_equal(ncol(ref_aug$spanning_forests), 0)
    expect_equal(nrow(ref_aug$linking_edges), 0)

    # combining appends the sampled draws in order
    combined_aug <- attr(rbind(plans, plans), "augmented_samples")
    expect_equal(combined_aug$spanning_forests, cbind(aug$spanning_forests, aug$spanning_forests))
    shifted_linking_edges <- aug$linking_edges
    shifted_linking_edges$draw <- shifted_linking_edges$draw + 10L
    expected_linking_edges <- rbind(aug$linking_edges, shifted_linking_edges)
    rownames(expected_linking_edges) <- NULL
    expect_equal(combined_aug$linking_edges, expected_linking_edges)

    # unless some set of plans has none
    no_aug <- redist_smc(iowa_map, 10, sampling_space = "linking_edge", silent = TRUE)
    expect_null(attr(rbind(plans, no_aug), "augmented_samples"))
})

test_that("partial plans return a linking edge per region pair", {
    skip_if_not_installed("igraph")
    plans <- redist_smc(
        iowa_map, 10, n_steps = 2, sampling_space = "linking_edge", silent = TRUE,
        control = list(return_augmented_samples = TRUE)
    )
    aug <- attr(plans, "augmented_samples")
    plans_m <- get_plans_matrix(plans)
    for (i in seq_len(ncol(plans_m))) {
        expect_spanning_forest(aug$spanning_forests[, i], plans_m[, i], iowa_edges)
        expect_linking_edge_tree(
            aug$linking_edges[aug$linking_edges$draw == i, ], plans_m[, i], iowa_edges
        )
    }
})

test_that("full diagnostics without a final resample have no resample entry", {
    plans <- redist_smc(
        iowa_map, 10, sampling_space = "linking_edge", silent = TRUE,
        diagnostics = "all", resample = FALSE
    )
    diag <- attr(plans, "internal_diagnostics")[[1]]
    step_names <- c("smc_step1", "smc_step2", "smc_step3")
    expect_named(diag$region_ids_mat_list, step_names)
    expect_named(diag$forest_adjs_list, step_names)
    expect_named(diag$linking_edges_list, step_names)
})

test_that("full diagnostics store named per step forests and linking edges", {
    skip_if_not_installed("igraph")
    plans <- redist_smc(
        iowa_map, 10, sampling_space = "linking_edge", silent = TRUE,
        diagnostics = "all", ms_params = list(frequency = 1, mh_accept_per_smc = 1),
        control = list(return_augmented_samples = TRUE)
    )
    diag <- attr(plans, "internal_diagnostics")[[1]]
    step_names <- c(
        "smc_step1", "ms_step1", "smc_step2", "ms_step2", "smc_step3", "ms_step3",
        "resample"
    )

    # the resample entry is the final plans
    aug <- attr(plans, "augmented_samples")
    expect_equal(
        unname(diag$region_ids_mat_list$resample),
        unname(get_plans_matrix(subset_sampled(plans)))
    )
    expect_equal(diag$forest_adjs_list$resample, aug$spanning_forests)
    expect_equal(diag$linking_edges_list$resample, aug$linking_edges)

    expect_named(diag$region_ids_mat_list, step_names)
    expect_named(diag$forest_adjs_list, step_names)
    expect_named(diag$linking_edges_list, step_names)
    for (step in step_names) {
        step_plans <- diag$region_ids_mat_list[[step]]
        for (i in seq_len(ncol(step_plans))) {
            expect_spanning_forest(
                diag$forest_adjs_list[[step]][, i], step_plans[, i], iowa_edges
            )
            step_linking_edges <- diag$linking_edges_list[[step]]
            expect_linking_edge_tree(
                step_linking_edges[step_linking_edges$draw == i, ], step_plans[, i], iowa_edges
            )
        }
    }
})
