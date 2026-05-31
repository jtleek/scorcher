#===============================================================================
# GRAPH SPECIFICATION AND VALIDATION
#===============================================================================

#' Return the Declarative Scorch Graph Specification
#'
#' @param scorch_model A \code{scorch_model} object.
#'
#' @returns A tibble describing the graph without the live torch module objects.
#'
#' @family model construction
#'
#' @export
scorch_spec <- function(scorch_model) {
  scorch_model <- scorch_check_model(scorch_model)
  graph <- scorch_model$graph

  graph[c("name", "inputs", "node_type", "constructor", "args",
          "explicit_name", "param_count", "trainable", "output_shape")]
}

#' Validate a Scorch Model Graph
#'
#' @param scorch_model A \code{scorch_model} object.
#' @param strict Logical. If \code{TRUE}, validation issues throw an error. If
#'   \code{FALSE}, a tibble of issues is returned.
#'
#' @returns Invisibly returns \code{TRUE} when strict validation succeeds, or a
#'   tibble of issues when \code{strict = FALSE}.
#'
#' @family model construction
#'
#' @export
validate_scorch_graph <- function(scorch_model, strict = TRUE) {
  scorch_model <- scorch_check_model(scorch_model)
  graph <- scorch_model$graph
  issues <- list()

  add_issue <- function(level, location, message) {
    issues[[length(issues) + 1L]] <<- list(
      level = level,
      location = location,
      message = message
    )
  }

  if (length(scorch_model$inputs) == 0) {
    add_issue("error", "inputs", "Model has no inputs.")
  }

  if (nrow(graph) == 0) {
    add_issue("error", "graph", "Model has no graph nodes.")
  }

  if (length(scorch_model$outputs) == 0) {
    add_issue("error", "outputs", "Model has no outputs.")
  }

  duplicated_inputs <- unique(scorch_model$inputs[duplicated(scorch_model$inputs)])
  if (length(duplicated_inputs) > 0) {
    add_issue("error", "inputs",
              paste("Duplicate input names:",
                    paste(duplicated_inputs, collapse = ", ")))
  }

  duplicated_nodes <- unique(graph$name[duplicated(graph$name)])
  if (length(duplicated_nodes) > 0) {
    add_issue("error", "graph",
              paste("Duplicate node names:",
                    paste(duplicated_nodes, collapse = ", ")))
  }

  bad_outputs <- setdiff(scorch_model$outputs, graph$name)
  if (length(bad_outputs) > 0) {
    add_issue("error", "outputs",
              paste("Output node(s) not found:",
                    paste(bad_outputs, collapse = ", ")))
  }

  seen <- scorch_model$inputs
  for (i in seq_len(nrow(graph))) {
    node_inputs <- graph$inputs[[i]]
    missing_inputs <- setdiff(node_inputs, seen)
    if (length(missing_inputs) > 0) {
      add_issue(
        "error",
        graph$name[i],
        paste("Missing or out-of-order input(s):",
              paste(missing_inputs, collapse = ", "))
      )
    }
    seen <- c(seen, graph$name[i])
  }

  issue_tbl <- if (length(issues) == 0) {
    tibble::tibble(level = character(), location = character(),
                   message = character())
  } else {
    tibble::tibble(
      level = vapply(issues, `[[`, character(1), "level"),
      location = vapply(issues, `[[`, character(1), "location"),
      message = vapply(issues, `[[`, character(1), "message")
    )
  }

  if (strict && nrow(issue_tbl) > 0) {
    stop(paste(issue_tbl$message, collapse = "\n"), call. = FALSE)
  }

  if (strict) invisible(TRUE) else issue_tbl
}

#' Convert a Scorch Model to a Torch Module
#'
#' @param scorch_model A \code{scorch_model} object.
#'
#' @returns A \code{torch::nn_module} instance.
#'
#' @family model construction
#'
#' @export
as_torch_module <- function(scorch_model) {
  scorch_model <- scorch_check_model(scorch_model)
  validate_scorch_graph(scorch_model)

  if (isTRUE(scorch_model$compiled) && !is.null(scorch_model$nn_model)) {
    return(scorch_model$nn_model)
  }

  mod <- scorch_build_module(
    graph = scorch_model$graph,
    inputs = scorch_model$inputs,
    outputs = scorch_model$outputs
  )

  mod()
}

