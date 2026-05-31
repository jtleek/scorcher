#===============================================================================
# FUNCTION TO ADD A CONCATENATION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Concatenation Node to a Scorch Model
#'
#' @description
#' Concatenates the outputs of two or more upstream nodes along a
#' specified dimension. Commonly used to merge parallel branches
#' in multi-input or late-fusion architectures.
#'
#' @param scorch_model A \code{scorch_model} object created by
#'   \code{\link{initiate_scorch}}.
#'
#' @param name A unique character string identifying this node in the
#'   model graph. Names wire the computation graph -- other nodes
#'   reference them via their \code{inputs} argument to define
#'   branching, fusion, and skip connections. Names are arbitrary but
#'   appear in error messages and \code{\link{plot_scorch_model}}
#'   output. Common prefixes: \code{"fc"} (linear), \code{"conv"}
#'   (convolution), \code{"act"} (activation). Use number suffixes
#'   for multiples (e.g., \code{"fc1"}, \code{"fc2"}).
#'
#' @param inputs Character vector of two or more upstream node names
#'   whose outputs will be concatenated.
#'
#' @param dim Integer. Dimension along which to concatenate (default 2,
#'   the feature dimension). Dim 1 is the batch dimension in R torch
#'   (1-indexed).
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' The node is implemented as a lightweight \code{torch::nn_module}
#' that calls \code{torch::torch_cat()} on its inputs. It has no
#' learnable parameters.
#'
#' @examples
#' \dontrun{
#' # Merge two branches for late fusion
#' model <- model |>
#'   scorch_concat("merged", inputs = c("branch_a", "branch_b"), dim = 2)
#' }
#'
#' @family model construction
#'
#' @export

scorch_concat <- function(scorch_model,
                          name,
                          inputs = NULL,
                          dim = 2,
                          .name = NULL,
                          .from = NULL) {

  scorch_model <- scorch_check_model(scorch_model)

  name_expr <- if (missing(.name)) NULL else substitute(.name)
  legacy_name_expr <- if (missing(name)) NULL else substitute(name)
  from_expr <- if (missing(.from)) NULL else substitute(.from)
  inputs_expr <- if (missing(inputs)) NULL else substitute(inputs)

  inputs <- scorch_resolve_inputs(
    scorch_model,
    inputs = if (is.null(inputs_expr)) NULL else
      scorch_parse_refs_expr(inputs_expr, arg = "inputs"),
    from = if (is.null(from_expr)) NULL else
      scorch_parse_refs_expr(from_expr, arg = ".from"),
    allow_multi = TRUE
  )

  if (length(inputs) < 2) {
    stop("`scorch_concat()` requires two or more input nodes.", call. = FALSE)
  }

  node_name <- scorch_prepare_node_name(
    scorch_model,
    explicit_expr = name_expr,
    legacy_expr = legacy_name_expr,
    auto_prefix = "concat"
  )
  scorch_model <- node_name$model
  name <- node_name$name

  if (name %in% scorch_model$graph$name || name %in% scorch_model$inputs) {
    stop("Node name '", name, "' already exists in the model graph.",
         call. = FALSE)
  }

  #- Build a lightweight module that concatenates its inputs.

  concat_mod <- torch::nn_module(

    initialize = function() {},

    forward = function(...) {

      torch::torch_cat(list(...), dim = dim)
    }
  )()

  #- Append to graph.

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = concat_mod,
    inputs = inputs,
    node_type = "function",
    constructor = "concat",
    args = list(dim = dim),
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
