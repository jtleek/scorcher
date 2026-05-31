#===============================================================================
# FUNCTION TO ADD A DROPOUT NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Dropout Node to a Scorch Model
#'
#' @description
#' Convenience wrapper that adds a \code{torch::nn_dropout} node to
#' the Scorch model graph.
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
#' @param inputs Character vector of upstream node names. If \code{NULL}
#'   (default), resolved automatically (last node or sole input).
#'
#' @param .name Optional node name. Can be a string or an unquoted name.
#'
#' @param .from Optional upstream node reference. Can be a string or an
#'   unquoted name.
#'
#' @param p Numeric. Dropout probability (default 0.5).
#'
#' @param ... Additional arguments passed to \code{torch::nn_dropout()}.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' This is equivalent to calling
#' \code{scorch_layer(model, dropout, .from = input, p = 0.5)} but
#' provides a more readable API for a common operation.
#'
#' @examples
#' \dontrun{
#' model <- model |>
#'   scorch_layer(linear, in_features = 32, out_features = 16) |>
#'   scorch_layer(relu) |>
#'   scorch_dropout(p = 0.3)
#' }
#'
#' @family model construction
#'
#' @export

scorch_dropout <- function(scorch_model,
                           name,
                           inputs = NULL,
                           p = 0.5,
                           .name = NULL,
                           .from = NULL,
                           ...) {

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
      scorch_parse_refs_expr(from_expr, arg = ".from")
  )

  node_name <- scorch_prepare_node_name(
    scorch_model,
    explicit_expr = name_expr,
    legacy_expr = legacy_name_expr,
    auto_prefix = "dropout"
  )
  scorch_model <- node_name$model
  name <- node_name$name

  do_mod <- torch::nn_dropout(p = p, ...)

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = do_mod,
    inputs = inputs,
    node_type = "layer",
    constructor = "dropout",
    args = c(list(p = p), list(...)),
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
