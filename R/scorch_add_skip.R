#===============================================================================
# FUNCTION TO ADD A SKIP CONNECTION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Skip Connection Node to a Scorch Model
#'
#' @description
#' Creates a node that performs element-wise addition of two upstream
#' tensors. Used to implement residual / skip connections where the
#' output is \code{x + skip}.
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
#' @param inputs A length-2 character vector: \code{c("main_path",
#'   "skip_path")}. Both upstream nodes must produce tensors of the
#'   same shape.
#'
#' @param .name Optional node name. Can be a string or an unquoted name.
#'
#' @param .from Optional length-2 upstream node references. Can be a character
#'   vector, unquoted names, or \code{c(...)} of unquoted names.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' The node is implemented as a lightweight \code{torch::nn_module}
#' that sums its two inputs. It has no learnable parameters. This
#' replaces the old \code{use_residual} argument in
#' \code{\link{scorch_layer}}.
#'
#' @examples
#' \dontrun{
#' # Residual connection around a linear + relu block
#' model <- model |>
#'   scorch_layer(linear, .from = x,
#'                in_features = 32, out_features = 32,
#'                .name = update) |>
#'   scorch_layer(relu, .name = activated) |>
#'   scorch_add_skip(.from = c(activated, x), .name = residual)
#' }
#'
#' @family model construction
#'
#' @export

scorch_add_skip <- function(scorch_model,
                            name,
                            inputs = NULL,
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

  if (length(inputs) != 2) {

    stop("Skip connections require exactly two input nodes.", call. = FALSE)
  }

  node_name <- scorch_prepare_node_name(
    scorch_model,
    explicit_expr = name_expr,
    legacy_expr = legacy_name_expr,
    auto_prefix = "skip"
  )
  scorch_model <- node_name$model
  name <- node_name$name

  #- Build a lightweight module that sums its two inputs.

  skip_mod <- torch::nn_module(

    initialize = function() {},

    forward = function(x, skip) {

      x + skip
    }
  )()

  #- Append to graph.

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = skip_mod,
    inputs = inputs,
    node_type = "function",
    constructor = "add_skip",
    args = list(),
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
