#===============================================================================
# FUNCTION TO ADD A CUSTOM FUNCTION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Custom Function Node to a Scorch Model
#'
#' @description
#' Wraps an arbitrary R function as a node in the Scorch computation
#' graph. This is useful for operations that are not standard torch
#' layers but are needed in the forward pass (e.g., reshaping,
#' scaling, custom activations).
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
#' @param func A function to apply during the forward pass. It will
#'   receive the upstream tensor(s) as its first argument(s), followed
#'   by any additional arguments captured via \code{...}.
#'
#' @param inputs Character vector of upstream node names. If \code{NULL}
#'   (default), resolved automatically (last node or sole input).
#'
#' @param .name Optional node name. Can be a string or an unquoted name.
#'
#' @param .from Optional upstream node reference or references. Can be a
#'   string, an unquoted name, or \code{c(...)} of unquoted names.
#'
#' @param ... Additional arguments passed to \code{func} at forward
#'   time. These are captured at model-build time and baked into the
#'   node.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' The function is wrapped in a lightweight \code{torch::nn_module} so
#' it can participate in the graph alongside standard layers. Any extra
#' arguments in \code{...} are captured at build time and passed to
#' \code{func} on every forward call.
#'
#' Unlike \code{\link{scorch_layer}}, which instantiates a torch
#' \code{nn_module} constructor, \code{scorch_function} accepts plain
#' R functions. Use this for stateless operations that do not have
#' learnable parameters.
#'
#' @examples
#' \dontrun{
#' # Scale a tensor by a constant
#' model <- model |>
#'   scorch_function("scale", function(x, factor) x * factor,
#'                   factor = 0.1)
#'
#' # Reshape before a linear layer
#' model <- model |>
#'   scorch_function("reshape", function(x) x$view(c(x$size(1), -1)))
#' }
#'
#' @family model construction
#'
#' @export

scorch_function <- function(scorch_model,
                            name,
                            func = NULL,
                            inputs = NULL,
                            .name = NULL,
                            .from = NULL,
                            ...) {

  scorch_model <- scorch_check_model(scorch_model)

  name_expr <- if (missing(.name)) NULL else substitute(.name)
  from_expr <- if (missing(.from)) NULL else substitute(.from)
  inputs_expr <- if (missing(inputs)) NULL else substitute(inputs)

  if (missing(func) || is.null(func)) {
    func <- name
    legacy_name_expr <- NULL
  } else {
    legacy_name_expr <- substitute(name)
  }

  if (!is.function(func)) {
    stop("`func` must be a function.", call. = FALSE)
  }

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
    auto_prefix = "function"
  )
  scorch_model <- node_name$model
  name <- node_name$name

  #- Capture extra arguments to bake into the forward pass.

  extra_args <- list(...)

  #- Wrap the function in a lightweight nn_module.

  func_mod <- torch::nn_module(

    initialize = function() {},

    forward = function(...) {

      do.call(func, c(list(...), extra_args))
    }
  )()

  #- Append to graph.

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = func_mod,
    inputs = inputs,
    node_type = "function",
    constructor = "function",
    args = extra_args,
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
