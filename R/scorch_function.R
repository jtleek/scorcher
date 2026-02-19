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
#' @param name A single character string giving a unique name for this
#'   node in the computation graph.
#'
#' @param func A function to apply during the forward pass. It will
#'   receive the upstream tensor(s) as its first argument(s), followed
#'   by any additional arguments captured via \code{...}.
#'
#' @param inputs Character vector of upstream node names. If \code{NULL}
#'   (default), resolved automatically (last node or sole input).
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
                            func,
                            inputs = NULL,
                            ...) {

  #- Capture extra arguments to bake into the forward pass.

  extra_args <- list(...)

  #- Wrap the function in a lightweight nn_module.

  func_mod <- torch::nn_module(

    initialize = function() {},

    forward = function(...) {

      do.call(func, c(list(...), extra_args))
    }
  )()

  #- Resolve inputs when not specified explicitly.

  if (is.null(inputs)) {

    if (nrow(scorch_model$graph) == 0) {

      if (length(scorch_model$inputs) != 1) {

        stop("Must specify 'inputs' when multiple inputs exist.",
             call. = FALSE)
      }

      inputs <- scorch_model$inputs

    } else {

      inputs <- tail(scorch_model$graph$name, 1)
    }
  }

  #- Append to graph.

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name    = name,
    module  = list(func_mod),
    inputs  = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
