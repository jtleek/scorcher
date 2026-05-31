#===============================================================================
# OPTIONAL LUZ INTEGRATION
#===============================================================================

#' Convert a Scorch Model to a Torch Module Generator
#'
#' @param scorch_model A \code{scorch_model} object.
#'
#' @param instantiate Logical. If \code{FALSE} (default), return a
#'   \code{torch::nn_module} generator suitable for \pkg{luz}. If
#'   \code{TRUE}, return an instantiated module.
#'
#' @returns A \code{torch::nn_module} generator, or an instantiated module when
#'   \code{instantiate = TRUE}.
#'
#' @family model training
#'
#' @export
as_torch <- function(scorch_model, instantiate = FALSE) {
  scorch_model <- scorch_check_model(scorch_model)
  validate_scorch_graph(scorch_model)

  module <- scorch_build_module(
    graph = scorch_model$graph,
    inputs = scorch_model$inputs,
    outputs = scorch_model$outputs
  )

  if (isTRUE(instantiate)) module() else module
}

#' Convert a Scorch Model to a Luz-Compatible Module
#'
#' @param scorch_model A \code{scorch_model} object.
#'
#' @returns A \code{torch::nn_module} generator suitable for \pkg{luz}.
#'
#' @family model training
#'
#' @export
as_luz_module <- function(scorch_model) {
  as_torch(scorch_model, instantiate = FALSE)
}
