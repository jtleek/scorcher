#===============================================================================
# FUNCTION TO ADD INPUT NODES TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add an Input Node to a Scorch Model
#'
#' @description
#' Registers a named input node on a Scorch model. Each input node
#' corresponds to one tensor that will be passed to the model at
#' training or inference time. For single-input models, one call is
#' needed; for multi-input models (e.g., late fusion), call once per
#' input stream.
#'
#' @param scorch_model A \code{scorch_model} object created by
#'   \code{\link{initiate_scorch}}.
#'
#' @param name A single character string giving a unique name for this
#'   input. Downstream layers reference this name via their \code{inputs}
#'   argument.
#'
#' @returns The updated \code{scorch_model} with \code{name} appended to
#'   its \code{inputs} field.
#'
#' @details
#' Input names must be unique within a model. Attempting to add a
#' duplicate name will raise an error. The names declared here are
#' used by \code{\link{compile_scorch}} to wire the forward pass:
#' tensors are matched to input nodes by name (for multiple inputs)
#' or by position (for a single input).
#'
#' @examples
#' \dontrun{
#' # Single input
#' model <- initiate_scorch() |>
#'   scorch_input("features")
#'
#' # Multi-input (e.g., two feature streams for late fusion)
#' model <- initiate_scorch() |>
#'   scorch_input("stream_a") |>
#'   scorch_input("stream_b")
#' }
#'
#' @family model construction
#'
#' @export

scorch_input <- function(scorch_model, name) {

  #- Validate: no duplicate input names.

  if (name %in% scorch_model$inputs) {

    stop("Input '", name, "' already exists.", call. = FALSE)
  }

  scorch_model$inputs <- c(scorch_model$inputs, name)

  scorch_model
}

#=== END =======================================================================
