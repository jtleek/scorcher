#===============================================================================
# FUNCTION TO MARK OUTPUT NODES ON A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Mark Output Nodes on a Scorch Model
#'
#' @description
#' Declares which graph nodes should be treated as model outputs.
#' For single-output models, pass a single node name; for multi-output
#' models (e.g., multi-head architectures), pass a character vector
#' of node names.
#'
#' @param scorch_model A \code{scorch_model} object created by
#'   \code{\link{initiate_scorch}}.
#'
#' @param outputs A character vector of node names to designate as
#'   model outputs. These names must match existing nodes in the
#'   graph. For single-output models, pass one name. For multi-output
#'   models, pass multiple (e.g., \code{c("cls_head", "reg_head")}).
#'
#' @returns The updated \code{scorch_model} with its \code{outputs}
#'   field set to \code{outputs}.
#'
#' @details
#' The output nodes determine what the compiled model returns from
#' its forward pass. For a single output, the model returns one tensor.
#' For multiple outputs, it returns a list of tensors -- one per output
#' node, in the order specified here. When using multiple outputs,
#' \code{\link{compile_scorch}} expects a named list of loss functions
#' matching these output names.
#'
#' @examples
#' \dontrun{
#' # Single output
#' model <- model |>
#'   scorch_output("fc_out")
#'
#' # Multi-output (e.g., regression head + classification head)
#' model <- model |>
#'   scorch_output(c("reg_head", "cls_head"))
#' }
#'
#' @family model construction
#'
#' @export

scorch_output <- function(scorch_model, outputs) {

  #- Validate: outputs must be a character vector.

  if (!is.character(outputs)) {

    stop("`outputs` must be a character vector of node names.", call. = FALSE)
  }

  scorch_model$outputs <- outputs

  scorch_model
}

#=== END =======================================================================
