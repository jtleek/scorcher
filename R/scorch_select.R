#===============================================================================
# SCORCH SELECT
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Select a Dimension from a Scorch Model
#'
#' The `scorch_select` function adds a layer to the scorch model that selects
#' a specific dimension from the input tensor. This is useful for applying
#' operations or layers only to a particular dimension of the input tensor.
#'
#' @param scorch_model A scorch model object to which the dimension selection
#' layer will be added.
#'
#' @param dim An integer specifying the dimension to select from the input
#' tensor. This should be a valid dimension index based on the input tensor's
#' dimensions.
#'
#' @return The modified scorch model with the dimension selection layer
#' added to its architecture.
#'
#' @examples
#'
#' # Example usage:
#' scorch_model <- dl |>
#'   initiate_scorch() |>
#'   scorch_select(dim = 1)
#'
#' @export

scorch_select <- function(scorch_model, dim) {

  select_layer <- nn_module(

    classname = "scorch_select",

    initialize = function() {

      self$dim <- dim
    },

    forward = function(x) {

      ## Ensure Dimension is Within Bounds

      if (self$dim > length(dim(x))) {

        stop(paste("Dimension", self$dim,

          "out of bounds for input with dimensions:",

          paste(dim(x), collapse = ", ")))
      }

      ## Select Specified Dimension

      x <- x[, self$dim, drop = F]

      return(x)
    }
  )

  ## Add Select Layer to Scorch Architecture

  scorch_model$scorch_architecture <- append(

    scorch_model$scorch_architecture, list(select_layer(), type = "function")
  )

  return(scorch_model)
}

#=== END =======================================================================
