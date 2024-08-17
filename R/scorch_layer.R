#===============================================================================
# SCORCH LAYER
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Layer to a Scorch Model
#'
#' @description
#' This function adds a neural network layer to the scorch model architecture.
#'
#' @param scorch_model A scorch model object to which a layer is to be added.
#'
#' @param nn_obj A neural network object representing the layer to be added
#' to the scorch model.
#'
#' @return The updated scorch model with the new layer added to its
#' architecture.
#'
#' @export
#'
#' @examples
#'
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- scorch_create_dataloader(input, output, batch_size = 2)
#'
#' scorch_model <- dl |> initiate_scorch() |>
#'
#'   scorch_layer("linear", 11, 5)

scorch_layer <- function(scorch_model, nn_obj){

  scorch_model$scorch_architecture <- append(

    scorch_model$scorch_architecture, list(nn_obj, type = "layer"))

  scorch_model
}

#=== END =======================================================================
