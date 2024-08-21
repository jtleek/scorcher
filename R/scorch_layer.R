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
#' @param layer_type A string specifying the type of layer to add (e.g.,
#' `"linear"`, `"conv2d"`). This string will be converted to the corresponding
#' torch function (e.g., nn_linear, nn_conv2d).
#'
#' @param in_features Optional. An integer specifying the number of input
#' features for the layer. Used for layers such as `nn_linear`.
#'
#' @param out_features Optional. An integer specifying the number of output
#' features for the layer. Used for layers such as `nn_linear`.
#'
#' @param ... Additional arguments passed to the layer function. These can
#' include other required or optional parameters depending on the layer type.
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

scorch_layer <- function(scorch_model, layer_type,

  in_features = NULL, out_features = NULL, ...) {

  function_name <- paste0("nn_", layer_type)

  nn_function <- get(function_name, envir = asNamespace("torch"))

  args_list <- list(...)

  if (!is.null(in_features))  args_list$in_features  <- in_features
  if (!is.null(out_features)) args_list$out_features <- out_features

  nn_obj <- do.call(nn_function, args_list)

  scorch_model$scorch_architecture <- append(

    scorch_model$scorch_architecture, list(nn_obj, type = "layer"))

  scorch_model
}

#=== END =======================================================================
