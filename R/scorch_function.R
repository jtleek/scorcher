#' Add a Custom Function to a Scorch Model
#'
#' @description
#' This function adds a custom function to the scorch model architecture.
#'
#' @param scorch_model A scorch model object to which a custom function
#' is to be added.
#'
#' @param func A function to be added to the scorch model.
#'
#' @param ... Additional arguments to be passed to the custom function.
#'
#' @return The updated scorch model with the custom function added to its
#' architecture.
#'
#' @export
#'
#' @examples
#'
#' custom_function <- function(x, factor) { x * factor }
#'
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- create_dataloader(input, output, batch_size = 2)
#'
#' scorch_model = dl |> initiate_scorch() |>
#'
#'   scorch_layer(torch::nn_linear(11, 5)) |>
#'
#'   scorch_function(custom_function, factor = 2)

scorch_function = function(scorch_model, func, ...) {

  func_call = function(x) {

    return(func(x, ...))
  }

  scorch_model$scorch_architecture = append(scorch_model$scorch_architecture,

    list(func_call, type = "function"))

  scorch_model
}
