#===============================================================================
# FUNCTIONS TO INITIATE A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Initiate a Scorch Model
#'
#' @param dl An input data loader, created with scorch_create_dataloader
#'
#' @return A scorch model object
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
#' dl |> initiate_scorch()

initiate_scorch <- function(dl) {

  l <- list(dl = dl, scorch_architecture = list())

  create_scorch_model_class(l)
}

#=== HELPERS ===================================================================

#--- SCORCH MODEL CLASS --------------------------------------------------------

#' Create a Scorch Model Class
#'
#' @description
#' #' This function creates an object of class 'scorch_model'.
#'
#' @param obj An object to be converted to a scorch model.
#'
#' @return The input object with the class attribute set to 'scorch_model'.
#'
#' @export
#'
#' @examples
#'
#' scorch_model <- create_scorch_model_class(list(scorch_architecture = list()))
#'
#' class(scorch_model)

create_scorch_model_class <- function(obj) {

  structure(obj, class = "scorch_model")
}

#--- PRINT METHOD --------------------------------------------------------------

#' Print Method for Scorch Model
#'
#' @description
#' This function defines the print method for objects of class 'scorch_model'.
#'
#' @param x An object of class 'scorch_model'.
#'
#' @param ... Additional arguments to be passed to the print function.
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
#'   scorch_layer(torch::nn_linear(11, 5))
#'
#' print(scorch_model)

print.scorch_model = function(x, ...) {

  cat("This scorch model has a dataloader object with features: \n\n")

  print(x$dl)

  cat("\n\n and model architecture:\n\n")

  n_layer = length(x$scorch_architecture) / 2

  if (n_layer == 0) {

    cat(" * No layers\n\n")

  } else {

    for(i in 1:n_layer) {

      cat(glue::glue(" * Layer {i} is a ",

        "{crayon::red(class(x$scorch_architecture[(i*2-1)][[1]])[1])}",

        " layer\n\n"))
    }
  }
}

#=== END =======================================================================
