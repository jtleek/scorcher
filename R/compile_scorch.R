#===============================================================================
# COMPILE SCORCH MODEL
#===============================================================================

utils::globalVariables(c("self", "aux"))

#=== MAIN FUNCTION =============================================================

#-------------------------------------------------------------------------------
# NOTES:
#
#   1. Main update is that the model now compiles by traversing the topology
#      defined by the scorch_model graph (tibble), rather than sequentially
#      adding layers according to their list position.
#
#   2. Also now handles multiple inputs/outputs.
#-------------------------------------------------------------------------------

#' Compile a Scorch Model
#'
#' Traverse the topology defined in the scorch model graph to build a single
#' `torch::nn_module`. Also attaches the specified loss function and optimizer.
#'
#' @param sm A `scorch_model` object built via `initiate_scorch()` and layers.
#'
#' @param loss_fn A loss function, e.g., `nn_mse_loss()` or named list of
#' losses for multi-head models.
#'
#' @param optimizer_fn Optimizer constructor, e.g., `optim_adam`.
#'
#' @param optimizer_params Named list of optimizer parameters
#' (e.g., `list(lr = 1e-3)`).
#'
#' @return The same `scorch_model` with `nn_model`, `optimizer`, and `loss_fn`
#' set, and `compiled = TRUE`.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(mtcars$wt, mtcars$mpg, batch_size=16)
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("wt") |>
#'
#'   scorch_layer("h1", "linear", in_features = 1, out_features = 8) |>
#'
#'   scorch_output("h1")
#'
#' sm <- compile_scorch(sm,
#'
#'   loss_fn = torch::nn_mse_loss(),
#'
#'   optimizer_fn = torch::optim_adam,
#'
#'   optimizer_params = list(lr=0.01))
#'
#' print(sm)
#'
#' @import torch
#'
#' @export

compile_scorch <- function(

    sm,
    loss_fn = nn_mse_loss(),
    optimizer_fn = optim_adam,
    optimizer_params = list(lr = 1e-3)) {

    graph   <- sm$graph
    inputs  <- sm$inputs
    outputs <- sm$outputs

    mod <- torch::nn_module(

        initialize = function() {

            for (i in seq_len(nrow(graph))) {

                self[[graph$name[i]]] <- graph$module[[i]]
            }
        },

        forward = function(...) {

            args <- list(...)

            env <- new.env(parent = emptyenv())

            #- Assign inputs

            if (length(inputs) == 1) {

                env[[inputs]] <- args[[1]]

            } else {

                for (nm in names(args)) env[[nm]] <- args[[nm]]
            }

            #- Compute each node

            for (i in seq_len(nrow(graph))) {

                node <- graph[i, ]

                in_vals <- lapply(node$inputs[[1]], function(nm) env[[nm]])

                out <- do.call(self[[node$name]], in_vals)

                env[[node$name]] <- out
            }

            #- Return outputs

            if (length(outputs) == 1) {

                env[[outputs]]

            } else {

                purrr::map(outputs, ~ env[[.x]])
            }
        }
    )

    sm$nn_model <- mod()

    sm$optimizer <- do.call(optimizer_fn,

        c(list(params = sm$nn_model$parameters), optimizer_params)
    )

    sm$loss_fn <- loss_fn

    sm$compiled <- TRUE

    return(sm)
}

#=== HELPERS ===================================================================

#--- COMPILED SCORCH MODEL CLASS -----------------------------------------------

#' Create a Compiled Scorch Model Class
#'
#' @description
#' This function creates an object of class 'compiled_scorch_model' by
#' appending it to the existing classes of the input object.
#'
#' @param obj An object to be converted to a compiled scorch model.
#'
#' @return The input object with the class attribute set to include
#' 'compiled_scorch_model'.
#'
#' @export
#'
#' @examples
#'
#' model <- list(children = list(modules = "module details"))
#'
#' compiled_model <- create_scorch_nn_module_class(model)
#'
#' class(compiled_model)

create_scorch_nn_module_class <- function(obj) {

  class(obj) <- c("compiled_scorch_model", class(obj))

  return(obj)
}

#--- PRINT METHOD --------------------------------------------------------------

#' Print Method for Scorch Neural Network Module
#'
#' @description
#' This function defines the print method for objects of class
#' 'scorch_nn_module'.
#'
#' @param x An object of class 'scorch_nn_module'.
#'
#' @param ... Additional arguments to be passed to the print function.
#'
#' @export
#'
#' @examples
#'
#' model <- list(children = list(modules = "module details"))
#'
#' model$children$modules <- "modules: 123\nlayer1 #64\nlayer2 #32"
#'
#' class(model) <- c("scorch_nn_module", class(model))
#'
#' print(model)

print.scorch_nn_module <- function(x, ...) {

  output <- utils::capture.output(x$children$modules)

  n_param <- stringr::str_extract(output[1], "\\d+")

  n_params_layer <- gsub("#", "", stringr::str_extract(

    output[grep("#", output)], "#\\d+"))

  cat(glue::glue("This scorch_nn_module has ",

    "{crayon::red(n_param)} parameters \n\n"))

  cat("\n\n")

  for(i in 1:length(n_params_layer)) {

    cat(glue::glue(" * Layer {i} has ", "

      {crayon::red(n_params_layer[i])} parameters\n\n"))
  }
}

#=== END =======================================================================
