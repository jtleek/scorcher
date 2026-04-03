#===============================================================================
# FUNCTION TO COMPILE A SCORCH MODEL
#===============================================================================

utils::globalVariables("self")

#=== MAIN FUNCTION =============================================================

#' Compile a Scorch Model
#'
#' @description
#' Compiles a Scorch model by traversing the graph topology to build a
#' \code{torch::nn_module}, then attaches an optimizer and loss function.
#' After compilation, the model is ready for training with
#' \code{\link{fit_scorch}}.
#'
#' @param scorch_model A \code{scorch_model} object built with
#'   \code{\link{initiate_scorch}}, \code{\link{scorch_input}},
#'   \code{\link{scorch_layer}}, and \code{\link{scorch_output}}.
#'
#' @param loss_fn A loss function or a named list of loss functions.
#'   For single-output models, pass a single loss (e.g.,
#'   \code{torch::nn_mse_loss()}). For multi-output models, pass a
#'   named list with one entry per output node (e.g.,
#'   \code{list(reg_head = nn_mse_loss(), cls_head = nn_cross_entropy_loss())}).
#'
#' @param optimizer_fn An optimizer constructor function (e.g.,
#'   \code{torch::optim_adam}, \code{torch::optim_sgd}). Not an
#'   instantiated optimizer -- the constructor is called internally
#'   with the model parameters.
#'
#' @param optimizer_params A named list of optimizer parameters (e.g.,
#'   \code{list(lr = 1e-3, weight_decay = 1e-4)}). Passed to
#'   \code{optimizer_fn} along with the model parameters.
#'
#' @returns The same \code{scorch_model} object with the following
#'   fields populated:
#'   \describe{
#'     \item{\code{nn_model}}{The compiled \code{torch::nn_module}.}
#'     \item{\code{optimizer}}{The instantiated optimizer.}
#'     \item{\code{loss_fn}}{The loss function (or named list).}
#'     \item{\code{compiled}}{Set to \code{TRUE}.}
#'   }
#'
#' @details
#' Compilation works by:
#'   \enumerate{
#'     \item Registering every graph node as a sub-module of a new
#'       \code{nn_module}. This enables torch to track all learnable
#'       parameters.
#'     \item Building a \code{forward()} method that traverses the
#'       graph in row order, passing each node its upstream tensors
#'       via a name-lookup environment.
#'     \item Instantiating the optimizer with the model parameters.
#'   }
#'
#' The graph must have at least one input (\code{\link{scorch_input}})
#' and one output (\code{\link{scorch_output}}) declared before
#' compilation.
#'
#' @examples
#' \dontrun{
#' # Single-output model
#' model <- model |>
#'   compile_scorch(
#'     loss_fn          = torch::nn_mse_loss(),
#'     optimizer_fn     = torch::optim_adam,
#'     optimizer_params = list(lr = 1e-3)
#'   )
#'
#' # Multi-output model
#' model <- model |>
#'   compile_scorch(
#'     loss_fn = list(
#'       reg_head = torch::nn_mse_loss(),
#'       cls_head = torch::nn_cross_entropy_loss()
#'     ),
#'     optimizer_fn     = torch::optim_adam,
#'     optimizer_params = list(lr = 1e-3)
#'   )
#' }
#' @import torch
#'
#' @family model construction
#'
#' @export

compile_scorch <- function(scorch_model,
                           loss_fn          = torch::nn_mse_loss(),
                           optimizer_fn     = torch::optim_adam,
                           optimizer_params = list(lr = 1e-3)) {

  graph   <- scorch_model$graph
  inputs  <- scorch_model$inputs
  outputs <- scorch_model$outputs

  #- Validate model before compiling.

  if (length(inputs) == 0)
    stop("Model has no inputs. Add at least one with scorch_input().",
         call. = FALSE)

  if (nrow(graph) == 0)
    stop("Model has no layers. Add at least one with scorch_layer().",
         call. = FALSE)

  if (length(outputs) == 0)
    stop("Model has no outputs. Mark at least one with scorch_output().",
         call. = FALSE)

  bad_outputs <- setdiff(outputs, graph$name)
  if (length(bad_outputs) > 0)
    stop("Output node(s) not found in graph: ",
         paste(bad_outputs, collapse = ", "), call. = FALSE)

  if (is.list(loss_fn) && length(outputs) > 1) {
    missing_loss <- setdiff(outputs, names(loss_fn))
    if (length(missing_loss) > 0)
      stop("Loss function missing for output(s): ",
           paste(missing_loss, collapse = ", "), call. = FALSE)
  }

  #- Build the nn_module by registering all graph nodes as sub-modules
  #- and defining the forward pass as a graph traversal.

  mod <- torch::nn_module(

    initialize = function() {

      for (i in seq_len(nrow(graph))) {

        self[[graph$name[i]]] <- graph$module[[i]]
      }
    },

    forward = function(...) {

      args <- list(...)

      env  <- new.env(parent = emptyenv())

      #- Assign inputs to the environment.

      if (length(inputs) == 1) {

        env[[inputs]] <- args[[1]]

      } else {

        for (nm in names(args)) env[[nm]] <- args[[nm]]
      }

      #- Compute each node in graph order.

      for (i in seq_len(nrow(graph))) {

        node    <- graph[i, ]

        in_vals <- lapply(node$inputs[[1]], function(nm) env[[nm]])

        out     <- do.call(self[[node$name]], in_vals)

        env[[node$name]] <- out
      }

      #- Return outputs.

      if (length(outputs) == 1) {

        env[[outputs]]

      } else {

        purrr::map(outputs, ~ env[[.x]])
      }
    }
  )

  #- Instantiate the module, optimizer, and attach everything.

  scorch_model$nn_model  <- mod()

  scorch_model$optimizer <- do.call(optimizer_fn,
                          c(list(params = scorch_model$nn_model$parameters),
                            optimizer_params))

  scorch_model$optimizer_fn     <- optimizer_fn
  scorch_model$optimizer_params <- optimizer_params

  scorch_model$loss_fn   <- loss_fn

  scorch_model$compiled  <- TRUE

  scorch_model
}

#=== END =======================================================================
