#===============================================================================
# FUNCTIONS TO INITIATE A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#-------------------------------------------------------------------------------
# NOTES:
#
# I've kept this where it takes in a dataloader and outputs a scorch_model
# object, but I've:
#
#   1. Changed the object holding the architecture from a list to a tibble,
#      which will store the layers in a graph structure.
#
#   2. Added more components to the scorch_model object to help with the
#      bookkeeping for more complex architectures.
#-------------------------------------------------------------------------------

#' Initiate a Scorch Model
#'
#' Creates a new `scorch_model` object to which you can add layers, inputs, and
#' outputs.
#'
#' This function initializes an empty model graph and optionally attaches a
#' dataloader created by \link{scorch_create_dataloader}.
#'
#' @param dl Optional dataloader created by \link{scorch_create_dataloader}
#' to attach to the model (default is `NULL`).
#'
#' @return A `scorch_model` object with the following components:
#'
#' \describe{
#'   \item{graph}{A tibble storing layer definitions in graph order.}
#'   \item{inputs}{Character vector of input node names.}
#'   \item{outputs}{Character vector of output node names.}
#'   \item{compiled}{Logical flag indicating if the model has been compiled.}
#'   \item{nn_model}{The compiled `torch::nn_module`, available once compiled.}
#'   \item{optimizer}{The optimizer object, available once compiled.}
#'   \item{loss_fn}{The loss function(s), available once compiled.}
#'   \item{dl}{The attached dataloader, if provided.}
#' }
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(
#'
#'     input  = mtcars$wt,
#'     output = mtcars$mpg,
#'     batch_size = 16
#' )
#'
#' sm <- initiate_scorch(dl)
#'
#' print(sm)
#'
#' @import tibble
#'
#' @export

initiate_scorch <- function(

    dl = NULL) {

    #- Create the base structure for the scorch_model object

    sm <- list(

        graph = tibble::tibble(

            name = character(),
            module = list(),
            inputs = list()
        ),

        inputs    = character(),
        outputs   = character(),
        compiled  = FALSE,
        nn_model  = NULL,
        optimizer = NULL,
        loss_fn   = NULL,
        dl        = NULL
    )

    class(sm) <- "scorch_model"

    #- If a dataloader is provided, attach it

    if (!is.null(dl)) {

      if (!inherits(dl, "dataloader")) {

        stop("`dl` must be a torch::dataloader", call. = FALSE)
      }

      sm$dl <- dl
    }

    return(sm)
}

#=== HELPERS ===================================================================

#--- SCORCH MODEL CLASS --------------------------------------------------------

#' Create a Scorch Model Class
#'
#' @description
#' This function creates an object of class 'scorch_model'.
#'
#' @param obj An object to be converted to a scorch model.
#'
#' @return The input object with the class attribute set to 'scorch_model'.
#'
#' @examples
#'
#' # scorch_model <- create_scorch_model_class(list(scorch_architecture = list()))
#'
#' # class(scorch_model)

# @export

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
#' @examples
#'
#' # input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' # output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' # dl <- scorch_create_dataloader(input, output, batch_size = 2)
#'
#' # scorch_model <- dl |> initiate_scorch() |>
#'
#' #   scorch_layer("linear", 11, 5)
#'
#' # print(scorch_model)

# @export

print.scorch_model <- function(x, ...) {

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
