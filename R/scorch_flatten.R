#===============================================================================
# FUNCTION TO ADD A FLATTEN NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Flatten Node to a Scorch Model
#'
#' @description
#' Convenience wrapper that adds a \code{torch::nn_flatten} node to the
#' Scorch model graph. Flattens contiguous dimensions of the input
#' tensor, typically used between convolutional and linear layers.
#'
#' @param scorch_model A \code{scorch_model} object created by
#'   \code{\link{initiate_scorch}}.
#'
#' @param name A unique character string identifying this node in the
#'   model graph. Names wire the computation graph -- other nodes
#'   reference them via their \code{inputs} argument to define
#'   branching, fusion, and skip connections. Names are arbitrary but
#'   appear in error messages and \code{\link{plot_scorch_model}}
#'   output. Common prefixes: \code{"fc"} (linear), \code{"conv"}
#'   (convolution), \code{"act"} (activation). Use number suffixes
#'   for multiples (e.g., \code{"fc1"}, \code{"fc2"}).
#'
#' @param inputs Character vector of upstream node names. If \code{NULL}
#'   (default), resolved automatically (last node or sole input).
#'
#' @param start_dim Integer. First dimension to flatten (default 2,
#'   preserving the batch dimension).
#'
#' @param end_dim Integer. Last dimension to flatten (default -1,
#'   meaning the last dimension).
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' This is equivalent to calling
#' \code{scorch_layer(model, name, "flatten", start_dim = 2)} but
#' provides a more readable API for a common operation. The default
#' \code{start_dim = 2} preserves the batch dimension (dim 1) and
#' flattens everything else.
#'
#' @examples
#' \dontrun{
#' # After convolutional layers, flatten before a linear layer
#' model <- model |>
#'   scorch_flatten("flat", inputs = "pool2") |>
#'   scorch_layer("fc1", "linear",
#'                in_features = 128, out_features = 64)
#' }
#'
#' @family model construction
#'
#' @export

scorch_flatten <- function(scorch_model,
                           name,
                           inputs = NULL,
                           start_dim = 2,
                           end_dim = -1) {

  #- Instantiate the flatten module.

  flatten_mod <- torch::nn_flatten(start_dim = start_dim, end_dim = end_dim)

  #- Resolve inputs when not specified explicitly.

  if (is.null(inputs)) {

    if (nrow(scorch_model$graph) == 0) {

      if (length(scorch_model$inputs) != 1) {

        stop("Must specify 'inputs' when multiple inputs exist.",
             call. = FALSE)
      }

      inputs <- scorch_model$inputs

    } else {

      inputs <- utils::tail(scorch_model$graph$name, 1)
    }
  }

  #- Append to graph.

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name    = name,
    module  = list(flatten_mod),
    inputs  = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
