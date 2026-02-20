#===============================================================================
# FUNCTION TO ADD A BATCH NORMALIZATION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Batch Normalization Node to a Scorch Model
#'
#' @description
#' Convenience wrapper that adds a \code{torch::nn_batch_norm1d} node
#' to the Scorch model graph.
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
#' @param num_features Integer. Number of features (the C dimension in
#'   BatchNorm1d).
#'
#' @param ... Additional arguments passed to
#'   \code{torch::nn_batch_norm1d()} (e.g., \code{momentum},
#'   \code{eps}).
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' This is equivalent to calling
#' \code{scorch_layer(model, name, "batch_norm1d", inputs, num_features = n)}
#' but provides a more readable API for a common operation.
#'
#' @examples
#' \dontrun{
#' model <- model |>
#'   scorch_layer("fc1", "linear", in_features = 10, out_features = 32) |>
#'   scorch_batchnorm("bn1", num_features = 32) |>
#'   scorch_layer("act1", "relu")
#' }
#'
#' @family model construction
#'
#' @export

scorch_batchnorm <- function(scorch_model,
                             name,
                             inputs = NULL,
                             num_features,
                             ...) {

  bn_mod <- torch::nn_batch_norm1d(num_features = num_features, ...)

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
    name   = name,
    module = list(bn_mod),
    inputs = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
