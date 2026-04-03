#===============================================================================
# FUNCTION TO ADD A BATCH NORMALIZATION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Batch Normalization Node to a Scorch Model
#'
#' @description
#' Convenience wrapper that adds a batch normalization node to the
#' Scorch model graph. Supports 1D, 2D, and 3D variants via the
#' \code{type} argument.
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
#' @param num_features Integer. Number of features (the C dimension).
#'
#' @param type Character. Which batch norm variant to use:
#'   \code{"1d"} (default) for 2D/3D inputs (linear layers),
#'   \code{"2d"} for 4D inputs (conv2d layers),
#'   \code{"3d"} for 5D inputs (conv3d layers).
#'
#' @param ... Additional arguments passed to the underlying
#'   \code{torch::nn_batch_norm*} function (e.g., \code{momentum},
#'   \code{eps}).
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' This is equivalent to calling
#' \code{scorch_layer(model, name, "batch_norm1d", inputs, num_features = n)}
#' (or \code{"batch_norm2d"}, \code{"batch_norm3d"}) but provides a
#' more readable API for a common operation.
#'
#' @examples
#' \dontrun{
#' # After linear layers (1D)
#' model <- model |>
#'   scorch_layer("fc1", "linear", in_features = 10, out_features = 32) |>
#'   scorch_batchnorm("bn1", num_features = 32) |>
#'   scorch_layer("act1", "relu")
#'
#' # After conv2d layers (2D)
#' model <- model |>
#'   scorch_layer("conv1", "conv2d", in_channels = 3, out_channels = 16,
#'                kernel_size = 3) |>
#'   scorch_batchnorm("bn1", num_features = 16, type = "2d") |>
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
                             type = "1d",
                             ...) {

  #- Resolve inputs when not specified explicitly.

  if (is.null(inputs)) {

    if (nrow(scorch_model$graph) == 0) {

      if (length(scorch_model$inputs) == 0) {

        stop("No inputs declared. Add at least one with scorch_input().",
             call. = FALSE)

      } else if (length(scorch_model$inputs) > 1) {

        stop("Must specify 'inputs' when multiple inputs exist.",
             call. = FALSE)
      }

      inputs <- scorch_model$inputs

    } else {

      inputs <- utils::tail(scorch_model$graph$name, 1)
    }
  }

  #- Validate name before building the module.

  if (name %in% scorch_model$graph$name || name %in% scorch_model$inputs)
    stop("Node name '", name, "' already exists in the model graph.",
         call. = FALSE)

  bn_fn <- switch(type,
    "1d" = torch::nn_batch_norm1d,
    "2d" = torch::nn_batch_norm2d,
    "3d" = torch::nn_batch_norm3d,
    stop("Unknown batchnorm type '", type,
         "'. Use '1d', '2d', or '3d'.", call. = FALSE)
  )

  bn_mod <- bn_fn(num_features = num_features, ...)

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
