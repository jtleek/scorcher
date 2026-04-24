#===============================================================================
# FUNCTION TO ADD A TRANSPOSED CONVOLUTION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Transposed Convolution Node to a Scorch Model
#'
#' @description
#' Convenience wrapper that adds a transposed convolution
#' (deconvolution) node to the Scorch model graph. Supports 1D, 2D,
#' and 3D variants. Commonly used in decoder pathways, U-Nets,
#' autoencoders, and generative models.
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
#' @param in_channels Integer. Number of input channels.
#'
#' @param out_channels Integer. Number of output channels.
#'
#' @param kernel_size Integer or tuple. Size of the convolving kernel.
#'
#' @param type Character. Dimensionality variant:
#'   \code{"1d"}, \code{"2d"} (default), or \code{"3d"}.
#'
#' @param ... Additional arguments passed to the underlying
#'   \code{torch::nn_conv_transpose*} function (e.g., \code{stride},
#'   \code{padding}, \code{output_padding}).
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' This is equivalent to calling
#' \code{scorch_layer(model, name, "conv_transpose2d", ...)} but
#' provides a more readable API with a \code{type} parameter for
#' selecting dimensionality.
#'
#' @examples
#' \dontrun{
#' # 2D transposed convolution for upsampling
#' model <- model |>
#'   scorch_conv_transpose("deconv1",
#'                         in_channels = 64, out_channels = 32,
#'                         kernel_size = 4, stride = 2, padding = 1)
#'
#' # 1D transposed convolution for sequence generation
#' model <- model |>
#'   scorch_conv_transpose("deconv1", type = "1d",
#'                         in_channels = 128, out_channels = 64,
#'                         kernel_size = 3)
#' }
#'
#' @family model construction
#'
#' @export

scorch_conv_transpose <- function(scorch_model,
                                  name,
                                  inputs = NULL,
                                  in_channels,
                                  out_channels,
                                  kernel_size,
                                  type = "2d",
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

  #- Dispatch to the correct transposed convolution function.

  conv_fn <- switch(type,
    "1d" = torch::nn_conv_transpose1d,
    "2d" = torch::nn_conv_transpose2d,
    "3d" = torch::nn_conv_transpose3d,
    stop("Unknown conv_transpose type '", type,
         "'. Use '1d', '2d', or '3d'.", call. = FALSE)
  )

  conv_mod <- conv_fn(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      ...)

  #- Append to graph.

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(conv_mod),
    inputs = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
