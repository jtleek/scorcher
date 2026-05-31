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
#' \code{scorch_layer(model, conv_transpose2d, ...)} but
#' provides a more readable API with a \code{type} parameter for
#' selecting dimensionality.
#'
#' @examples
#' \dontrun{
#' # 2D transposed convolution for upsampling
#' model <- model |>
#'   scorch_conv_transpose(in_channels = 64, out_channels = 32,
#'                         kernel_size = 4, stride = 2, padding = 1)
#'
#' # 1D transposed convolution for sequence generation
#' model <- model |>
#'   scorch_conv_transpose(type = "1d",
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
                                  .name = NULL,
                                  .from = NULL,
                                  ...) {

  scorch_model <- scorch_check_model(scorch_model)

  name_expr <- if (missing(.name)) NULL else substitute(.name)
  legacy_name_expr <- if (missing(name)) NULL else substitute(name)
  from_expr <- if (missing(.from)) NULL else substitute(.from)
  inputs_expr <- if (missing(inputs)) NULL else substitute(inputs)

  inputs <- scorch_resolve_inputs(
    scorch_model,
    inputs = if (is.null(inputs_expr)) NULL else
      scorch_parse_refs_expr(inputs_expr, arg = "inputs"),
    from = if (is.null(from_expr)) NULL else
      scorch_parse_refs_expr(from_expr, arg = ".from")
  )

  node_name <- scorch_prepare_node_name(
    scorch_model,
    explicit_expr = name_expr,
    legacy_expr = legacy_name_expr,
    auto_prefix = paste0("conv_transpose", type)
  )
  scorch_model <- node_name$model
  name <- node_name$name

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

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = conv_mod,
    inputs = inputs,
    node_type = "layer",
    constructor = paste0("conv_transpose", type),
    args = c(list(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = kernel_size,
                  type = type), list(...)),
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
