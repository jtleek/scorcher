#===============================================================================
# FUNCTION TO ADD A POOLING NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Pooling Node to a Scorch Model
#'
#' @description
#' Convenience wrapper that adds a pooling node to the Scorch model
#' graph. Supports max, average, adaptive max, and adaptive average
#' pooling in 1D, 2D, and 3D variants.
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
#' @param .name Optional node name. Can be a string or an unquoted name.
#'
#' @param .from Optional upstream node reference. Can be a string or an
#'   unquoted name.
#'
#' @param method Character. Pooling method:
#'   \code{"max"} (default), \code{"avg"}, \code{"adaptive_max"},
#'   or \code{"adaptive_avg"}.
#'
#' @param type Character. Dimensionality variant:
#'   \code{"1d"}, \code{"2d"} (default), or \code{"3d"}.
#'
#' @param ... Additional arguments passed to the underlying
#'   \code{torch::nn_*_pool*} function (e.g., \code{kernel_size},
#'   \code{stride}, \code{output_size}).
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' For standard pooling (\code{"max"}, \code{"avg"}), the
#' \code{kernel_size} argument is required. For adaptive pooling
#' (\code{"adaptive_max"}, \code{"adaptive_avg"}), the
#' \code{output_size} argument is required instead.
#'
#' @examples
#' \dontrun{
#' # Max pooling after conv2d
#' model <- model |>
#'   scorch_layer(conv2d, in_channels = 3, out_channels = 16,
#'                kernel_size = 3) |>
#'   scorch_pool(kernel_size = 2)
#'
#' # Adaptive average pooling to fixed output size
#' model <- model |>
#'   scorch_pool("apool", method = "adaptive_avg", output_size = 1)
#'
#' # 1D average pooling for sequence data
#' model <- model |>
#'   scorch_pool("pool1", method = "avg", type = "1d", kernel_size = 3)
#' }
#'
#' @family model construction
#'
#' @export

scorch_pool <- function(scorch_model,
                        name,
                        inputs = NULL,
                        method = "max",
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
    auto_prefix = paste(method, type, "pool", sep = "_")
  )
  scorch_model <- node_name$model
  name <- node_name$name

  #- Dispatch to the correct pooling function.

  pool_key <- paste0(method, "_", type)

  pool_fn <- switch(pool_key,
    "max_1d"          = torch::nn_max_pool1d,
    "max_2d"          = torch::nn_max_pool2d,
    "max_3d"          = torch::nn_max_pool3d,
    "avg_1d"          = torch::nn_avg_pool1d,
    "avg_2d"          = torch::nn_avg_pool2d,
    "avg_3d"          = torch::nn_avg_pool3d,
    "adaptive_max_1d" = torch::nn_adaptive_max_pool1d,
    "adaptive_max_2d" = torch::nn_adaptive_max_pool2d,
    "adaptive_max_3d" = torch::nn_adaptive_max_pool3d,
    "adaptive_avg_1d" = torch::nn_adaptive_avg_pool1d,
    "adaptive_avg_2d" = torch::nn_adaptive_avg_pool2d,
    "adaptive_avg_3d" = torch::nn_adaptive_avg_pool3d,
    stop("Unknown pool method/type '", method, "' / '", type,
         "'. method: 'max', 'avg', 'adaptive_max', 'adaptive_avg'. ",
         "type: '1d', '2d', '3d'.", call. = FALSE)
  )

  pool_mod <- pool_fn(...)

  #- Append to graph.

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = pool_mod,
    inputs = inputs,
    node_type = "layer",
    constructor = paste0(method, "_pool", type),
    args = c(list(method = method, type = type), list(...)),
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
