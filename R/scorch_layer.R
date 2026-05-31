#===============================================================================
# FUNCTION TO ADD A LAYER NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Layer Node to a Scorch Model
#'
#' @description
#' Adds a named layer node to the Scorch model graph. The layer is
#' instantiated from a torch \code{nn_*} constructor and wired to one
#' or more upstream nodes. This is the primary function for building
#' model architectures in scorcher.
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
#' @param layer_fn The layer to add. Can be specified in three ways:
#'   \enumerate{
#'     \item A string: \code{"linear"}, \code{"conv2d"}, \code{"relu"},
#'       \code{"multihead_attention"}.
#'       The \code{nn_} prefix is added automatically if missing.
#'     \item An unquoted name: \code{linear}, \code{conv2d},
#'       \code{multihead_attention}.
#'       Resolved the same way as a string.
#'     \item A function: \code{torch::nn_linear}, or any \code{nn_module}
#'       constructor. Used as-is.
#'   }
#'
#' @param inputs Character vector of upstream node names that feed into
#'   this layer. If \code{NULL} (default), inputs are resolved
#'   automatically: the last node in the graph is used, or, if the graph
#'   is empty, the sole input declared via \code{\link{scorch_input}}.
#'   Must be specified explicitly when the model has multiple inputs and
#'   the graph is empty.
#'
#' @param ... Additional arguments passed to the \code{layer_fn}
#'   constructor (e.g., \code{in_features}, \code{out_features},
#'   \code{kernel_size}, \code{p}). For \code{multihead_attention},
#'   forward-pass options such as \code{attn_mask} and \code{causal}
#'   are also accepted.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' Each call appends one row to the graph tibble with columns
#' \code{name}, \code{module} (the instantiated \code{nn_module}), and
#' \code{inputs} (character vector of upstream node names). The graph
#' topology is later traversed by \code{\link{compile_scorch}} to
#' build the forward pass.
#'
#' For residual / skip connections, use \code{\link{scorch_add_skip}}
#' instead of the old \code{use_residual} argument.
#'
#' For \code{multihead_attention}, provide query, key, and value node
#' references with \code{.from = c(query, key, value)}. The attention
#' node returns the attention output tensor. Set \code{causal = TRUE}
#' to apply an upper-triangular attention mask for autoregressive
#' transformer blocks.
#'
#' @examples
#' \dontrun{
#' # Unquoted layer type with automatic naming
#' model <- model |>
#'   scorch_layer(linear, in_features = 10, out_features = 32)
#'
#' # Activation with automatic naming
#' model <- model |>
#'   scorch_layer(relu)
#'
#' # Direct nn_module constructor
#' model <- model |>
#'   scorch_layer(torch::nn_linear, in_features = 32, out_features = 1)
#'
#' # Explicit input wiring (for multi-input models)
#' model <- model |>
#'   scorch_layer(linear, .from = stream_a,
#'                in_features = 10, out_features = 16)
#'
#' # Multi-head self-attention
#' model <- model |>
#'   scorch_layer(multihead_attention,
#'                embed_dim = 64, num_heads = 4, causal = TRUE,
#'                .from = c(embeddings, embeddings, embeddings),
#'                .name = attention)
#' }
#'
#' @family model construction
#'
#' @export

scorch_layer <- function(scorch_model,
                         name,
                         layer_fn = NULL,
                         inputs = NULL,
                         .name = NULL,
                         .from = NULL,
                         ...) {

  scorch_model <- scorch_check_model(scorch_model)

  name_expr <- if (missing(.name)) NULL else substitute(.name)
  from_expr <- if (missing(.from)) NULL else substitute(.from)
  inputs_expr <- if (missing(inputs)) NULL else substitute(inputs)

  if (missing(layer_fn) || is.null(layer_fn)) {
    layer_expr <- substitute(name)
    legacy_name_expr <- NULL
    layer_value <- tryCatch(eval(layer_expr, parent.frame()),
                            error = function(e) layer_expr)
  } else {
    legacy_name_expr <- substitute(name)
    layer_expr <- substitute(layer_fn)
    layer_value <- tryCatch(eval(layer_expr, parent.frame()),
                            error = function(e) layer_expr)
  }

  constructor <- scorch_constructor_name(layer_expr, fallback = "layer")
  layer_fn <- scorch_resolve_layer_fn(layer_value, layer_expr)
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
    auto_prefix = constructor
  )
  scorch_model <- node_name$model
  name <- node_name$name

  #- Instantiate the module.

  args <- list(...)
  layer_args <- scorch_split_layer_args(args, constructor)
  module <- do.call(layer_fn, layer_args$constructor)
  module <- scorch_finalize_layer_module(
    module,
    constructor,
    forward_args = layer_args$forward,
    causal = layer_args$causal,
    batch_first = layer_args$batch_first
  )

  #- Append to graph.

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = module,
    inputs = inputs,
    node_type = "layer",
    constructor = constructor,
    args = args,
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
