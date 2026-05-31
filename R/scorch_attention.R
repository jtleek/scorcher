#===============================================================================
# FUNCTION TO ADD A MULTI-HEAD ATTENTION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Multi-Head Attention Node to a Scorch Model
#'
#' @description
#' Adds a \code{torch::nn_multihead_attention} node to the Scorch
#' model graph. The node expects three upstream inputs representing
#' query, key, and value tensors.
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
#' @param inputs Character vector of three upstream node names:
#'   \code{c("query", "key", "value")}.
#'
#' @param .name Optional node name. Can be a string or an unquoted name.
#'
#' @param .from Optional upstream query, key, and value references. Can be a
#'   character vector, unquoted names, or \code{c(...)} of unquoted names.
#'
#' @param embed_dim Integer. Total embedding dimension.
#'
#' @param num_heads Integer. Number of attention heads.
#'
#' @param ... Additional arguments passed to
#'   \code{torch::nn_multihead_attention()} (e.g., \code{dropout}).
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' The \code{embed_dim} must be divisible by \code{num_heads}. The
#' three inputs correspond to the query, key, and value tensors
#' passed to the attention mechanism.
#'
#' @examples
#' \dontrun{
#' model <- model |>
#'   scorch_attention("attn1",
#'                    inputs    = c("query", "key", "value"),
#'                    embed_dim = 64,
#'                    num_heads = 4)
#' }
#'
#' @family model construction
#'
#' @export

scorch_attention <- function(scorch_model,
                             name,
                             inputs = NULL,
                             embed_dim,
                             num_heads,
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
      scorch_parse_refs_expr(from_expr, arg = ".from"),
    allow_multi = TRUE
  )

  if (length(inputs) != 3) {
    stop("`scorch_attention()` requires query, key, and value inputs.",
         call. = FALSE)
  }

  node_name <- scorch_prepare_node_name(
    scorch_model,
    explicit_expr = name_expr,
    legacy_expr = legacy_name_expr,
    auto_prefix = "attention"
  )
  scorch_model <- node_name$model
  name <- node_name$name

  args <- c(list(embed_dim = embed_dim, num_heads = num_heads), list(...))
  layer_args <- scorch_split_layer_args(args, "multihead_attention")
  raw_attn <- do.call(torch::nn_multihead_attention, layer_args$constructor)
  attn_mod <- scorch_finalize_layer_module(
    raw_attn,
    "multihead_attention",
    forward_args = layer_args$forward,
    causal = layer_args$causal,
    batch_first = layer_args$batch_first
  )

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = attn_mod,
    inputs = inputs,
    node_type = "block",
    constructor = "multihead_attention",
    args = args,
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
