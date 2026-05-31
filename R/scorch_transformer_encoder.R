#===============================================================================
# FUNCTION TO ADD A TRANSFORMER ENCODER LAYER TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Transformer Encoder Layer to a Scorch Model
#'
#' @description
#' Adds a \code{torch::nn_transformer_encoder_layer} node to the
#' Scorch model graph. Implements one layer of the standard
#' transformer encoder (self-attention + feed-forward).
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
#' @param inputs Character vector of a single upstream node name.
#'
#' @param .name Optional node name. Can be a string or an unquoted name.
#'
#' @param .from Optional upstream node reference. Can be a string or an
#'   unquoted name.
#'
#' @param embed_dim Integer. The d_model dimension (total embedding
#'   size).
#'
#' @param num_heads Integer. Number of attention heads.
#'
#' @param dim_feedforward Integer. Dimension of the feed-forward
#'   network (default 2048).
#'
#' @param dropout Numeric. Dropout rate (default 0.1).
#'
#' @param ... Additional arguments passed to
#'   \code{torch::nn_transformer_encoder_layer()}.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' The \code{embed_dim} must be divisible by \code{num_heads}. This
#' adds a single encoder layer; stack multiple calls for deeper
#' transformers.
#'
#' @examples
#' \dontrun{
#' # Two-layer transformer encoder
#' model <- model |>
#'   scorch_transformer_encoder("enc1", inputs = "embeddings",
#'                              embed_dim = 64, num_heads = 4) |>
#'   scorch_transformer_encoder("enc2", inputs = "enc1",
#'                              embed_dim = 64, num_heads = 4)
#' }
#'
#' @family model construction
#'
#' @export

scorch_transformer_encoder <- function(scorch_model,
                                       name,
                                       inputs = NULL,
                                       embed_dim,
                                       num_heads,
                                       dim_feedforward = 2048,
                                       dropout = 0.1,
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
    auto_prefix = "transformer_encoder"
  )
  scorch_model <- node_name$model
  name <- node_name$name

  tr_mod <- torch::nn_transformer_encoder_layer(

    d_model         = embed_dim,
    nhead           = num_heads,
    dim_feedforward = dim_feedforward,
    dropout         = dropout,
    ...
  )

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = tr_mod,
    inputs = inputs,
    node_type = "block",
    constructor = "transformer_encoder",
    args = c(list(embed_dim = embed_dim,
                  num_heads = num_heads,
                  dim_feedforward = dim_feedforward,
                  dropout = dropout), list(...)),
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
