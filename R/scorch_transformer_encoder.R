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
#' @param name A single character string giving a unique name for this
#'   node in the computation graph.
#'
#' @param inputs Character vector of a single upstream node name.
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
                                       inputs,
                                       embed_dim,
                                       num_heads,
                                       dim_feedforward = 2048,
                                       dropout = 0.1,
                                       ...) {

  tr_mod <- torch::nn_transformer_encoder_layer(

    d_model         = embed_dim,
    nhead           = num_heads,
    dim_feedforward = dim_feedforward,
    dropout         = dropout,
    ...
  )

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(tr_mod),
    inputs = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
