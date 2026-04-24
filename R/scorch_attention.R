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
                             inputs,
                             embed_dim,
                             num_heads,
                             ...) {

  #- Validate inputs and name before building the module.

  all_names <- c(scorch_model$inputs, scorch_model$graph$name)
  bad_inputs <- setdiff(inputs, all_names)
  if (length(bad_inputs) > 0)
    stop("Input node(s) not found in model: ",
         paste(bad_inputs, collapse = ", "), call. = FALSE)

  if (name %in% scorch_model$graph$name || name %in% scorch_model$inputs)
    stop("Node name '", name, "' already exists in the model graph.",
         call. = FALSE)

  #- Wrap nn_multihead_attention so forward() returns only the output
  #- tensor, not the (output, weights) tuple that torch returns.

  raw_attn <- torch::nn_multihead_attention(embed_dim, num_heads, ...)

  attn_mod <- torch::nn_module(
    initialize = function() {
      self$attn <- raw_attn
    },
    forward = function(query, key, value) {
      self$attn(query, key, value)[[1]]
    }
  )()

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(attn_mod),
    inputs = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
