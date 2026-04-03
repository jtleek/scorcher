#===============================================================================
# FUNCTION TO ADD A SKIP CONNECTION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Skip Connection Node to a Scorch Model
#'
#' @description
#' Creates a node that performs element-wise addition of two upstream
#' tensors. Used to implement residual / skip connections where the
#' output is \code{x + skip}.
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
#' @param inputs A length-2 character vector: \code{c("main_path",
#'   "skip_path")}. Both upstream nodes must produce tensors of the
#'   same shape.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' The node is implemented as a lightweight \code{torch::nn_module}
#' that sums its two inputs. It has no learnable parameters. This
#' replaces the old \code{use_residual} argument in
#' \code{\link{scorch_layer}}.
#'
#' @examples
#' \dontrun{
#' # Residual connection around a linear + relu block
#' model <- model |>
#'   scorch_layer("fc1", "linear", inputs = "x",
#'                in_features = 32, out_features = 32) |>
#'   scorch_layer("act1", "relu") |>
#'   scorch_add_skip("res1", inputs = c("act1", "x"))
#' }
#'
#' @family model construction
#'
#' @export

scorch_add_skip <- function(scorch_model,
                            name,
                            inputs) {

  if (length(inputs) != 2) {

    stop("`inputs` must be length 2.", call. = FALSE)
  }

  #- Validate inputs and name before building the module.

  all_names <- c(scorch_model$inputs, scorch_model$graph$name)
  bad_inputs <- setdiff(inputs, all_names)
  if (length(bad_inputs) > 0)
    stop("Input node(s) not found in model: ",
         paste(bad_inputs, collapse = ", "), call. = FALSE)

  if (name %in% scorch_model$graph$name || name %in% scorch_model$inputs)
    stop("Node name '", name, "' already exists in the model graph.",
         call. = FALSE)

  #- Build a lightweight module that sums its two inputs.

  skip_mod <- torch::nn_module(

    initialize = function() {},

    forward = function(x, skip) {

      x + skip
    }
  )()

  #- Append to graph.

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(skip_mod),
    inputs = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
