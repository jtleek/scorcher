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
#'     \item A string: \code{"linear"}, \code{"conv2d"}, \code{"relu"}.
#'       The \code{nn_} prefix is added automatically if missing.
#'     \item An unquoted name: \code{linear}, \code{conv2d}.
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
#'   \code{kernel_size}, \code{p}).
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
#' @examples
#' \dontrun{
#' # String (most common)
#' model <- model |>
#'   scorch_layer("fc1", "linear", in_features = 10, out_features = 32)
#'
#' # Unquoted symbol
#' model <- model |>
#'   scorch_layer("act1", relu)
#'
#' # Direct nn_module constructor
#' model <- model |>
#'   scorch_layer("fc2", torch::nn_linear,
#'                in_features = 32, out_features = 1)
#'
#' # Explicit input wiring (for multi-input models)
#' model <- model |>
#'   scorch_layer("branch_a", "linear", inputs = "stream_a",
#'                in_features = 10, out_features = 16)
#' }
#'
#' @family model construction
#'
#' @export

scorch_layer <- function(scorch_model,
                         name,
                         layer_fn,
                         inputs = NULL,
                         ...) {

  #- Capture the raw call to distinguish strings, symbols, and functions.

  mc <- match.call()

  fn_expr <- mc$layer_fn

  if (is.symbol(fn_expr) || is.character(layer_fn)) {

    #- Either an unquoted name (symbol) or a string.

    fn_name <- if (is.symbol(fn_expr)) as.character(fn_expr) else layer_fn

    #- Auto-prepend nn_ if not already present.

    if (!grepl("^nn_", fn_name)) fn_name <- paste0("nn_", fn_name)

    #- Check that the function exists in torch.

    if (!exists(fn_name, envir = asNamespace("torch"), mode = "function")) {

      stop("No torch layer called '", fn_name, "'.", call. = FALSE)
    }

    layer_fn <- get(fn_name, envir = asNamespace("torch"))

  } else if (is.function(layer_fn)) {

    #- User passed a constructor directly -- keep it.
    NULL

  } else {

    stop("`layer_fn` must be a torch layer name or function.", call. = FALSE)
  }

  #- Resolve inputs when not specified explicitly.

  if (is.null(inputs)) {

    if (nrow(scorch_model$graph) == 0) {

      #- Graph is empty: must have exactly one declared input.

      if (length(scorch_model$inputs) == 0) {

        stop("No inputs declared. Add at least one with scorch_input().",
             call. = FALSE)

      } else if (length(scorch_model$inputs) > 1) {

        stop("Must specify 'inputs' when multiple inputs exist.",
             call. = FALSE)
      }

      inputs <- scorch_model$inputs

    } else {

      #- Default to the last node in the graph.

      inputs <- utils::tail(scorch_model$graph$name, 1)
    }
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

  #- Instantiate the module.

  module <- do.call(layer_fn, list(...))

  #- Append to graph.

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name    = name,
    module  = list(module),
    inputs  = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
