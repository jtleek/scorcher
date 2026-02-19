#===============================================================================
# FUNCTION TO ADD A DROPOUT NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Dropout Node to a Scorch Model
#'
#' @description
#' Convenience wrapper that adds a \code{torch::nn_dropout} node to
#' the Scorch model graph.
#'
#' @param scorch_model A \code{scorch_model} object created by
#'   \code{\link{initiate_scorch}}.
#'
#' @param name A single character string giving a unique name for this
#'   node in the computation graph.
#'
#' @param inputs Character vector of upstream node names. If \code{NULL}
#'   (default), resolved automatically (last node or sole input).
#'
#' @param p Numeric. Dropout probability (default 0.5).
#'
#' @param ... Additional arguments passed to \code{torch::nn_dropout()}.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' This is equivalent to calling
#' \code{scorch_layer(model, name, "dropout", inputs, p = 0.5)} but
#' provides a more readable API for a common operation.
#'
#' @examples
#' \dontrun{
#' model <- model |>
#'   scorch_layer("fc1", "linear", in_features = 32, out_features = 16) |>
#'   scorch_layer("act1", "relu") |>
#'   scorch_dropout("drop1", p = 0.3)
#' }
#'
#' @family model construction
#'
#' @export

scorch_dropout <- function(scorch_model,
                           name,
                           inputs = NULL,
                           p = 0.5,
                           ...) {

  do_mod <- torch::nn_dropout(p = p, ...)

  #- Resolve inputs when not specified explicitly.

  if (is.null(inputs)) {

    if (nrow(scorch_model$graph) == 0) {

      if (length(scorch_model$inputs) != 1) {

        stop("Must specify 'inputs' when multiple inputs exist.",
             call. = FALSE)
      }

      inputs <- scorch_model$inputs

    } else {

      inputs <- tail(scorch_model$graph$name, 1)
    }
  }

  #- Append to graph.

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(do_mod),
    inputs = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
