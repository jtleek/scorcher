#===============================================================================
# FUNCTION TO INITIATE A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Initiate a Scorch Model
#'
#' @description
#' Creates an empty Scorch model object with a graph-based architecture.
#' This is the first step in the scorcher pipeline: it initializes the
#' data structure that subsequent functions (\code{\link{scorch_input}},
#' \code{\link{scorch_layer}}, \code{\link{scorch_output}},
#' \code{\link{compile_scorch}}) build upon.
#'
#' @param dl Optional \code{torch::dataloader} to attach to the model.
#'   If \code{NULL} (default), no dataloader is attached and one can be
#'   added later. If provided, it must inherit from \code{"dataloader"}.
#'
#' @returns A \code{scorch_model} object (a list with class
#'   \code{"scorch_model"}) containing the following fields:
#'
#'   \describe{
#'     \item{\code{graph}}{A tibble with columns \code{name} (character),
#'       \code{module} (list of \code{nn_module} objects), and \code{inputs}
#'       (list of character vectors). Starts empty; rows are added by
#'       \code{\link{scorch_layer}} and related functions.}
#'     \item{\code{inputs}}{Character vector of input node names, set by
#'       \code{\link{scorch_input}}.}
#'     \item{\code{outputs}}{Character vector of output node names, set by
#'       \code{\link{scorch_output}}.}
#'     \item{\code{compiled}}{Logical. \code{FALSE} until
#'       \code{\link{compile_scorch}} is called.}
#'     \item{\code{nn_model}}{The compiled \code{torch::nn_module}, or
#'       \code{NULL} before compilation.}
#'     \item{\code{optimizer}}{The optimizer object, or \code{NULL} before
#'       compilation.}
#'     \item{\code{loss_fn}}{The loss function (or named list of loss
#'       functions), or \code{NULL} before compilation.}
#'     \item{\code{dl}}{The attached \code{torch::dataloader}, or
#'       \code{NULL} if none was provided.}
#'   }
#'
#' @details
#' The Scorch model uses a graph-based architecture stored as a tibble.
#' Each row represents a named node in the computation graph, holding its
#' \code{nn_module} and a character vector of upstream node names. This
#' allows scorcher to represent complex architectures including multi-input
#' models, branching paths, skip connections, and multi-output heads.
#'
#' A typical pipeline looks like:
#'
#' \preformatted{
#' model <- initiate_scorch(dl) |>
#'   scorch_input("input") |>
#'   scorch_layer("fc1", "linear", in_features = 10, out_features = 32) |>
#'   scorch_layer("act1", "relu") |>
#'   scorch_layer("fc2", "linear", in_features = 32, out_features = 1) |>
#'   scorch_output("fc2") |>
#'   compile_scorch()
#' }
#'
#' @examples
#' \dontrun{
#' # With a dataloader
#' model <- initiate_scorch(dl)
#'
#' # Without a dataloader (attach later)
#' model <- initiate_scorch()
#' }
#'
#' @family model construction
#'
#' @export

initiate_scorch <- function(dl = NULL) {

  #- Create the base structure for a scorch_model object.
  #- The graph tibble will hold one row per named node, with columns:
  #-   name:    Character -- unique node identifier
  #-   module:  List -- the nn_module for this node
  #-   inputs:  List of character vectors -- upstream node names

  sm <- list(

    graph     = tibble::tibble(name    = character(),
                               module  = list(),
                               inputs  = list()),

    inputs    = character(),
    outputs   = character(),
    compiled  = FALSE,
    nn_model  = NULL,
    optimizer = NULL,
    loss_fn   = NULL,
    dl        = NULL
  )

  class(sm) <- "scorch_model"

  #- If a dataloader is provided, validate and attach it.

  if (!is.null(dl)) {

    if (!inherits(dl, "dataloader")) {

      stop("`dl` must be a torch::dataloader.", call. = FALSE)
    }

    sm$dl <- dl
  }

  sm
}

#=== END =======================================================================
