#===============================================================================
# FUNCTION TO ADD A CONCATENATION NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Concatenation Node to a Scorch Model
#'
#' @description
#' Concatenates the outputs of two or more upstream nodes along a
#' specified dimension. Commonly used to merge parallel branches
#' in multi-input or late-fusion architectures.
#'
#' @param scorch_model A \code{scorch_model} object created by
#'   \code{\link{initiate_scorch}}.
#'
#' @param name A single character string giving a unique name for this
#'   node in the computation graph.
#'
#' @param inputs Character vector of two or more upstream node names
#'   whose outputs will be concatenated.
#'
#' @param dim Integer. Dimension along which to concatenate (default 1,
#'   which is typically the feature dimension after the batch dimension).
#'   Use \code{dim = 2} when concatenating feature vectors.
#'
#' @returns The updated \code{scorch_model} with a new row appended to
#'   its \code{graph} tibble.
#'
#' @details
#' The node is implemented as a lightweight \code{torch::nn_module}
#' that calls \code{torch::torch_cat()} on its inputs. It has no
#' learnable parameters.
#'
#' @examples
#' \dontrun{
#' # Merge two branches for late fusion
#' model <- model |>
#'   scorch_concat("merged", inputs = c("branch_a", "branch_b"), dim = 2)
#' }
#'
#' @family model construction
#'
#' @export

scorch_concat <- function(scorch_model,
                          name,
                          inputs,
                          dim = 1) {

  #- Build a lightweight module that concatenates its inputs.

  concat_mod <- torch::nn_module(

    initialize = function() {},

    forward = function(...) {

      torch::torch_cat(list(...), dim = dim)
    }
  )()

  #- Append to graph.

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(concat_mod),
    inputs = list(inputs)
  )

  scorch_model
}

#=== END =======================================================================
