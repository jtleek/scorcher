#=== SCORCH INPUT ==============================================================

#-------------------------------------------------------------------------------
# NOTES:
#
#   1. This is for bookkeeping
#   2. It's more tensorflow-y, but it made sense in my head
#-------------------------------------------------------------------------------

#' Register an Input Node in a Scorch Model
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for the input (character).
#'
#' @return The updated `scorch_model`.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(mtcars$wt, mtcars$mpg)
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("wt")
#'
#' print(sm$inputs)
#'
#' @export

scorch_input <- function(

    scorch_model,
    name) {

    if (name %in% scorch_model$inputs) {

        stop("Input '", name, "' already exists.", call. = FALSE)
    }

    scorch_model$inputs <- c(scorch_model$inputs, name)

    return(scorch_model)
}

#=== END =======================================================================
