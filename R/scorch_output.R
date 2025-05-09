#=== SCORCH OUTPUT =============================================================

#-------------------------------------------------------------------------------
# NOTES:
#
#   1. Also for bookkeeping
#   2. Also tensorflow-y
#-------------------------------------------------------------------------------

#' Mark Output Nodes in a Scorch Model
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param outputs Character vector of node names to mark as outputs.
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
#'   scorch_input("wt") |>
#'
#'   scorch_output("wt")
#'
#' print(sm$outputs)
#'
#' @export

scorch_output <- function(

    scorch_model,
    outputs) {

    scorch_model$outputs <- outputs

    return(scorch_model)
}


#=== END =======================================================================
