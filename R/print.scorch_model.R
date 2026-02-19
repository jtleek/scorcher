#===============================================================================
# PRINT METHOD FOR SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Print a Scorch Model
#'
#' @description
#' Displays a summary of a \code{scorch_model} object, including its
#' input and output nodes and a table of the graph architecture. If
#' \code{detailed = TRUE} and the model has been compiled, weight
#' dimensions are also shown.
#'
#' @param x A \code{scorch_model} object.
#'
#' @param detailed Logical. If \code{TRUE}, show weight dimensions for
#'   each layer (requires the model to be compiled). Default
#'   \code{FALSE}.
#'
#' @param ... Additional arguments (currently unused).
#'
#' @returns \code{x}, invisibly.
#'
#' @examples
#' \dontrun{
#' print(model)
#' print(model, detailed = TRUE)
#' }
#'
#' @family model construction
#'
#' @export

print.scorch_model <- function(x,
                               detailed = FALSE,
                               ...) {

  cat("<< Scorch Model >>\n")

  cat(" Inputs : ", paste(x$inputs, collapse = ", "), "\n")

  cat(" Outputs: ", paste(x$outputs, collapse = ", "), "\n")

  cat(" Compiled:", x$compiled, "\n\n")

  #- If graph is empty, nothing more to show.

  if (nrow(x$graph) == 0) {

    cat(" (no layers added yet)\n")

    return(invisible(x))
  }

  #- If compiled, show module types from nn_model.

  if (x$compiled) {

    df <- x$graph |>

      dplyr::mutate(

        module_type = purrr::map_chr(name, ~ class(x$nn_model[[.x]])[1]),
        inputs      = purrr::map_chr(inputs, ~ paste(.x, collapse = ", "))
      )

    if (detailed) {

      df <- df |>

        dplyr::mutate(

          dims = purrr::map_chr(name, function(nm) {

            mod <- x$nn_model[[nm]]

            if (!is.null(mod$weight) &&
                inherits(mod$weight, "torch_tensor")) {

              paste(mod$weight$size(), collapse = "x")

            } else {

              ""
            }
          })
        ) |>

        dplyr::select(name, inputs, module_type, dims)

    } else {

      df <- df |>

        dplyr::select(name, inputs, module_type)
    }

  } else {

    #- Not yet compiled: show graph structure only.

    df <- x$graph |>

      dplyr::mutate(
        inputs = purrr::map_chr(inputs, ~ paste(.x, collapse = ", "))
      ) |>

      dplyr::select(name, inputs)
  }

  print(df)

  invisible(x)
}

#=== END =======================================================================
