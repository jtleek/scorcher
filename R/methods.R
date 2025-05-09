#-------------------------------------------------------------------------------
# NOTES:
#
# This is not nearly as nice as the print method you already had, but I am just
# showing here that we can exploit storing the architecture as a graph in a
# tibble, so maybe we can add more info later?
#-------------------------------------------------------------------------------

#' Print a Scorch Model Summary
#'
#' @param x A `scorch_model` object.
#'
#' @param detailed Logical; if TRUE, include parameter shapes.
#'
#' @param ... Not used.
#'
#' @return Invisibly returns the original object.
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
#'   scorch_layer("h1", "linear", in_features = 1, out_features = 4) |>
#'
#'   scorch_output("h1") |>
#'
#'   compile_scorch()
#'
#' print(sm)
#'
#' @export

print.scorch_model <- function(

    x,
    detailed = FALSE,
    ...) {

    cat("<< Scorch Model >>\n")

    cat(" Inputs : ", paste(x$inputs, collapse = ", "), "\n")

    cat(" Outputs: ", paste(x$outputs, collapse = ", "), "\n\n")

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

                })) |>

            dplyr::select(name, inputs, module_type, dims)

    } else {

        df <- df |>

            dplyr::select(name, inputs, module_type)
    }

    print(df)

    invisible(x)
}

#' Plot a Scorch Model Architecture
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param detailed Logical; if TRUE, include parameter shapes.
#'
#' @return A `grViz` graph object.
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
#'   scorch_layer("h1", "linear", in_features = 1, out_features = 4) |>
#'
#'   scorch_output("h1") |>
#'
#'   compile_scorch()
#'
#' plot_scorch_model(sm)
#'
#' @import DiagrammeR
#'
#' @export

plot_scorch_model <- function(

    scorch_model,
    detailed = FALSE) {

    nodes <- scorch_model$graph$name

    inputs <- scorch_model$inputs

    outputs <- scorch_model$outputs

    #- Node definitions

    node_defs <- c(

        #- Input nodes

        vapply(inputs, function(nm) {

            sprintf('%s [shape=oval, style=filled, fillcolor="lightblue", label="%s"]', nm, nm)
        }, ""),

        #- Internal/output nodes

        vapply(nodes, function(nm) {

            mod <- scorch_model$nn_model[[nm]]

            type <- class(mod)[1]

            dims_lbl <- ""

            if (detailed && !is.null(mod$weight) && inherits(mod$weight, "torch_tensor")) {

                dims_lbl <- paste0("\n(", paste(mod$weight$size(), collapse="x"), ")")
            }

            shape <- if (nm %in% outputs) "doublecircle" else "box"

            label <- if (detailed) {

                paste0(nm, "\n", type, dims_lbl)

            } else {

                paste0(nm, "\n", type)
            }

            sprintf('%s [shape=%s, label="%s"]', nm, shape, label)

        }, "")
    )

    #- Edge definitions

    edge_defs <- unlist(lapply(seq_len(nrow(scorch_model$graph)), function(i) {

        to    <- scorch_model$graph$name[i]

        froms <- scorch_model$graph$inputs[[i]]

        vapply(froms, function(fr) sprintf("%s -> %s", fr, to), "")
    }))

    dot <- paste(

        "digraph G {",
        "rankdir=LR;",
        paste(node_defs, collapse = ";\n"), ";",
        paste(edge_defs, collapse = ";\n"), ";",
        "}"
    )

    DiagrammeR::grViz(dot)
}
