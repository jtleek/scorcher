#===============================================================================
# PLOT METHOD FOR SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Plot a Scorch Model Architecture
#'
#' @description
#' Renders the Scorch model computation graph as a directed graph
#' using Graphviz via the DiagrammeR package. Input nodes are shown
#' as ovals, output nodes as double circles, and all other nodes as
#' boxes.
#'
#' @param scorch_model A compiled \code{scorch_model} object.
#'
#' @param detailed Logical. If \code{TRUE}, show weight dimensions
#'   next to each node label. Default \code{FALSE}.
#'
#' @returns A \code{DiagrammeR} htmlwidget object (rendered
#'   automatically in interactive sessions).
#'
#' @details
#' Requires the DiagrammeR package to be installed. If it is not
#' available, the function raises an informative error.
#'
#' Nodes are laid out left-to-right (\code{rankdir=LR}). Edges
#' follow the \code{inputs} declared in the graph tibble.
#'
#' @examples
#' \dontrun{
#' plot_scorch_model(model)
#' plot_scorch_model(model, detailed = TRUE)
#' }
#'
#' @family model construction
#'
#' @export

plot_scorch_model <- function(scorch_model,
                              detailed = FALSE) {

  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {

    stop("Package 'DiagrammeR' is required for plot_scorch_model(). ",
         "Install it with install.packages('DiagrammeR').",
         call. = FALSE)
  }

  nodes   <- scorch_model$graph$name
  inputs  <- scorch_model$inputs
  outputs <- scorch_model$outputs

  #- Node definitions.

  node_defs <- c(

    #- Input nodes (ovals).

    vapply(inputs, function(nm) {

      sprintf('%s [shape=oval, style=filled, fillcolor="lightblue", label="%s"]',
              nm, nm)
    }, ""),

    #- Internal and output nodes.

    vapply(nodes, function(nm) {

      mod  <- scorch_model$nn_model[[nm]]

      type <- class(mod)[1]

      dims_lbl <- ""

      if (detailed &&
          !is.null(mod$weight) &&
          inherits(mod$weight, "torch_tensor")) {

        dims_lbl <- paste0("\n(",
                           paste(mod$weight$size(), collapse = "x"),
                           ")")
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

  #- Edge definitions.

  edge_defs <- unlist(lapply(seq_len(nrow(scorch_model$graph)), function(i) {

    to    <- scorch_model$graph$name[i]

    froms <- scorch_model$graph$inputs[[i]]

    vapply(froms, function(fr) sprintf("%s -> %s", fr, to), "")
  }))

  #- Assemble DOT string and render.

  dot <- paste(
    "digraph G {",
    "rankdir=LR;",
    paste(node_defs, collapse = ";\n"), ";",
    paste(edge_defs, collapse = ";\n"), ";",
    "}"
  )

  DiagrammeR::grViz(dot)
}

#=== END =======================================================================
