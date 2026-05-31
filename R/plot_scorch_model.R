#===============================================================================
# PLOT METHOD FOR SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Plot a Scorch Model Architecture
#'
#' @description
#' Renders a publication-ready Scorch model architecture diagram using Graphviz
#' via the DiagrammeR package. The diagram shows inputs, layers, outputs,
#' upstream connections, parameter counts, and dimensions that can be inferred
#' from layer arguments.
#'
#' @param scorch_model A \code{scorch_model} object.
#'
#' @param detail How much to show in each node. \code{"simple"} shows only the
#'   layer type. \code{"full"} adds the node name, upstream inputs, parameter
#'   counts, and dimensions that can be inferred from layer arguments.
#'
#' @param detailed Deprecated logical alias for \code{detail}. Use
#'   \code{detail = "full"} or \code{detail = "simple"}.
#'
#' @param rankdir Graphviz rank direction. Defaults to \code{"TB"} for
#'   top-to-bottom diagrams. Use \code{"LR"} for left-to-right diagrams.
#'
#' @param input_shapes Optional named list or named character vector of input
#'   shapes to display, for example \code{list(features = "batch x 4")}.
#'
#' @param ... Additional arguments passed from \code{plot()} to
#'   \code{plot_scorch_model()}.
#'
#' @returns A \code{DiagrammeR} htmlwidget object.
#'
#' @examples
#' \dontrun{
#' plot(model)
#' plot(model, detail = "simple")
#' plot(model, detail = "full", input_shapes = list(features = "batch x 4"))
#' plot_scorch_model(model, rankdir = "LR")
#' }
#'
#' @family model construction
#'
#' @export

plot_scorch_model <- function(scorch_model,
                              detail = c("full", "simple"),
                              detailed = NULL,
                              rankdir = c("TB", "LR"),
                              input_shapes = NULL,
                              ...) {

  scorch_model <- scorch_check_model(scorch_model)
  detail <- match.arg(detail)
  if (!is.null(detailed)) {
    detail <- if (isTRUE(detailed)) "full" else "simple"
  }
  rankdir <- match.arg(rankdir)

  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("Package 'DiagrammeR' is required for architecture diagrams. ",
         "Install it with install.packages('DiagrammeR').",
         call. = FALSE)
  }

  dot <- scorch_architecture_dot(
    scorch_model,
    detail = detail,
    rankdir = rankdir,
    input_shapes = input_shapes
  )

  DiagrammeR::grViz(dot)
}

#' @export
plot.scorch_model <- function(x, ...) {
  plot_scorch_model(x, ...)
}

scorch_architecture_dot <- function(scorch_model,
                                    detail = "full",
                                    rankdir = "LR",
                                    input_shapes = NULL) {

  graph <- scorch_model$graph
  inputs <- scorch_model$inputs
  outputs <- scorch_model$outputs

  node_names <- c(inputs, graph$name)
  node_ids <- stats::setNames(paste0("n", seq_along(node_names)), node_names)

  input_defs <- vapply(inputs, function(nm) {
    shape <- scorch_shape_label(input_shapes[[nm]] %||% NULL)
    details <- if (detail == "full") {
      c(
        paste("name:", nm),
        if (nzchar(shape)) paste("shape:", shape)
      )
    } else {
      character()
    }
      scorch_dot_node(
        id = node_ids[[nm]],
        header = "Input",
        details = details
    )
  }, character(1))

  graph_defs <- if (nrow(graph) == 0) {
    character()
  } else {
    vapply(seq_len(nrow(graph)), function(i) {
      nm <- graph$name[i]
      constructor <- graph$constructor[i]
      args <- graph$args[[i]]
      param_count <- graph$param_count[i]
      is_output <- nm %in% outputs

      details <- if (detail == "full") {
        c(
          paste("name:", nm),
          scorch_dimension_label(constructor, args),
          paste("from:", paste(graph$inputs[[i]], collapse = ", ")),
          if (!is.na(param_count)) {
            paste("params:", format(param_count, big.mark = ","))
          }
        )
      } else {
        character()
      }

      scorch_dot_node(
        id = node_ids[[nm]],
        header = scorch_pretty_layer_type(constructor),
        details = details,
        header_fill = if (is_output) "#14532D" else "#020617",
        border = if (is_output) "#14532D" else "#020617"
      )
    }, character(1))
  }

  edge_defs <- if (nrow(graph) == 0) {
    character()
  } else {
    unlist(lapply(seq_len(nrow(graph)), function(i) {
      to <- graph$name[i]
      froms <- graph$inputs[[i]]
      vapply(froms, function(from) {
        sprintf(
          "%s -> %s",
          node_ids[[from]], node_ids[[to]]
        )
      }, character(1))
    }), use.names = FALSE)
  }

  paste(
    "digraph scorch_model {",
    sprintf("graph [rankdir=%s, bgcolor=\"transparent\", pad=0.35, nodesep=0.65, ranksep=0.95, splines=ortho];", rankdir),
    "node [fontname=\"Helvetica\", fontsize=12, margin=0, shape=plain, color=\"#020617\", fontcolor=\"#0F172A\"];",
    "edge [fontname=\"Helvetica\", fontsize=9, color=\"#94A3B8\", arrowsize=0.65, penwidth=1.15];",
    paste(c(input_defs, graph_defs), collapse = ";\n"),
    ";",
    paste(edge_defs, collapse = ";\n"),
    ";",
    "}",
    sep = "\n"
  )
}

scorch_dot_node <- function(id,
                            header,
                            details = character(),
                            header_fill = "#020617",
                            border = "#020617") {
  header <- scorch_html_escape(header)
  details <- details[!is.na(details) & nzchar(details)]
  details <- vapply(details, scorch_html_escape, character(1))

  detail_rows <- if (length(details) == 0) {
    ""
  } else {
    paste0(
      "<TR><TD BGCOLOR=\"white\"><FONT POINT-SIZE=\"10\">",
      details,
      "</FONT></TD></TR>",
      collapse = ""
    )
  }

  table <- paste0(
    "<TABLE BORDER=\"1\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"8\" COLOR=\"",
    border,
    "\">",
    "<TR><TD BGCOLOR=\"", header_fill, "\"><FONT COLOR=\"white\"><B>",
    header,
    "</B></FONT></TD></TR>",
    detail_rows,
    "</TABLE>"
  )

  sprintf(
    "%s [label=<%s>]",
    id, table
  )
}

scorch_dot_escape <- function(x) {
  x <- as.character(x)
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub('"', '\\"', x)
  x
}

scorch_html_escape <- function(x) {
  x <- as.character(x)
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  x
}

scorch_pretty_layer_type <- function(x) {
  x <- scorch_sanitize_prefix(x)
  words <- strsplit(x, "_", fixed = TRUE)[[1]]
  paste0(toupper(substring(words, 1, 1)), substring(words, 2),
         collapse = " ")
}

scorch_shape_label <- function(x) {
  if (is.null(x)) return("")
  if (length(x) > 1) return(paste(x, collapse = " x "))
  as.character(x)
}

scorch_dimension_label <- function(constructor, args) {
  constructor <- scorch_sanitize_prefix(constructor)

  if (constructor == "linear") {
    return(paste0("dims: ", args$in_features, " -> ", args$out_features))
  }

  if (constructor == "embedding") {
    return(paste0("dims: vocab ", args$num_embeddings,
                  " -> ", args$embedding_dim))
  }

  if (constructor %in% c("conv1d", "conv2d", "conv3d")) {
    return(paste0("channels: ", args$in_channels, " -> ", args$out_channels,
                  "; kernel: ", paste(args$kernel_size, collapse = " x ")))
  }

  if (constructor %in% c("conv_transpose1d", "conv_transpose2d",
                         "conv_transpose3d")) {
    return(paste0("channels: ", args$in_channels, " -> ", args$out_channels,
                  "; kernel: ", paste(args$kernel_size, collapse = " x ")))
  }

  if (constructor == "multihead_attention") {
    return(paste0("dims: embed ", args$embed_dim,
                  "; heads: ", args$num_heads,
                  if (isTRUE(args$causal)) "; causal" else ""))
  }

  if (constructor == "layer_norm") {
    return(paste0("norm shape: ", paste(args$normalized_shape,
                                        collapse = " x ")))
  }

  if (constructor == "dropout") {
    return(paste0("p: ", args$p %||% 0.5))
  }

  if (constructor == "concat") {
    return(paste0("concat dim: ", args$dim))
  }

  if (constructor == "add_skip") {
    return("dims: unchanged")
  }

  ""
}

scorch_node_fill <- function(node_type) {
  switch(
    node_type,
    layer = "#FFF7ED",
    block = "#F3E8FF",
    "function" = "#ECFEFF",
    "#F8FAFC"
  )
}

scorch_node_border <- function(node_type) {
  switch(
    node_type,
    layer = "#C2410C",
    block = "#7E22CE",
    "function" = "#0E7490",
    "#334155"
  )
}

#=== END =======================================================================
