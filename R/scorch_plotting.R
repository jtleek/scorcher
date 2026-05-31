#===============================================================================
# STATIC PLOTTING HELPERS
#===============================================================================

#' Return Scorch Graph Plot Data
#'
#' @param scorch_model A \code{scorch_model} object.
#'
#' @returns A list with \code{nodes} and \code{edges} tibbles.
#'
#' @family model construction
#'
#' @export
scorch_graph_data <- function(scorch_model) {
  scorch_model <- scorch_check_model(scorch_model)
  graph <- scorch_model$graph

  input_nodes <- tibble::tibble(
    name = scorch_model$inputs,
    node_type = "input",
    constructor = "input",
    param_count = 0,
    trainable = FALSE,
    order = seq_along(scorch_model$inputs)
  )

  graph_nodes <- tibble::tibble(
    name = graph$name,
    node_type = graph$node_type,
    constructor = graph$constructor,
    param_count = graph$param_count,
    trainable = graph$trainable,
    order = length(scorch_model$inputs) + seq_len(nrow(graph))
  )

  nodes <- rbind(input_nodes, graph_nodes)
  nodes$is_output <- nodes$name %in% scorch_model$outputs

  edges <- if (nrow(graph) == 0) {
    tibble::tibble(from = character(), to = character())
  } else {
    do.call(rbind, lapply(seq_len(nrow(graph)), function(i) {
      tibble::tibble(from = graph$inputs[[i]], to = graph$name[i])
    }))
  }

  list(nodes = nodes, edges = edges)
}

#' Autoplot a Scorch Model
#'
#' @param object A \code{scorch_model} object.
#' @param type Plot type: \code{"architecture"} or \code{"parameters"}.
#' @param ... Unused.
#'
#' @returns A ggplot object.
#'
#' @export
autoplot.scorch_model <- function(object,
                                  type = c("architecture", "parameters"),
                                  ...) {
  type <- match.arg(type)
  gd <- scorch_graph_data(object)
  nodes <- gd$nodes

  if (type == "parameters") {
    graph_nodes <- nodes[nodes$node_type != "input", , drop = FALSE]
    return(
      ggplot2::ggplot(graph_nodes,
                      ggplot2::aes(x = stats::reorder(name, param_count),
                                   y = param_count,
                                   fill = trainable)) +
        ggplot2::geom_col() +
        ggplot2::coord_flip() +
        ggplot2::labs(x = NULL, y = "Parameters", fill = "Trainable")
    )
  }

  edges <- gd$edges
  nodes$x <- ifelse(nodes$node_type == "input", 0, 1)
  nodes$y <- rev(seq_len(nrow(nodes)))

  edges <- merge(edges, nodes[c("name", "x", "y")],
                 by.x = "from", by.y = "name", all.x = TRUE)
  names(edges)[names(edges) %in% c("x", "y")] <- c("x_from", "y_from")
  edges <- merge(edges, nodes[c("name", "x", "y")],
                 by.x = "to", by.y = "name", all.x = TRUE)
  names(edges)[names(edges) %in% c("x", "y")] <- c("x_to", "y_to")

  ggplot2::ggplot() +
    ggplot2::geom_segment(
      data = edges,
      ggplot2::aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
      arrow = grid::arrow(length = grid::unit(0.12, "inches")),
      linewidth = 0.35
    ) +
    ggplot2::geom_point(
      data = nodes,
      ggplot2::aes(x = x, y = y, shape = node_type, color = is_output),
      size = 3
    ) +
    ggplot2::geom_text(
      data = nodes,
      ggplot2::aes(x = x + 0.04, y = y, label = name),
      hjust = 0,
      size = 3
    ) +
    ggplot2::scale_x_continuous(limits = c(-0.1, 1.8), breaks = NULL) +
    ggplot2::scale_y_continuous(breaks = NULL) +
    ggplot2::labs(x = NULL, y = NULL, shape = "Node type",
                  color = "Output") +
    ggplot2::theme_minimal()
}

#' Autoplot a Scorch Run
#'
#' @param object A \code{scorch_run} object.
#' @param type Plot type: \code{"history"}, \code{"audit"}, or
#'   \code{"timeline"}.
#' @param ... Unused.
#'
#' @returns A ggplot object.
#'
#' @export
autoplot.scorch_run <- function(object,
                                type = c("history", "audit", "timeline"),
                                ...) {
  if (!inherits(object, "scorch_run")) {
    stop("`object` must be a scorch_run.", call. = FALSE)
  }

  type <- match.arg(type)

  if (type == "history") {
    if (is.null(object$history)) {
      stop("This run has no training history.", call. = FALSE)
    }

    return(
      ggplot2::ggplot(object$history,
                      ggplot2::aes(x = epoch, y = loss)) +
        ggplot2::geom_line() +
        ggplot2::geom_point(size = 1.5) +
        ggplot2::labs(x = "Epoch", y = "Loss")
    )
  }

  if (type == "audit") {
    audit <- scorch_audit(object)
    return(
      ggplot2::ggplot(audit,
                      ggplot2::aes(x = stats::reorder(check, status),
                                   fill = status)) +
        ggplot2::geom_bar() +
        ggplot2::coord_flip() +
        ggplot2::labs(x = NULL, y = "Checks", fill = "Status")
    )
  }

  timeline <- tibble::tibble(
    event = "snapshot",
    timestamp = as.POSIXct(object$timestamp)
  )

  ggplot2::ggplot(timeline, ggplot2::aes(x = timestamp, y = event)) +
    ggplot2::geom_point(size = 3) +
    ggplot2::labs(x = NULL, y = NULL)
}
