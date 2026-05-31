#===============================================================================
# FUNCTION TO ADD A REUSABLE BLOCK NODE TO A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add a Reusable Block to a Scorch Model
#'
#' @description
#' Adds a reusable block to the Scorch model graph. The recommended use is a
#' graph block: provide unquoted layer calls and let Scorcher expand the block
#' into standard graph nodes. This keeps diffusion, transformer, and GPT
#' examples accessible without requiring users to define module classes.
#'
#' @param scorch_model A \code{scorch_model} object created by
#'   \code{\link{initiate_scorch}}.
#'
#' @param block Optional advanced escape hatch: a module instance, module
#'   constructor, or function returning a module. If \code{layers} is omitted
#'   and \code{block} is a list, it is treated as \code{layers}.
#'
#' @param inputs Character vector of upstream node names. If \code{NULL}
#'   (default), resolved automatically.
#'
#' @param ... In graph block mode, additional unquoted layer calls, for example
#'   \code{gelu()} after \code{linear(...)}. In advanced module mode, arguments
#'   passed to \code{block}.
#'
#' @param .name Optional string or unquoted name for the block output node.
#'
#' @param .from Optional string, unquoted reference, or \code{c(...)} of
#'   upstream node references.
#'
#' @param layers Optional compatibility form for layer specifications. Each
#'   specification may be an unquoted call captured with \code{quote()}, a named
#'   or unnamed list whose first unnamed value, \code{layer}, or \code{type}
#'   gives the layer name, or a character layer name.
#'
#' @param residual If \code{TRUE}, add the block input to the block output with
#'   \code{\link{scorch_add_skip}}.
#'
#' @param repeats Number of times to repeat \code{layers}.
#'
#' @returns The updated \code{scorch_model}. In graph block mode, the block
#'   output node is named by \code{.name}.
#'
#' @examples
#' \dontrun{
#' model <- initiate_scorch() |>
#'   scorch_input(x) |>
#'   scorch_layer(linear, in_features = 2, out_features = 32, .name = hidden) |>
#'   scorch_block(
#'     linear(in_features = 32, out_features = 32),
#'     gelu(),
#'     residual = TRUE,
#'     .from = hidden,
#'     .name = residual_block
#'   ) |>
#'   scorch_output(residual_block)
#' }
#'
#' @family model construction
#'
#' @export

scorch_block <- function(scorch_model,
                         block = NULL,
                         inputs = NULL,
                         ...,
                         .name = NULL,
                         .from = NULL,
                         layers = NULL,
                         residual = FALSE,
                         repeats = 1L) {

  scorch_model <- scorch_check_model(scorch_model)
  block_expr <- if (missing(block)) NULL else substitute(block)
  inputs_expr <- if (missing(inputs)) NULL else substitute(inputs)
  dots_expr <- as.list(substitute(list(...)))[-1]
  env <- parent.frame()
  graph_inputs_expr <- inputs_expr

  if (is.null(layers) && !is.null(block_expr) &&
      scorch_is_block_layer_call(block_expr)) {
    layers <- list(block_expr)
    if (!is.null(inputs_expr) && scorch_is_block_layer_call(inputs_expr)) {
      layers <- c(layers, list(inputs_expr))
      graph_inputs_expr <- NULL
      inputs <- NULL
    }
    layers <- c(layers, dots_expr)
    block <- NULL
  }

  if (is.null(layers) && is.list(block) && !inherits(block, "nn_module")) {
    layers <- block
    block <- NULL
  }

  if (!is.null(layers)) {
    return(scorch_block_graph(
      scorch_model = scorch_model,
      layers = layers,
      inputs = inputs,
      .name = if (missing(.name)) NULL else substitute(.name),
      .from = if (missing(.from)) NULL else substitute(.from),
      inputs_expr = graph_inputs_expr,
      residual = residual,
      repeats = repeats,
      env = env
    ))
  }

  scorch_block_module(
    scorch_model = scorch_model,
    block = block,
    inputs = inputs,
    ...,
    .name = if (missing(.name)) NULL else substitute(.name),
    .from = if (missing(.from)) NULL else substitute(.from),
    inputs_expr = inputs_expr,
    block_expr = block_expr
  )
}

scorch_block_graph <- function(scorch_model,
                               layers,
                               inputs,
                               .name,
                               .from,
                               inputs_expr,
                               residual = FALSE,
                               repeats = 1L,
                               env = parent.frame()) {

  if (!is.list(layers) || length(layers) == 0) {
    stop("`layers` must be a non-empty list of layer specifications.",
         call. = FALSE)
  }

  repeats <- as.integer(repeats)
  if (length(repeats) != 1 || is.na(repeats) || repeats < 1L) {
    stop("`repeats` must be a positive integer.", call. = FALSE)
  }

  block_inputs <- scorch_resolve_inputs(
    scorch_model,
    inputs = if (is.null(inputs_expr)) NULL else
      scorch_parse_refs_expr(inputs_expr, arg = "inputs"),
    from = if (is.null(.from)) NULL else
      scorch_parse_refs_expr(.from, arg = ".from")
  )

  if (length(block_inputs) != 1) {
    stop("Graph blocks currently require exactly one upstream input.",
         call. = FALSE)
  }

  node_name <- scorch_prepare_node_name(
    scorch_model,
    explicit_expr = .name,
    legacy_expr = NULL,
    auto_prefix = "block"
  )
  scorch_model <- node_name$model
  block_name <- node_name$name

  current <- block_inputs

  for (repeat_idx in seq_len(repeats)) {
    repeat_input <- current

    for (layer_idx in seq_along(layers)) {
      spec <- scorch_block_layer_spec(layers[[layer_idx]], env = env)
      is_last <- repeat_idx == repeats && layer_idx == length(layers)
      node_name_i <- if (is_last && !isTRUE(residual)) {
        block_name
      } else {
        paste(block_name, repeat_idx, layer_idx, spec$constructor, sep = "_")
      }

      layer_inputs <- current
      if (isTRUE(spec$self_attention)) {
        layer_inputs <- rep(current, 3)
      }

      scorch_model <- do.call(
        scorch_layer,
        c(
          list(
            scorch_model = scorch_model,
            name = node_name_i,
            layer_fn = spec$layer,
            inputs = layer_inputs
          ),
          spec$args
        )
      )

      current <- node_name_i
    }

    if (isTRUE(residual)) {
      skip_name <- if (repeat_idx == repeats) {
        block_name
      } else {
        paste(block_name, repeat_idx, "residual", sep = "_")
      }

      scorch_model <- do.call(
        scorch_add_skip,
        list(
          scorch_model = scorch_model,
          name = skip_name,
          inputs = c(repeat_input, current)
        )
      )

      current <- skip_name
    }
  }

  scorch_model
}

scorch_is_block_layer_call <- function(expr) {
  if (!is.call(expr) || length(expr) < 1L) {
    return(FALSE)
  }

  constructor <- scorch_constructor_name(expr, fallback = "")
  if (!nzchar(constructor)) {
    return(FALSE)
  }

  fn_name <- constructor
  if (!grepl("^nn_", fn_name)) {
    fn_name <- paste0("nn_", fn_name)
  }

  exists(fn_name, envir = asNamespace("torch"), mode = "function")
}

scorch_block_layer_spec <- function(spec, env = parent.frame()) {
  if (is.call(spec)) {
    layer <- scorch_constructor_name(spec, fallback = "layer")
    args <- lapply(as.list(spec[-1]), eval, envir = env)
    return(scorch_block_layer_spec_from_parts(layer = layer, args = args))
  }

  if (is.character(spec) && length(spec) == 1) {
    return(list(layer = spec, constructor = spec, args = list(),
                self_attention = FALSE))
  }

  if (is.function(spec)) {
    constructor <- scorch_constructor_name(substitute(spec), fallback = "layer")
    return(list(layer = spec, constructor = constructor, args = list(),
                self_attention = FALSE))
  }

  if (!is.list(spec) || length(spec) == 0) {
    stop("Each block layer must be a string, function, or list.",
         call. = FALSE)
  }

  layer <- spec$layer %||% spec$type
  args <- spec

  unnamed <- names(args) %||% rep("", length(args))
  first_unnamed <- which(!nzchar(unnamed))[1]
  if (is.null(layer) && !is.na(first_unnamed)) {
    layer <- args[[first_unnamed]]
    args[[first_unnamed]] <- NULL
  }

  args$layer <- NULL
  args$type <- NULL

  scorch_block_layer_spec_from_parts(layer = layer, args = args)
}

scorch_block_layer_spec_from_parts <- function(layer, args) {
  self_attention <- isTRUE(args$self_attention) || isTRUE(args$self)
  args$self_attention <- NULL
  args$self <- NULL

  if (is.null(layer)) {
    stop("Each block layer specification must include a layer name.",
         call. = FALSE)
  }

  constructor <- if (is.character(layer) && length(layer) == 1) {
    layer
  } else {
    "layer"
  }

  list(
    layer = layer,
    constructor = constructor,
    args = args,
    self_attention = self_attention
  )
}

scorch_block_module <- function(scorch_model,
                                block,
                                inputs,
                                ...,
                                .name,
                                .from,
                                inputs_expr,
                                block_expr) {

  if (is.null(block)) {
    stop("Provide `layers` for graph block mode or `block` for module mode.",
         call. = FALSE)
  }

  inputs <- scorch_resolve_inputs(
    scorch_model,
    inputs = if (is.null(inputs_expr)) NULL else
      scorch_parse_refs_expr(inputs_expr, arg = "inputs"),
    from = if (is.null(.from)) NULL else
      scorch_parse_refs_expr(.from, arg = ".from")
  )

  constructor <- scorch_constructor_name(block_expr, fallback = "block")
  node_name <- scorch_prepare_node_name(
    scorch_model,
    explicit_expr = .name,
    legacy_expr = NULL,
    auto_prefix = constructor
  )
  scorch_model <- node_name$model
  name <- node_name$name

  args <- list(...)
  module <- block

  if (is.function(module) && !inherits(module, "nn_module")) {
    module <- do.call(module, args)
  }

  if (!inherits(module, "nn_module")) {
    stop("`block` must be a module instance or constructor.", call. = FALSE)
  }

  scorch_add_graph_node(
    scorch_model,
    name = name,
    module = module,
    inputs = inputs,
    node_type = "block",
    constructor = constructor,
    args = args,
    explicit_name = node_name$explicit
  )
}

#=== END =======================================================================
