#===============================================================================
# INTERNAL UTILITIES FOR SCORCH MODEL GRAPHS
#===============================================================================

utils::globalVariables("self")

`%||%` <- function(x, y) if (is.null(x)) y else x

scorch_empty_graph <- function() {
  tibble::tibble(
    name          = character(),
    module        = list(),
    inputs        = list(),
    node_type     = character(),
    constructor   = character(),
    args          = list(),
    explicit_name = logical(),
    param_count   = numeric(),
    trainable     = logical(),
    output_shape  = list()
  )
}

scorch_ensure_graph_schema <- function(graph) {
  if (is.null(graph)) {
    return(scorch_empty_graph())
  }

  template <- scorch_empty_graph()

  for (nm in names(template)) {
    if (!nm %in% names(graph)) {
      graph[[nm]] <- template[[nm]]
    }
  }

  graph[names(template)]
}

scorch_check_model <- function(scorch_model) {
  if (!inherits(scorch_model, "scorch_model")) {
    stop("`scorch_model` must be a scorch_model object.", call. = FALSE)
  }

  scorch_model$graph <- scorch_ensure_graph_schema(scorch_model$graph)

  if (is.null(scorch_model$auto_name_counts)) {
    scorch_model$auto_name_counts <- list()
  }

  scorch_model
}

scorch_parse_name_expr <- function(expr, arg = "name", allow_null = FALSE) {
  if (is.null(expr)) {
    if (allow_null) return(NULL)
    stop("`", arg, "` is required.", call. = FALSE)
  }

  if (is.symbol(expr)) {
    nm <- as.character(expr)
    if (nm %in% c("", "NULL")) {
      if (allow_null) return(NULL)
      stop("`", arg, "` is required.", call. = FALSE)
    }
    return(nm)
  }

  if (is.character(expr)) {
    if (length(expr) == 0 && allow_null) return(NULL)
    if (length(expr) != 1) {
      stop("`", arg, "` must be a single name.", call. = FALSE)
    }
    return(expr)
  }

  stop("`", arg, "` must be a string or unquoted name.", call. = FALSE)
}

scorch_parse_refs_expr <- function(expr, arg = "inputs", allow_null = TRUE) {
  if (is.null(expr)) {
    if (allow_null) return(NULL)
    stop("`", arg, "` is required.", call. = FALSE)
  }

  if (is.character(expr)) {
    return(expr)
  }

  if (is.symbol(expr)) {
    return(as.character(expr))
  }

  if (is.call(expr) && identical(expr[[1]], as.name("c"))) {
    pieces <- as.list(expr[-1])
    refs <- unlist(lapply(pieces, scorch_parse_refs_expr, arg = arg,
                          allow_null = FALSE), use.names = FALSE)
    return(refs)
  }

  stop("`", arg, "` must be a string, unquoted name, or c(...) of names.",
       call. = FALSE)
}

scorch_resolve_inputs <- function(scorch_model,
                                  inputs = NULL,
                                  from = NULL,
                                  allow_multi = FALSE) {
  scorch_model <- scorch_check_model(scorch_model)

  resolved <- from %||% inputs

  if (is.null(resolved)) {
    if (nrow(scorch_model$graph) == 0) {
      if (length(scorch_model$inputs) == 0) {
        stop("No inputs declared. Add at least one with scorch_input().",
             call. = FALSE)
      }

      if (length(scorch_model$inputs) > 1 && !allow_multi) {
        stop("Must specify `.from` or `inputs` when multiple inputs exist.",
             call. = FALSE)
      }

      resolved <- scorch_model$inputs
    } else {
      resolved <- utils::tail(scorch_model$graph$name, 1)
    }
  }

  all_names <- c(scorch_model$inputs, scorch_model$graph$name)
  bad_inputs <- setdiff(resolved, all_names)

  if (length(bad_inputs) > 0) {
    stop("Input node(s) not found in model: ",
         paste(bad_inputs, collapse = ", "), call. = FALSE)
  }

  resolved
}

scorch_sanitize_prefix <- function(prefix) {
  prefix <- sub("^nn_", "", prefix)
  prefix <- gsub("[^A-Za-z0-9]+", "_", prefix)
  prefix <- gsub("^_+|_+$", "", prefix)
  prefix <- tolower(prefix)
  if (!nzchar(prefix)) "node" else prefix
}

scorch_next_name <- function(scorch_model, prefix) {
  scorch_model <- scorch_check_model(scorch_model)
  prefix <- scorch_sanitize_prefix(prefix)
  existing <- c(scorch_model$inputs, scorch_model$graph$name)

  count <- scorch_model$auto_name_counts[[prefix]] %||% 0L

  repeat {
    count <- count + 1L
    candidate <- paste0(prefix, "_", count)
    if (!candidate %in% existing) break
  }

  scorch_model$auto_name_counts[[prefix]] <- count
  list(model = scorch_model, name = candidate)
}

scorch_validate_new_name <- function(scorch_model, name) {
  if (!is.character(name) || length(name) != 1 || !nzchar(name)) {
    stop("Node names must be non-empty single strings.", call. = FALSE)
  }

  if (name %in% scorch_model$graph$name || name %in% scorch_model$inputs) {
    stop("Node name '", name, "' already exists in the model graph.",
         call. = FALSE)
  }
}

scorch_count_parameters <- function(module) {
  tryCatch({
    params <- module$parameters
    if (is.function(params)) params <- params()
    if (is.null(params) || length(params) == 0) return(0)

    sum(vapply(params, function(p) {
      shape <- tryCatch(as.numeric(p$shape), error = function(e) numeric())
      if (length(shape) == 0) return(0)
      prod(shape)
    }, numeric(1)))
  }, error = function(e) NA_real_)
}

scorch_add_graph_node <- function(scorch_model,
                                  name,
                                  module,
                                  inputs,
                                  node_type,
                                  constructor,
                                  args = list(),
                                  explicit_name = TRUE,
                                  output_shape = NULL) {
  scorch_model <- scorch_check_model(scorch_model)
  scorch_validate_new_name(scorch_model, name)

  param_count <- scorch_count_parameters(module)

  scorch_model$graph <- tibble::add_row(
    scorch_model$graph,
    name          = name,
    module        = list(module),
    inputs        = list(inputs),
    node_type     = node_type,
    constructor   = constructor,
    args          = list(args),
    explicit_name = isTRUE(explicit_name),
    param_count   = param_count,
    trainable     = isTRUE(!is.na(param_count) && param_count > 0),
    output_shape  = list(output_shape)
  )

  scorch_model
}

scorch_prepare_node_name <- function(scorch_model,
                                     explicit_expr = NULL,
                                     legacy_expr = NULL,
                                     auto_prefix = "node") {
  scorch_model <- scorch_check_model(scorch_model)

  if (!is.null(explicit_expr)) {
    return(list(
      model = scorch_model,
      name = scorch_parse_name_expr(explicit_expr, arg = ".name"),
      explicit = TRUE
    ))
  }

  if (!is.null(legacy_expr)) {
    return(list(
      model = scorch_model,
      name = scorch_parse_name_expr(legacy_expr, arg = "name"),
      explicit = TRUE
    ))
  }

  generated <- scorch_next_name(scorch_model, auto_prefix)
  list(model = generated$model, name = generated$name, explicit = FALSE)
}

scorch_constructor_name <- function(expr, fallback = "custom") {
  if (is.null(expr)) return(fallback)
  if (is.symbol(expr)) return(as.character(expr))
  if (is.character(expr) && length(expr) == 1) return(expr)
  if (is.call(expr) && identical(expr[[1]], as.name("::"))) {
    return(as.character(expr[[3]]))
  }
  if (is.call(expr)) return(as.character(expr[[1]]))
  fallback
}

scorch_resolve_layer_fn <- function(layer_fn, layer_expr) {
  if (is.function(layer_fn)) {
    return(layer_fn)
  }

  if (is.symbol(layer_fn) || is.symbol(layer_expr) || is.character(layer_fn)) {
    fn_name <- if (is.character(layer_fn)) layer_fn else as.character(layer_expr)
    if (!grepl("^nn_", fn_name)) fn_name <- paste0("nn_", fn_name)

    if (!exists(fn_name, envir = asNamespace("torch"), mode = "function")) {
      stop("No torch layer called '", fn_name, "'.", call. = FALSE)
    }

    return(get(fn_name, envir = asNamespace("torch")))
  }

  stop("`layer_fn` must be a torch layer name or function.", call. = FALSE)
}

scorch_split_layer_args <- function(args, constructor) {
  if (!identical(scorch_sanitize_prefix(constructor), "multihead_attention")) {
    return(list(constructor = args, forward = list(),
                causal = FALSE, batch_first = FALSE))
  }

  forward_names <- c(
    "key_padding_mask", "need_weights", "attn_mask",
    "average_attn_weights", "is_causal"
  )
  forward_args <- args[intersect(names(args), forward_names)]
  constructor_args <- args[setdiff(names(args), c(forward_names, "causal"))]

  list(
    constructor = constructor_args,
    forward = forward_args,
    causal = isTRUE(args$causal),
    batch_first = isTRUE(constructor_args$batch_first)
  )
}

scorch_causal_attention_mask <- function(query, key, batch_first = FALSE) {
  target_len <- if (isTRUE(batch_first)) query$shape[2] else query$shape[1]
  source_len <- if (isTRUE(batch_first)) key$shape[2] else key$shape[1]

  torch::torch_ones(
    c(target_len, source_len),
    dtype = torch::torch_bool(),
    device = query$device
  )$triu(diagonal = 1)
}

scorch_finalize_layer_module <- function(module,
                                         constructor,
                                         forward_args = list(),
                                         causal = FALSE,
                                         batch_first = FALSE) {
  constructor_key <- scorch_sanitize_prefix(constructor)

  if (identical(constructor_key, "embedding")) {
    raw_module <- module

    return(torch::nn_module(
      initialize = function() {
        self$embedding <- raw_module
      },
      forward = function(x) {
        self$embedding(x$to(dtype = torch::torch_long()))
      }
    )())
  }

  if (!identical(constructor_key, "multihead_attention")) {
    return(module)
  }

  raw_module <- module

  torch::nn_module(
    initialize = function() {
      self$attn <- raw_module
    },
    forward = function(query, key, value) {
      args <- forward_args
      if (isTRUE(causal) && is.null(args$attn_mask)) {
        args$attn_mask <- scorch_causal_attention_mask(
          query, key, batch_first = batch_first
        )
      }
      do.call(self$attn, c(list(query, key, value), args))[[1]]
    }
  )()
}

scorch_as_named_tensor_list <- function(x, default_name = "input") {
  if (inherits(x, "torch_tensor")) {
    out <- list(x)
    names(out) <- default_name
    return(out)
  }

  if (!is.list(x)) {
    out <- list(x)
    names(out) <- default_name
    return(out)
  }

  if (is.null(names(x)) || any(!nzchar(names(x)))) {
    names(x) <- paste0(default_name, seq_along(x))
  }

  x
}

scorch_move_tensor_list <- function(x, device, default_name = "input") {
  x <- scorch_as_named_tensor_list(x, default_name = default_name)
  lapply(x, function(item) {
    if (inherits(item, "torch_tensor")) item$to(device = device) else item
  })
}

scorch_build_module <- function(graph, inputs, outputs) {
  graph <- scorch_ensure_graph_schema(graph)

  torch::nn_module(
    initialize = function() {
      for (i in seq_len(nrow(graph))) {
        self[[graph$name[i]]] <- graph$module[[i]]
      }
    },

    forward = function(...) {
      args <- list(...)
      env <- new.env(parent = emptyenv())

      if (length(inputs) == 1) {
        env[[inputs]] <- args[[1]]
      } else {
        arg_names <- names(args)
        if (!is.null(arg_names) && all(nzchar(arg_names))) {
          for (nm in arg_names) env[[nm]] <- args[[nm]]
        } else {
          for (i in seq_along(inputs)) env[[inputs[i]]] <- args[[i]]
        }
      }

      for (i in seq_len(nrow(graph))) {
        node <- graph[i, ]
        in_vals <- lapply(node$inputs[[1]], function(nm) {
          val <- env[[nm]]
          if (is.null(val)) {
            stop("Input '", nm, "' was not available when evaluating node '",
                 node$name, "'.", call. = FALSE)
          }
          val
        })
        env[[node$name]] <- do.call(self[[node$name]], in_vals)
      }

      if (length(outputs) == 1) {
        env[[outputs]]
      } else {
        stats::setNames(purrr::map(outputs, ~ env[[.x]]), outputs)
      }
    }
  )
}
