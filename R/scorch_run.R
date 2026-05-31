#===============================================================================
# REPRODUCIBLE SCORCH RUNS
#===============================================================================

scorch_package_version <- function(pkg) {
  tryCatch(as.character(utils::packageVersion(pkg)), error = function(e) "dev")
}

scorch_model_hash <- function(scorch_model) {
  spec <- scorch_spec(scorch_model)
  spec$args <- lapply(spec$args, function(x) {
    tryCatch(as.character(x), error = function(e) "<unserializable>")
  })
  digest::digest(spec)
}

scorch_data_fingerprint <- function(data) {
  if (is.null(data)) return(NULL)

  base <- list(
    class = class(data),
    digest = tryCatch(digest::digest(data), error = function(e) NA_character_)
  )

  if (inherits(data, "torch_tensor")) {
    base$shape <- tryCatch(as.numeric(data$shape), error = function(e) NULL)
    base$dtype <- tryCatch(as.character(data$dtype), error = function(e) NULL)
    return(base)
  }

  if (is.data.frame(data) || is.matrix(data) || is.array(data)) {
    base$dim <- dim(data)
    base$names <- names(data)
    base$column_classes <- if (is.data.frame(data)) {
      vapply(data, function(x) paste(class(x), collapse = "/"), character(1))
    } else {
      NULL
    }
  }

  base
}

#' Snapshot a Reproducible Scorch Run
#'
#' @param scorch_model A \code{scorch_model} object.
#' @param data Optional data object to fingerprint.
#' @param config Optional named list of run configuration.
#'
#' @returns A \code{scorch_run} object.
#'
#' @family reproducibility
#'
#' @export
scorch_snapshot <- function(scorch_model, data = NULL, config = list()) {
  scorch_model <- scorch_check_model(scorch_model)

  run <- list(
    graph = scorch_spec(scorch_model),
    graph_hash = scorch_model_hash(scorch_model),
    compiled = isTRUE(scorch_model$compiled),
    outputs = scorch_model$outputs,
    inputs = scorch_model$inputs,
    history = scorch_model$history,
    training = scorch_model$metadata$training %||% list(),
    config = config,
    data = scorch_data_fingerprint(data),
    versions = list(
      r = R.version.string,
      torch = scorch_package_version("torch"),
      scorcher = scorch_package_version("scorcher")
    ),
    platform = list(
      os = Sys.info()[["sysname"]],
      release = Sys.info()[["release"]],
      machine = Sys.info()[["machine"]]
    ),
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )

  class(run) <- c("scorch_run", "list")
  run
}

#' Audit a Scorch Run
#'
#' @param run A \code{scorch_run} object.
#'
#' @returns A tibble with audit checks.
#'
#' @family reproducibility
#'
#' @export
scorch_audit <- function(run) {
  if (!inherits(run, "scorch_run")) {
    stop("`run` must be a scorch_run object.", call. = FALSE)
  }

  checks <- list(
    list("graph_hash", !is.null(run$graph_hash), "Graph hash is present."),
    list("compiled", isTRUE(run$compiled), "Model was compiled at snapshot."),
    list("history", !is.null(run$history), "Training history is present."),
    list("seed", !is.null(run$training$seed), "Training seed is recorded."),
    list("data", !is.null(run$data), "Data fingerprint is present."),
    list("versions", !is.null(run$versions$torch), "Package versions are present.")
  )

  tibble::tibble(
    check = vapply(checks, `[[`, character(1), 1),
    status = vapply(checks, function(x) if (isTRUE(x[[2]])) "pass" else "warn",
                    character(1)),
    message = vapply(checks, `[[`, character(1), 3)
  )
}

#' Save a Scorch Run
#'
#' @param run A \code{scorch_run} object.
#' @param path File path. If no extension is supplied, \code{.pt} is added.
#' @param overwrite Logical. Overwrite an existing file?
#'
#' @returns Invisibly returns the saved path.
#'
#' @family reproducibility
#'
#' @export
scorch_save_run <- function(run, path, overwrite = FALSE) {
  if (!inherits(run, "scorch_run")) {
    stop("`run` must be a scorch_run object.", call. = FALSE)
  }

  if (!is.character(path) || length(path) != 1) {
    stop("`path` must be a single character string.", call. = FALSE)
  }

  if (!nzchar(tools::file_ext(path))) {
    path <- paste0(path, ".pt")
  }

  if (file.exists(path) && !overwrite) {
    stop("File already exists: '", basename(path), "'. ",
         "Use overwrite = TRUE to replace it.", call. = FALSE)
  }

  dir_path <- dirname(path)
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
  }

  torch::torch_save(run, path)
  invisible(path)
}

#' Load a Scorch Run
#'
#' @param path File path created by \code{scorch_save_run()}.
#'
#' @returns A \code{scorch_run} object.
#'
#' @family reproducibility
#'
#' @export
scorch_load_run <- function(path) {
  if (!file.exists(path)) {
    stop("File not found: '", path, "'.", call. = FALSE)
  }

  run <- torch::torch_load(path)
  class(run) <- unique(c("scorch_run", class(run)))
  run
}

#' Compare Scorch Runs
#'
#' @param ... \code{scorch_run} objects.
#'
#' @returns A tibble with one row per run.
#'
#' @family reproducibility
#'
#' @export
scorch_compare_runs <- function(...) {
  runs <- list(...)
  if (length(runs) == 1 && is.list(runs[[1]]) &&
      !inherits(runs[[1]], "scorch_run")) {
    runs <- runs[[1]]
  }

  bad <- !vapply(runs, inherits, logical(1), "scorch_run")
  if (any(bad)) {
    stop("All inputs must be scorch_run objects.", call. = FALSE)
  }

  tibble::tibble(
    run = seq_along(runs),
    graph_hash = vapply(runs, function(x) x$graph_hash %||% NA_character_,
                        character(1)),
    backend = vapply(runs, function(x) x$training$backend %||% NA_character_,
                     character(1)),
    device = vapply(runs, function(x) x$training$device %||% NA_character_,
                    character(1)),
    epochs = vapply(runs, function(x) x$training$epochs %||% NA_real_,
                    numeric(1)),
    final_loss = vapply(runs, function(x) {
      if (is.null(x$history) || !("loss" %in% names(x$history))) return(NA_real_)
      utils::tail(x$history$loss, 1)
    }, numeric(1))
  )
}
