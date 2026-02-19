#===============================================================================
# FUNCTION FOR SAVING SCORCHER MODELS
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Save a Compiled Scorcher Model
#'
#' @description
#' Saves a compiled \code{scorch_model} object to disk as a single \code{.pt}
#' file using \code{\link[torch]{torch_save}}. The saved file includes the
#' model weights (state dict), graph architecture, optimizer state (optional),
#' loss function, and metadata about the environment at save time.
#'
#' This is the scorcher-specific save function. For saving bare
#' \code{nn_module} objects, use \code{\link[torch]{torch_save}} directly.
#'
#' @param object A compiled \code{scorch_model} object (i.e., one where
#'   \code{object$compiled} is \code{TRUE}).
#'
#' @param path Character string. File path to save to. If no file extension is
#'   provided, \code{.pt} is appended automatically. Extensions \code{.pt} and
#'   \code{.pth} are accepted silently; other extensions trigger a warning
#'   (the file always uses torch serialization format regardless of extension).
#'
#' @param include_optimizer Logical. If \code{TRUE} (default), the optimizer
#'   state is saved alongside the model weights. Set to \code{FALSE} for
#'   lightweight saves when you only need the trained weights for inference
#'   (the optimizer state can double the file size for optimizers like Adam).
#'
#' @param timestamp Logical. If \code{TRUE}, a timestamp is appended to the
#'   filename in the format \code{_MonDD_HH-MMam/pm} (e.g.,
#'   \code{model_feb18_02-30pm.pt}). Default is \code{FALSE}.
#'
#' @param overwrite Logical. If \code{FALSE} (default) and the target file
#'   already exists, the function stops with an error. Set to \code{TRUE} to
#'   overwrite existing files.
#'
#' @returns Invisibly returns the full path to the saved file.
#'
#' @details
#'
#' \strong{What Gets Saved}
#'
#' The saved \code{.pt} file contains a named list with the following fields:
#' \describe{
#'   \item{\code{state_dict}}{Model weights from
#'     \code{object$nn_model$state_dict()}. This is a named list of tensors,
#'     one per learnable parameter (e.g., layer weights and biases).}
#'   \item{\code{graph}}{The tibble graph architecture, converted to a plain
#'     list for serialization (tibbles don't serialize cleanly through
#'     \code{torch_save}). Contains three vectors: \code{name} (node names),
#'     \code{module} (nn_module objects), and \code{inputs} (which nodes feed
#'     into each node). This allows \code{\link{scorch_load}} to reconstruct
#'     the full \code{nn_module} without needing the original code.}
#'   \item{\code{inputs}}{Character vector of input node names -- which graph
#'     nodes are entry points for data.}
#'   \item{\code{outputs}}{Character vector of output node names -- which graph
#'     nodes produce the final predictions.}
#'   \item{\code{loss_fn}}{The loss function object used during training
#'     (e.g., \code{nn_cross_entropy_loss()}).}
#'   \item{\code{optimizer_state}}{Optimizer state dict (only if
#'     \code{include_optimizer = TRUE}). For Adam, this includes per-parameter
#'     first and second moment estimates (exp_avg, exp_avg_sq), step counts,
#'     and learning rate -- which is why it can roughly double file size.}
#'   \item{\code{optimizer_class}}{Character string naming the optimizer class
#'     (e.g., \code{"optim_adam"}). Used by \code{\link{scorch_load}} to
#'     reconstruct the correct optimizer type automatically.}
#'   \item{\code{metadata}}{A list containing environment information:
#'     \describe{
#'       \item{\code{timestamp}}{When the model was saved (human-readable).}
#'       \item{\code{r_version}}{R version string for reproducibility.}
#'       \item{\code{torch_version}}{Version of the torch package. Tensor
#'         serialization format can change between versions.}
#'       \item{\code{scorcher_version}}{Version of the scorcher package, or
#'         \code{"dev"} if not yet installed.}
#'       \item{\code{device}}{Device the model was on when saved
#'         (\code{"cpu"} or \code{"cuda"}). Useful when transferring models
#'         between a GPU cluster and a local laptop.}
#'       \item{\code{os}}{Operating system (\code{"Linux"}, \code{"Darwin"},
#'         \code{"Windows"}).}
#'       \item{\code{include_optimizer}}{Logical flag recording whether the
#'         optimizer state was included in this save.}
#'     }
#'   }
#' }
#'
#' \strong{Overwrite Protection}
#'
#' By default, \code{scorch_save} refuses to overwrite an existing file. This
#' prevents accidental loss of a trained model -- e.g., if a save script is
#' accidentally re-run after hours of training. Use \code{overwrite = TRUE} to
#' explicitly allow overwriting.
#'
#' \strong{Resuming Training}
#'
#' To resume training from a saved checkpoint, load with
#' \code{\link{scorch_load}} and continue with \code{\link{fit_scorch}}. The
#' optimizer state (including learning rate schedules, momentum buffers, etc.)
#' is restored automatically if it was saved.
#'
#' \strong{File Format}
#'
#' The file uses torch's native serialization format (not R's
#' \code{saveRDS}). This is necessary because R's serialization cannot handle
#' torch tensor objects. The \code{.pt} extension is the PyTorch convention;
#' \code{.pth} is also common.
#'
#' @examples
#' \dontrun{
#' # Save a trained model
#' scorch_save(my_model, "models/my_model.pt")
#'
#' # Save without optimizer (smaller file, inference only)
#' scorch_save(my_model, "models/my_model.pt",
#'             include_optimizer = FALSE)
#'
#' # Save with timestamp in filename
#' scorch_save(my_model, "models/my_model",
#'             timestamp = TRUE)
#' # Creates: models/my_model_feb18_02-30pm.pt
#'
#' # Overwrite an existing file
#' scorch_save(my_model, "models/my_model.pt",
#'             overwrite = TRUE)
#' }
#'
#' @family model I/O
#'
#' @export

scorch_save <- function(object,
                        path,
                        include_optimizer = TRUE,
                        timestamp = FALSE,
                        overwrite = FALSE) {

  # ===== Input validation ===================================================

  #- scorch_save only accepts scorch_model objects -- not bare nn_modules.
  #- This ensures we always have the full scorcher pipeline (graph, optimizer,
  #- loss, metadata) available to save. Users with bare nn_modules should use
  #- torch::torch_save() directly, which handles raw tensors and nn_modules.

  if (!inherits(object, "scorch_model")) {

    stop("`object` must be a scorch_model. ",
         "For bare nn_modules, use torch::torch_save() directly.",
         call. = FALSE)
  }

  #- The model must be compiled (i.e., compile_scorch() was called) because
  #- compilation is when the nn_module is created from the graph. An uncompiled
  #- model has no nn_model or state_dict to save -- it's just a graph blueprint.

  if (!isTRUE(object$compiled)) {

    stop("Model must be compiled before saving. ",
         "Run compile_scorch() first.",
         call. = FALSE)
  }

  if (!is.character(path) || length(path) != 1) {

    stop("`path` must be a single character string.", call. = FALSE)
  }

  if (!is.logical(include_optimizer) || length(include_optimizer) != 1) {

    stop("`include_optimizer` must be TRUE or FALSE.", call. = FALSE)
  }

  if (!is.logical(timestamp) || length(timestamp) != 1) {

    stop("`timestamp` must be TRUE or FALSE.", call. = FALSE)
  }

  if (!is.logical(overwrite) || length(overwrite) != 1) {

    stop("`overwrite` must be TRUE or FALSE.", call. = FALSE)
  }

  # ===== Path handling ======================================================

  #- We handle three cases for the file path:
  #-   1. No extension provided    -> auto-append ".pt"
  #-   2. Standard extension (.pt, .pth) -> accept silently
  #-   3. Non-standard extension (.rds, .rda, etc.) -> accept but warn
  #-
  #- The warning for non-standard extensions exists because the file always
  #- uses torch's serialization format internally, regardless of what the
  #- extension suggests. Saving as ".rds" would be misleading -- readRDS()
  #- would fail on it.
  #-
  #- If timestamp = TRUE, we insert a timestamp string before the extension.
  #- Format: _mon##_HH-MMam/pm (e.g., _feb18_02-30pm)
  #- This matches the old save_model_with_timestamp() convention from the
  #- original save_model_fn.R, but the timestamp is now in metadata too.

  ext <- tools::file_ext(path)

  if (ext == "") {

    #- No extension provided (e.g., "models/my_model").
    #- If timestamp is TRUE, append timestamp then .pt.
    #- If FALSE, just append .pt.

    if (timestamp) {

      time_str <- tolower(format(Sys.time(), "%b%d_%I-%M%p"))
      path <- paste0(path, "_", time_str, ".pt")

    } else {

      path <- paste0(path, ".pt")
    }

  } else {

    #- Extension was provided (e.g., "model.pt" or "model.rds").
    #- If timestamp is TRUE, insert timestamp between base name and extension.
    #- E.g., "model.pt" becomes "model_feb18_02-30pm.pt"

    if (timestamp) {

      time_str <- tolower(format(Sys.time(), "%b%d_%I-%M%p"))
      base <- tools::file_path_sans_ext(path)
      path <- paste0(base, "_", time_str, ".", ext)
    }

    #- Warn on non-standard extensions.
    #- .pt (PyTorch convention) and .pth (also widely used) are standard.
    #- Anything else could mislead users about the file format.

    if (!ext %in% c("pt", "pth")) {

      warning("Extension '.", ext, "' may be misleading. ",
              "File uses torch serialization format, not ", ext, ". ",
              "Standard extensions are .pt or .pth.",
              call. = FALSE)
    }
  }

  # ===== Overwrite protection ===============================================

  #- By default, refuse to overwrite an existing file.
  #- This prevents accidental loss of a trained model -- e.g., if a training
  #- script is accidentally re-run and overwrites a good checkpoint.
  #- The user must explicitly pass overwrite = TRUE to replace a file.

  if (file.exists(path) && !overwrite) {

    stop("File already exists: '", path, "'. ",
         "Use overwrite = TRUE to replace it.",
         call. = FALSE)
  }

  # ===== Create parent directory if needed ==================================

  #- Automatically create any missing parent directories in the save path.
  #- E.g., scorch_save(model, "results/exp3/checkpoints/model.pt") creates
  #- the full results/exp3/checkpoints/ directory tree if it doesn't exist.

  dir_path <- dirname(path)

  if (!dir.exists(dir_path)) {

    dir.create(dir_path, recursive = TRUE)

    message("Created directory: ", dir_path)
  }

  # ===== Detect device ======================================================

  #- Determine which device (cpu or cuda) the model is currently on.
  #- We check the device of the first parameter tensor. In torch, all
  #- parameters of a model should be on the same device (enforced by $to()),
  #- so checking the first one is sufficient.
  #-
  #- This is saved in metadata so that scorch_load() can report what device
  #- the model was trained/saved on -- useful when transferring models between
  #- a GPU cluster and a local laptop.

  model_device <- tryCatch({

    params <- object$nn_model$parameters

    if (length(params) > 0) {

      as.character(params[[1]]$device)

    } else {

      #- Model has no learnable parameters (unusual but possible for
      #- pure pass-through or embedding-only models).

      "cpu"
    }

  }, error = function(e) "unknown")

  # ===== Assemble save payload ==============================================

  #- We save everything as a single named list, then serialize it with
  #- torch_save(). This list IS the .pt file format for scorcher models.
  #- scorch_load() reads this exact structure back.

  #- Convert graph tibble to a plain list for serialization.
  #- Tibbles (from the tibble package) are internally complex R objects with
  #- attributes, row names, and column metadata that don't survive torch's
  #- serialization cleanly. A plain list with three vectors (name, module,
  #- inputs) preserves all the architectural information and serializes
  #- reliably. scorch_load() converts it back to a tibble on load.

  graph_as_list <- if (!is.null(object$graph)) {

    list(
      name    = object$graph$name,     # Character vector of node names
      module  = object$graph$module,   # List of nn_module objects
      inputs  = object$graph$inputs    # List of character vectors (DAG edges)
    )

  } else {

    NULL
  }

  #- Build metadata: a snapshot of the computing environment at save time.
  #- This serves two purposes:
  #-   1. Reproducibility -- know exactly what R/torch/scorcher versions
  #-      produced a model, which OS it ran on, and what device it used.
  #-   2. Diagnostics -- scorch_load() compares saved vs current versions
  #-      and warns on mismatches that could cause compatibility issues.
  #-
  #- scorcher_version uses tryCatch because during development (before the
  #- package is installed), packageVersion("scorcher") would error. We
  #- record "dev" in that case.

  metadata <- list(
    timestamp         = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
    r_version         = R.version.string,
    torch_version     = as.character(utils::packageVersion("torch")),
    scorcher_version  = tryCatch(
      as.character(utils::packageVersion("scorcher")),
      error = function(e) "dev"
    ),
    device            = model_device,
    os                = as.character(Sys.info()["sysname"]),
    include_optimizer = include_optimizer
  )

  #- Assemble the full payload -- the named list that gets serialized.
  #-
  #- state_dict: the trained weights -- the most important part. This is a
  #-   named list of tensors, one per learnable parameter.
  #- graph: the architecture blueprint -- allows scorch_load() to rebuild
  #-   the nn_module without needing the original model-building code.
  #- inputs/outputs: which graph nodes are entry/exit points for data.
  #- loss_fn: the loss function object -- needed to resume training with
  #-   the same objective.
  #- metadata: environment snapshot for reproducibility and diagnostics.

  payload <- list(
    state_dict = object$nn_model$state_dict(),
    graph      = graph_as_list,
    inputs     = object$inputs,
    outputs    = object$outputs,
    loss_fn    = object$loss_fn,
    metadata   = metadata
  )

  #- Optionally include optimizer state and class name.
  #-
  #- The optimizer state dict contains per-parameter internal state. For Adam,
  #- this includes first moment estimates (exp_avg), second moment estimates
  #- (exp_avg_sq), and step counts for every parameter. This is what allows
  #- training to resume smoothly -- without it, the optimizer "forgets" its
  #- momentum and adaptive learning rate history.
  #-
  #- The class name (e.g., "optim_adam") tells scorch_load() which optimizer
  #- constructor to use when reconstructing. Without it, we'd have the internal
  #- state but no way to create the right optimizer object to pour it into.
  #- This is like having fuel but not knowing if the engine is diesel or gas.
  #-
  #- Setting include_optimizer = FALSE skips both, producing a smaller file
  #- suitable for inference-only deployment.

  if (include_optimizer && !is.null(object$optimizer)) {

    payload$optimizer_state <- object$optimizer$state_dict()
    payload$optimizer_class <- class(object$optimizer)[1]
  }

  # ===== Save ===============================================================

  #- torch_save() serializes the payload list to disk using torch's native
  #- serialization format. This handles tensors, nn_modules, and arbitrarily
  #- nested lists correctly -- unlike R's saveRDS() which cannot serialize
  #- torch tensor objects (they live in C++/LibTorch memory, not R memory).

  torch::torch_save(payload, path)

  #- Report file size so the user knows how much disk space was used.
  #- Especially useful on cluster environments with storage quotas, or when
  #- comparing file sizes with/without optimizer state.

  file_size <- file.info(path)$size
  size_str <- format_file_size(file_size)

  message("Model saved to: ", path, " (", size_str, ")")

  invisible(path)
}

#=== UTILITY FUNCTIONS =========================================================

#' Format File Size for Display
#'
#' @description
#' Converts a file size in bytes to a human-readable string with appropriate
#' units (B, KB, MB, GB). Used internally by \code{\link{scorch_save}} to
#' report file size after saving.
#'
#' @param bytes Numeric. File size in bytes.
#'
#' @returns A character string with the size and unit (e.g., \code{"42.3 MB"},
#'   \code{"1.2 GB"}, \code{"768 B"}).
#'
#' @keywords internal

format_file_size <- function(bytes) {

  if (bytes < 1024) {

    sprintf("%d B", bytes)

  } else if (bytes < 1024^2) {

    sprintf("%.1f KB", bytes / 1024)

  } else if (bytes < 1024^3) {

    sprintf("%.1f MB", bytes / 1024^2)

  } else {

    sprintf("%.1f GB", bytes / 1024^3)
  }
}

#=== END =======================================================================
