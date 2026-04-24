#===============================================================================
# FUNCTION FOR LOADING SCORCHER MODELS
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Load a Saved Scorcher Model
#'
#' @description
#' Loads a \code{scorch_model} object previously saved with
#' \code{\link{scorch_save}}. The function reconstructs the full model from
#' the saved graph architecture, loads the trained weights (state dict),
#' restores the optimizer (if available), and moves the model to the specified
#' device.
#'
#' @param path Character string. Path to the \code{.pt} file created by
#'   \code{\link{scorch_save}}.
#'
#' @param device Character string. Device to load the model onto. One of
#'   \code{"cpu"} (default) or \code{"cuda"}. Use \code{"cpu"} when loading a
#'   model trained on GPU onto a machine without a GPU (common workflow: train
#'   on cluster, analyze on laptop).
#'
#' @param verbose Logical. If \code{TRUE} (default), prints a summary of the
#'   loaded model including when it was saved, package versions, and device
#'   info via \code{message()} and \code{cat()}.
#'
#' @returns A \code{scorch_model} object (a list of class
#'   \code{"scorch_model"}) with \code{compiled = TRUE}, containing:
#'   \describe{
#'     \item{\code{nn_model}}{The reconstructed and weight-loaded
#'       \code{nn_module}, moved to the specified device.}
#'     \item{\code{graph}}{The model architecture as a tibble (restored from
#'       the saved list representation). Contains columns: \code{name}
#'       (character), \code{module} (list of nn_modules), \code{inputs}
#'       (list of character vectors defining the DAG edges).}
#'     \item{\code{inputs}}{Character vector of input node names.}
#'     \item{\code{outputs}}{Character vector of output node names.}
#'     \item{\code{loss_fn}}{The loss function used during training.}
#'     \item{\code{optimizer}}{The fully reconstructed optimizer with restored
#'       state (if optimizer state and class were saved), or \code{NULL}
#'       otherwise.}
#'     \item{\code{compiled}}{Always \code{TRUE}.}
#'     \item{\code{metadata}}{The metadata from save time (timestamp, versions,
#'       device, OS).}
#'   }
#'
#' @details
#'
#' \strong{Model Reconstruction}
#'
#' The function reconstructs the \code{nn_module} from the saved graph
#' architecture using the same logic as \code{\link{compile_scorch}}:
#' \enumerate{
#'   \item Creates an \code{nn_module} whose \code{initialize()} method
#'     registers every graph node as a named sub-module.
#'   \item The \code{forward()} method traverses the graph in topological
#'     order, using an environment to pass intermediate results between nodes
#'     according to the DAG structure defined in the \code{inputs} column.
#' }
#'
#' This means you do not need the original model-building code to load a saved
#' model -- the graph contains everything needed to reconstruct the full
#' architecture and forward pass.
#'
#' \strong{Device Handling}
#'
#' When loading a model saved on GPU (\code{"cuda"}) onto a CPU-only machine,
#' use \code{device = "cpu"} (the default). The function handles the device
#' transfer automatically via \code{nn_model$to(device)}. When loading onto a
#' machine with a GPU, use \code{device = "cuda"} to place the model on the
#' GPU immediately. If \code{"cuda"} is requested but unavailable, the function
#' warns and falls back to CPU.
#'
#' \strong{Version Checking}
#'
#' The function compares the saved torch and scorcher versions against the
#' currently installed versions. If there is a mismatch, a warning is issued.
#' This does not prevent loading, but version differences can occasionally
#' cause compatibility issues with serialized tensors (torch's serialization
#' format can change between major versions).
#'
#' \strong{Optimizer Reconstruction}
#'
#' If the model was saved with \code{include_optimizer = TRUE}, the optimizer
#' is automatically reconstructed in three steps:
#' \enumerate{
#'   \item The saved class name (e.g., \code{"optim_adam"}) is looked up in
#'     a table of known torch optimizers.
#'   \item A fresh optimizer is created with the loaded model's parameters.
#'   \item The saved state dict (momentum buffers, learning rate history, etc.)
#'     is loaded into the fresh optimizer.
#' }
#'
#' The following torch optimizers are supported for auto-reconstruction:
#' \code{optim_adam}, \code{optim_adamw}, \code{optim_sgd},
#' \code{optim_rmsprop}, \code{optim_adagrad}, \code{optim_adadelta},
#' \code{optim_rprop}, \code{optim_asgd}, and \code{optim_lbfgs}.
#'
#' If a custom optimizer was used, the raw state dict is stored in
#' \code{model$optimizer} and a warning is issued. The user can then manually
#' create their optimizer and call
#' \code{optimizer$load_state_dict(model$optimizer)} to restore the state.
#'
#' @examples
#' \dontrun{
#' # Load a model onto CPU
#' model <- scorch_load("models/my_model.pt")
#'
#' # Load onto GPU
#' model <- scorch_load("models/my_model.pt", device = "cuda")
#'
#' # Load silently
#' model <- scorch_load("models/my_model.pt", verbose = FALSE)
#'
#' # Resume training from a checkpoint
#' model <- scorch_load("models/checkpoint.pt")
#' model <- fit_scorch(model, num_epochs = 10)
#'
#' # Check what environment the model was saved in
#' model$metadata$device       # "cuda" -- trained on GPU
#' model$metadata$os           # "Linux" -- trained on cluster
#' model$metadata$torch_version # "0.13.0"
#' }
#'
#' @family model I/O
#'
#' @export

scorch_load <- function(path,
                        device = "cpu",
                        verbose = TRUE) {

  #- Null-coalescing helper (avoids R >= 4.4 / rlang dependency for %||%).

  `%||%` <- function(x, y) if (is.null(x)) y else x

  # ===== Input validation ===================================================

  if (!is.character(path) || length(path) != 1) {

    stop("`path` must be a single character string.", call. = FALSE)
  }

  if (!file.exists(path)) {

    stop("File not found: '", path, "'.", call. = FALSE)
  }

  if (!is.character(device) || length(device) != 1) {

    stop("`device` must be a single character string ('cpu' or 'cuda').",
         call. = FALSE)
  }

  if (!device %in% c("cpu", "cuda")) {

    stop("`device` must be 'cpu' or 'cuda'. Got: '", device, "'.",
         call. = FALSE)
  }

  #- If the user requests CUDA but it's not available (e.g., running on a
  #- laptop without a GPU), fall back to CPU with a warning rather than
  #- crashing. This is a common scenario when downloading a model trained
  #- on a cluster to analyze locally.

  if (device == "cuda" && !torch::cuda_is_available()) {

    warning("CUDA requested but not available. Loading to CPU instead.",
            call. = FALSE)

    device <- "cpu"
  }

  # ===== Load the saved payload =============================================

  #- torch_load() reads the file saved by torch_save().
  #- The result is the exact named list that scorch_save() assembled:
  #- state_dict, graph, inputs, outputs, loss_fn, metadata, and optionally
  #- optimizer_state and optimizer_class.
  #-
  #- Wrapped in tryCatch because the file could be corrupted, truncated
  #- (common on cluster filesystems), or not a torch file at all.

  payload <- tryCatch({

    torch::torch_load(path)

  }, error = function(e) {

    stop("Error loading file '", path, "': ", e$message,
         call. = FALSE)
  })

  #- Validate that this file looks like it was created by scorch_save().
  #- We check for the required fields that every scorch_save() file has.
  #- This catches the case where someone passes a bare torch_save() file
  #- or a completely unrelated .pt file.

  required_fields <- c("state_dict", "graph", "inputs", "outputs", "metadata")

  missing_fields <- setdiff(required_fields, names(payload))

  if (length(missing_fields) > 0) {

    stop("File does not appear to be a scorcher model. Missing fields: ",
         paste(missing_fields, collapse = ", "), ". ",
         "Was this file created with scorch_save()?",
         call. = FALSE)
  }

  # ===== Version checking ===================================================

  #- Compare saved package versions against currently installed versions.
  #- Warn on mismatches but don't block loading -- most version differences
  #- are harmless, but occasionally torch's tensor serialization format
  #- changes between major versions, causing silent corruption.
  #-
  #- We skip the comparison if either version is "dev" (development mode,
  #- package not yet installed).

  saved_meta <- payload$metadata

  current_torch <- as.character(utils::packageVersion("torch"))

  current_scorcher <- tryCatch(
    as.character(utils::packageVersion("scorcher")),
    error = function(e) "dev"
  )

  if (!is.null(saved_meta$torch_version) &&
      saved_meta$torch_version != current_torch) {

    warning("Torch version mismatch: model saved with ",
            saved_meta$torch_version, ", current is ", current_torch,
            ". This may cause compatibility issues.",
            call. = FALSE)
  }

  if (!is.null(saved_meta$scorcher_version) &&
      saved_meta$scorcher_version != current_scorcher &&
      saved_meta$scorcher_version != "dev" &&
      current_scorcher != "dev") {

    warning("Scorcher version mismatch: model saved with ",
            saved_meta$scorcher_version, ", current is ", current_scorcher, ".",
            call. = FALSE)
  }

  # ===== Reconstruct graph ==================================================

  #- The graph was saved as a plain list (three vectors: name, module, inputs)
  #- because tibbles don't serialize cleanly through torch_save().
  #- Convert back to a tibble -- the standard format used by all scorcher
  #- functions (scorch_layer, compile_scorch, fit_scorch, etc.).
  #-
  #- The tibble has three columns:
  #-   name:   Character -- unique node name (e.g., "fc1", "relu1", "output")
  #-   module: List -- nn_module objects (the actual layers with weights)
  #-   inputs: List of character vectors -- which nodes feed into this one
  #-           (defines the DAG edges)

  graph <- if (!is.null(payload$graph)) {

    tibble::tibble(
      name   = payload$graph$name,
      module = payload$graph$module,
      inputs = payload$graph$inputs
    )

  } else {

    NULL
  }

  # ===== Reconstruct nn_module from graph ===================================

  #- This mirrors the logic in compile_scorch():
  #-
  #- The nn_module is built dynamically from the graph tibble. The key insight
  #- is that the graph defines a DAG (directed acyclic graph) where each node
  #- knows its inputs. The forward pass traverses nodes in order, using an
  #- environment as a "scratchpad" to store intermediate results.
  #-
  #- initialize(): Registers every graph node as a named sub-module on `self`.
  #-   This is critical because torch tracks parameters through registered
  #-   sub-modules -- without this, $parameters would be empty and the model
  #-   wouldn't have any learnable weights.
  #-
  #- forward(): Processes data through the graph:
  #-   1. Assign input data to the environment using the input node names.
  #-   2. For each graph node (in topological order):
  #-      a. Look up its input values from the environment.
  #-      b. Call the node's module with those inputs.
  #-      c. Store the result in the environment under the node's name.
  #-   3. Return the values stored under the output node names.
  #-
  #- The environment-based approach allows arbitrary DAG topologies: skip
  #- connections, multi-input fusion, branching, etc. -- any node can reference
  #- any earlier node as input.

  inputs  <- payload$inputs
  outputs <- payload$outputs

  mod <- torch::nn_module(

    initialize = function() {

      #- Register every graph node as a named sub-module.
      #- This makes torch aware of all parameters for optimization,
      #- device transfer ($to()), and state_dict serialization.

      for (i in seq_len(nrow(graph))) {

        self[[graph$name[i]]] <- graph$module[[i]]
      }
    },

    forward = function(...) {

      args <- list(...)

      #- Create a clean environment to store intermediate results.
      #- Using an environment (not a list) for O(1) name lookups and
      #- to avoid repeated list copying.

      env <- new.env(parent = emptyenv())

      #- Step 1: Assign input data to the environment.
      #- For single-input models: the first positional argument is assigned
      #- to the input node name.
      #- For multi-input models: arguments are matched by name.

      if (length(inputs) == 1) {

        env[[inputs]] <- args[[1]]

      } else {

        for (nm in names(args)) env[[nm]] <- args[[nm]]
      }

      #- Step 2: Traverse the graph in topological order.
      #- For each node:
      #-   a. Gather its input values from the environment (these were either
      #-      the original inputs or outputs from earlier nodes).
      #-   b. Call the node's module (nn_linear, nn_relu, etc.) with those
      #-      input values using do.call().
      #-   c. Store the result in the environment under this node's name,
      #-      making it available as input to later nodes.

      for (i in seq_len(nrow(graph))) {

        node    <- graph[i, ]
        in_vals <- lapply(node$inputs[[1]], function(nm) env[[nm]])
        out     <- do.call(self[[node$name]], in_vals)
        env[[node$name]] <- out
      }

      #- Step 3: Return the output(s).
      #- For single-output models: return the tensor directly.
      #- For multi-output models: return a named list of tensors.

      if (length(outputs) == 1) {

        env[[outputs]]

      } else {

        purrr::map(outputs, ~ env[[.x]])
      }
    }
  )

  #- Instantiate the module. This calls initialize(), which registers all
  #- the graph nodes as sub-modules -- creating the parameter tracking.

  nn_model <- mod()

  #- Load the saved weights into the reconstructed module.
  #- load_state_dict() matches saved parameter names to module parameter names
  #- and copies the tensor values. This is why the graph must produce the
  #- same parameter names as the original model -- which it does, because we
  #- saved and restored the exact same graph structure.

  nn_model$load_state_dict(payload$state_dict)

  #- Move the model to the requested device.
  #- $to() recursively moves all parameters and buffers to the target device.
  #- This handles the common case of loading a GPU-trained model onto CPU.

  torch_device <- torch::torch_device(device)

  nn_model <- nn_model$to(device = torch_device)

  # ===== Restore optimizer (if saved) =======================================

  #- If the optimizer class name was saved, we can fully reconstruct it:
  #-   1. Look up the constructor function from the class name string
  #-      (e.g., "optim_adam" -> torch::optim_adam).
  #-   2. Create a fresh optimizer instance with the loaded model's parameters.
  #-   3. Load the saved state dict into the fresh optimizer, restoring all
  #-      internal state: learning rates, momentum buffers, step counts,
  #-      adaptive learning rate history, etc.
  #-
  #- This three-step process is necessary because optimizers in torch are
  #- stateful objects tied to specific parameter tensors. We can't just
  #- deserialize an optimizer directly -- we need to create one bound to
  #- the NEW parameter tensors (from the reconstructed model), then pour
  #- the saved internal state into it.
  #-
  #- If the class name is unknown (e.g., a custom optimizer not in our
  #- lookup table), we fall back to storing the raw state dict. The user
  #- can create their optimizer manually and call $load_state_dict().

  optimizer <- NULL

  if (!is.null(payload$optimizer_state)) {

    if (!is.null(payload$optimizer_class)) {

      #- Look up the optimizer constructor from the class name.

      optim_fn <- optimizer_lookup(payload$optimizer_class)

      if (!is.null(optim_fn)) {

        #- Known optimizer: create fresh instance and load saved state.

        optimizer <- optim_fn(nn_model$parameters)
        optimizer$load_state_dict(payload$optimizer_state)

      } else {

        #- Unknown optimizer class (custom/third-party).
        #- Store the raw state dict for manual restoration.

        warning("Unknown optimizer class '", payload$optimizer_class, "'. ",
                "State dict stored but cannot auto-reconstruct. ",
                "Create your optimizer manually and call ",
                "optimizer$load_state_dict(model$optimizer_state).",
                call. = FALSE)

        optimizer <- payload$optimizer_state
      }

    } else {

      #- Optimizer state exists but no class name was saved.
      #- This can happen with models saved by an older version of scorcher
      #- that didn't save the class name.

      warning("Optimizer state found but no class name saved. ",
              "State dict stored but cannot auto-reconstruct. ",
              "This model may have been saved with an older version of ",
              "scorcher.",
              call. = FALSE)

      optimizer <- payload$optimizer_state
    }
  }

  # ===== Assemble scorch_model ==============================================

  #- Build the scorch_model list -- the same structure that compile_scorch()
  #- produces. This means the loaded model can be used immediately with
  #- fit_scorch(), print.scorch_model(), and any other scorcher function
  #- that expects a compiled scorch_model.

  sm <- list(
    nn_model  = nn_model,
    graph     = graph,
    inputs    = inputs,
    outputs   = outputs,
    loss_fn   = payload$loss_fn,
    optimizer = optimizer,
    compiled  = TRUE,
    metadata  = saved_meta
  )

  class(sm) <- c("scorch_model", class(sm))

  # ===== Verbose report =====================================================

  #- Print a summary of the loaded model so the user can verify:
  #-   - When and where the model was saved
  #-   - Whether package versions match
  #-   - What device it's being loaded to
  #-   - Whether the optimizer was restored
  #-
  #- Uses cat() for formatted output (not message()) because this is a
  #- structured report, not a status message. The initial "Loading..." line
  #- uses message() so it can be suppressed separately.

  if (verbose) {

    #- Helper for optional color formatting.
    #- Falls back to identity() if crayon is not installed.

    highlight <- if (requireNamespace("crayon", quietly = TRUE)) {

      crayon::red

    } else {

      identity
    }

    message("Loading scorcher model from: ", path)

    message(paste0(" * Saved on: ",
               highlight(saved_meta$timestamp %||% "unknown")))

    message(paste0(" * Scorcher version: ",
               highlight(saved_meta$scorcher_version %||% "unknown"),
               " (current: ", current_scorcher, ")"))

    message(paste0(" * Torch version: ",
               highlight(saved_meta$torch_version %||% "unknown"),
               " (current: ", current_torch, ")"))

    message(paste0(" * R version: ",
               highlight(saved_meta$r_version %||% "unknown")))

    message(paste0(" * Environment: ",
               highlight(paste0(
                 saved_meta$os %||% "unknown",
                 " (", saved_meta$device %||% "unknown", ")"
               ))))

    message(paste0(" * Loading to: ",
               highlight(device)))

    message(paste0(" * Optimizer state: ",
               highlight(
                 if (!is.null(payload$optimizer_state)) {
                   paste0("included (",
                          payload$optimizer_class %||% "unknown class", ")")
                 } else {
                   "not saved"
                 }
               )))
  }

  sm
}

#=== UTILITY FUNCTIONS =========================================================

#' Look Up a Torch Optimizer by Class Name
#'
#' @description
#' Maps an optimizer class name string (e.g., \code{"optim_adam"}) to the
#' corresponding torch constructor function. Used internally by
#' \code{\link{scorch_load}} to reconstruct optimizers from saved models.
#'
#' The lookup table covers all standard optimizers provided by the
#' \pkg{torch} package. Custom or third-party optimizers will return
#' \code{NULL}, and the caller falls back to storing the raw state dict
#' for manual restoration.
#'
#' @param class_name Character string. The class name of the optimizer, as
#'   returned by \code{class(optimizer)[1]}.
#'
#' @returns The optimizer constructor function (e.g., \code{torch::optim_adam}),
#'   or \code{NULL} if the class name is not recognized.
#'
#' @keywords internal

optimizer_lookup <- function(class_name) {

  #- Mapping of class name strings to torch optimizer constructor functions.
  #- These are all the optimizers provided by the torch R package.
  #-
  #- If a user creates a custom optimizer (e.g., by subclassing Optimizer
  #- or using a third-party package), it won't appear here. The calling
  #- code in scorch_load() handles that case by falling back to the raw
  #- state dict with a warning.

  optimizer_map <- list(
    "optim_adam"     = torch::optim_adam,
    "optim_adamw"    = torch::optim_adamw,
    "optim_sgd"      = torch::optim_sgd,
    "optim_rmsprop"  = torch::optim_rmsprop,
    "optim_adagrad"  = torch::optim_adagrad,
    "optim_adadelta" = torch::optim_adadelta,
    "optim_rprop"    = torch::optim_rprop,
    "optim_asgd"     = torch::optim_asgd,
    "optim_lbfgs"    = torch::optim_lbfgs
  )

  optimizer_map[[class_name]]
}

#=== END =======================================================================
