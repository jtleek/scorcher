#===============================================================================
# FUNCTION TO FIT A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Fit a Scorch Model
#'
#' @description
#' Trains a compiled Scorch model using the attached dataloader,
#' optimizer, and loss function. Supports single-output and
#' multi-output (multi-head) architectures.
#'
#' @param scorch_model A compiled \code{scorch_model} object. Must have been
#'   processed by \code{\link{compile_scorch}} before fitting.
#'
#' @param num_epochs Integer. Number of training epochs (default 10).
#'
#' @param verbose Logical. If \code{TRUE} (default), prints average
#'   loss after each epoch.
#'
#' @param preprocess_fn Optional function for custom batch
#'   preprocessing. Receives a batch and \code{...}, must return a
#'   list with \code{input} and \code{output} elements.
#'
#' @param clip_grad Character string specifying gradient clipping
#'   strategy: \code{"norm"} for max-norm clipping, \code{"value"}
#'   for value clipping, or \code{NULL} (default) for no clipping.
#'
#' @param clip_params Named list of clipping parameters. For
#'   \code{"norm"}: \code{list(max_norm = 1.0)}. For \code{"value"}:
#'   \code{list(clip_value = 0.5)}.
#'
#' @param device Character. Device to train on. Use \code{"auto"} to select
#'   CUDA when available and CPU otherwise, or specify \code{"cpu"} or
#'   \code{"cuda"}.
#'
#' @param seed Optional integer seed used to make the training run more
#'   reproducible.
#'
#' @param ... Additional arguments passed to \code{preprocess_fn}.
#'
#' @returns The trained \code{scorch_model} with its \code{nn_model}
#'   weights updated in place.
#'
#' @details
#' The training loop performs the following for each epoch:
#'   \enumerate{
#'     \item Iterates over batches from the attached dataloader.
#'     \item Moves inputs and targets to the appropriate device
#'       (CUDA if available, otherwise CPU).
#'     \item Computes predictions via the forward pass.
#'     \item Computes loss -- either a single loss function or the
#'       sum of per-output losses for multi-head models.
#'     \item Backpropagates and updates parameters.
#'   }
#'
#' For multi-output models, \code{compile_scorch} must have received
#' a named list of loss functions matching the output node names.
#' The total loss is the sum across all outputs.
#'
#' @examples
#' \dontrun{
#' # Basic training
#' model <- fit_scorch(model, num_epochs = 20)
#'
#' # With gradient clipping and custom preprocessing
#' model <- fit_scorch(
#'   model,
#'   num_epochs    = 50,
#'   preprocess_fn = my_preprocess,
#'   clip_grad     = "norm",
#'   clip_params   = list(max_norm = 1.0)
#' )
#' }
#'
#' @family model training
#'
#' @export

fit_scorch <- function(scorch_model,
                       num_epochs    = 10,
                       verbose       = TRUE,
                       preprocess_fn = NULL,
                       clip_grad     = NULL,
                       clip_params   = list(),
                       device        = "auto",
                       seed          = NULL,
                       ...) {

  scorch_model <- scorch_check_model(scorch_model)

  #- Validate that the model has been compiled.

  if (!isTRUE(scorch_model$compiled))
    stop("Model must be compiled with compile_scorch() before fitting.",
         call. = FALSE)

  if (is.null(scorch_model$dl))
    stop("No dataloader attached. Use initiate_scorch(dl = ...) to attach one.",
         call. = FALSE)

  if (!is.null(seed)) {
    set.seed(seed)
    torch::torch_manual_seed(seed)
  }

  #- Detect or validate device.

  device_name <- if (identical(device, "auto")) {
    if (torch::cuda_is_available()) "cuda" else "cpu"
  } else {
    as.character(device)
  }

  if (device_name == "cuda" && !torch::cuda_is_available()) {
    stop("CUDA was requested but is not available.", call. = FALSE)
  }

  if (verbose) {
    if (device_name == "cuda") {
      message("CUDA available. Training on GPU.")
    } else {
      message("Training on CPU.")
    }
  }

  torch_device <- torch::torch_device(device_name)

  scorch_model$nn_model <- scorch_model$nn_model$to(device = torch_device)

  normalize_batch <- function(batch) {
    if (!is.null(preprocess_fn)) {
      p <- preprocess_fn(batch, ...)
      if (!is.list(p) || is.null(p$input) || is.null(p$output)) {
        stop("`preprocess_fn` must return list(input = ..., output = ...).",
             call. = FALSE)
      }
      p
    } else {
      batch
    }
  }

  move_inputs <- function(x) {
    scorch_move_tensor_list(x, device = torch_device, default_name = "input")
  }

  move_outputs <- function(x) {
    scorch_move_tensor_list(x, device = torch_device, default_name = "output")
  }

  #- Recreate optimizer so it references the on-device parameters.
  #- After $to(device), the old optimizer still points to stale CPU tensors.
  #- For loaded models (via scorch_load), optimizer_fn may be NULL - fall back
  #- to reusing the existing optimizer directly.

  if (!is.null(scorch_model$optimizer_fn)) {

    optimizer <- do.call(scorch_model$optimizer_fn,
                         c(list(params = scorch_model$nn_model$parameters),
                           scorch_model$optimizer_params))

  } else {

    optimizer <- scorch_model$optimizer
  }


  #- Determine if there are multiple loss functions.

  loss_fns <- scorch_model$loss_fn

  multi_loss <- is.list(loss_fns)

  #- Set output names and batch count.

  outputs <- scorch_model$outputs

  n_out <- length(outputs)

  n_batches <- length(scorch_model$dl)

  history <- vector("list", num_epochs)

  #- Training loop.

  for (epoch in seq_len(num_epochs)) {

    total_loss <- 0

    coro::loop(for (batch in scorch_model$dl) {

      #- Prepare inputs and targets.

      p <- normalize_batch(batch)
      inputs <- move_inputs(p$input)
      tars <- p$output

      #- Move targets to device.

      targets <- move_outputs(tars)

      if (n_out == 1) {

        tar_list <- list(targets[[1]])

      } else {

        tar_list <- targets
      }

      #- Forward pass.

      optimizer$zero_grad()

      preds <- do.call(scorch_model$nn_model, inputs)

      #- Ensure preds is a list.

      if (n_out == 1) {

        pred_list <- list(preds)

      } else {

        pred_list <- preds
      }

      #- Compute loss.

      if (multi_loss) {

        #- Named list of losses: sum across all outputs.

        loss <- torch::torch_tensor(0, dtype = torch::torch_float(),
                                    device = torch_device)

        for (i in seq_along(outputs)) {

          nm   <- outputs[i]
          lf   <- loss_fns[[nm]]
          pl   <- pred_list[[i]]
          tl   <- tar_list[[i]]
          loss <- loss + lf(pl, tl)
        }

      } else {

        #- Single loss.

        loss <- loss_fns(pred_list[[1]], tar_list[[1]])
      }

      loss$backward()

      #- Gradient clipping.

      if (!is.null(clip_grad)) {

        if (clip_grad == "norm") {

          torch::nn_utils_clip_grad_norm_(scorch_model$nn_model$parameters,
                                          clip_params$max_norm)

        } else if (clip_grad == "value") {

          torch::nn_utils_clip_grad_value_(scorch_model$nn_model$parameters,
                                           clip_params$clip_value)
        }
      }

      optimizer$step()

      total_loss <- total_loss + loss$item()
    })

    if (verbose) {

      avg_loss <- total_loss / n_batches

      message(sprintf("Epoch %2d/%2d -- avg loss: %.4f",
                      epoch, num_epochs, avg_loss))
    }

    history[[epoch]] <- data.frame(
      epoch = epoch,
      loss = total_loss / n_batches,
      backend = "torch",
      device = device_name,
      stringsAsFactors = FALSE
    )
  }

  #- Write the trained optimizer back so scorch_save captures its state.

  scorch_model$optimizer <- optimizer
  scorch_model$history <- do.call(rbind, history)
  scorch_model$metadata$training <- list(
    backend = "torch",
    device = device_name,
    seed = seed,
    epochs = num_epochs,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )

  scorch_model
}

#=== END =======================================================================
