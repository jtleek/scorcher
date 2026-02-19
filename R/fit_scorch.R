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
#' @param sm A compiled \code{scorch_model} object. Must have been
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

fit_scorch <- function(sm,
                       num_epochs    = 10,
                       verbose       = TRUE,
                       preprocess_fn = NULL,
                       clip_grad     = NULL,
                       clip_params   = list(),
                       ...) {

  #- Detect device.

  device <- if (torch::cuda_is_available()) {

    message("CUDA available. Training on GPU.")

    torch::torch_device("cuda")

  } else {

    message("No GPU detected. Training on CPU.")

    torch::torch_device("cpu")
  }

  sm$nn_model <- sm$nn_model$to(device = device)

  optimizer <- sm$optimizer

  #- Determine if there are multiple loss functions.

  loss_fns <- sm$loss_fn

  multi_loss <- is.list(loss_fns)

  #- Set output names and batch count.

  outputs <- sm$outputs

  n_out <- length(outputs)

  n_batches <- length(sm$dl)

  #- Training loop.

  for (epoch in seq_len(num_epochs)) {

    total_loss <- 0

    coro::loop(for (batch in sm$dl) {

      #- Prepare inputs and targets.

      if (!is.null(preprocess_fn)) {

        p       <- preprocess_fn(batch, ...)

        inputs  <- lapply(p$input, function(x) x$to(device = device))

        tars    <- p$output

      } else {

        inputs <- lapply(batch$input, function(x) x$to(device = device))

        tars <- batch$output
      }

      #- Move targets to device.

      if (n_out == 1) {

        if (is.list(tars)) {

          tar_list <- list(tars[[1]]$to(device = device))

        } else {

          tar_list <- list(tars$to(device = device))
        }

      } else {

        tar_list <- lapply(tars, function(x) x$to(device = device))
      }

      #- Forward pass.

      optimizer$zero_grad()

      preds <- do.call(sm$nn_model, inputs)

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
                                    device = device)

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

          torch::nn_utils_clip_grad_norm_(sm$nn_model$parameters,
                                          clip_params$max_norm)

        } else if (clip_grad == "value") {

          torch::nn_utils_clip_grad_value_(sm$nn_model$parameters,
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
  }

  sm
}

#=== END =======================================================================
