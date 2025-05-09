#===============================================================================
# FUNCTIONS FOR FITTING A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#-------------------------------------------------------------------------------
# NOTES:
#
# Now supports either:
#
#   1. A single loss function (nn_module) for one output, or
#
#   2. A named list of loss functions, one per output name
#-------------------------------------------------------------------------------

#' Fit a Scorch Model
#'
#' Train a compiled scorch model for a given number of epochs. Supports either
#' a single loss function (single-output) or a named list of loss functions
#' (multi-head).
#'
#' @param sm A compiled `scorch_model` (output of \link{compile_scorch}).
#'
#' @param num_epochs Integer number of training epochs.
#'
#' @param verbose Logical; if `TRUE`, prints additional training details.
#'
#' @param preprocess_fn Optional function to transform batches before training.
#'
#' @param clip_grad One of `"norm"`, `"value"`, or `NULL` for gradient clipping.
#'
#' @param clip_params Named list of clipping parameters
#' (`max_norm` or `clip_value`).
#'
#' @param ... Additional arguments passed to the `preprocess_fn`, if provided.
#'
#' @return The trained `scorch_model` with updated `nn_model` weights.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(mtcars$wt, mtcars$mpg, batch_size = 16)
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("wt") |>
#'
#'   scorch_layer("h1", "linear", in_features = 1, out_features = 8) |>
#'
#'   scorch_output("h1") |>
#'
#'   compile_scorch()
#'
#' sm <- fit_scorch(sm, num_epochs = 2, verbose = FALSE)
#'
#' print(sm)
#'
#' @export

fit_scorch <- function(

    sm,
    num_epochs    = 10,
    verbose       = TRUE,
    preprocess_fn = NULL,
    clip_grad     = NULL,
    clip_params   = list(),
    ...) {

    device <- if (cuda_is_available()) {

        torch_device("cuda")

    } else {

        torch_device("cpu")
    }

    sm$nn_model <- sm$nn_model$to(device = device)

    optimizer <- sm$optimizer

    #- Determine if there are multiple loss functions

    loss_fns <- sm$loss_fn

    multi_loss <- is.list(loss_fns)

    #- Set output/batches

    outputs <- sm$outputs

    n_out <- length(outputs)

    n_batches <- length(sm$dl)

    #- Training loop

    for (epoch in seq_len(num_epochs)) {

        total_loss <- 0

        coro::loop(for (batch in sm$dl) {

            #- Prepare inputs/targets

            if (!is.null(preprocess_fn)) {

                p       <- preprocess_fn(batch, ...)

                inputs  <- lapply(p$input, function(x) x$to(device = device))

                tars    <- p$output

            } else {

                inputs <- lapply(batch$input, function(x) x$to(device = device))

                tars <- batch$output
            }

            #- Move targets to device

            if (n_out == 1) {

                #- Single output: Either tensor or list of length 1

                if (is.list(tars)) {

                    tar_list <- list(tars[[1]]$to(device = device))

                } else {

                    tar_list <- list(tars$to(device = device))
                }

            } else {

                tar_list <- lapply(tars, function(x) x$to(device = device))
            }

            #- Forward/backward

            optimizer$zero_grad()

            preds <- do.call(sm$nn_model, inputs)

            #- Ensure preds is a list

            pred_list <- if (n_out == 1) list(preds) else preds

            #- Compute loss

            if (multi_loss) {

                #- Named list of losses

                loss <- torch_tensor(0, dtype = torch_float(), device = device)

                for (i in seq_along(outputs)) {

                    nm   <- outputs[i]
                    lf   <- loss_fns[[nm]]
                    pl   <- pred_list[[i]]
                    tl   <- tar_list[[i]]
                    loss <- loss + lf(pl, tl)
                }

            } else {

                #- Single loss

                lf   <- loss_fns

                loss <- lf(pred_list[[1]], tar_list[[1]])
            }

            loss$backward()

            #- Gradient clipping

            if (!is.null(clip_grad)) {

                if (clip_grad == "norm") {

                    nn_utils_clip_grad_norm_(

                        sm$nn_model$parameters, clip_params$max_norm)

                } else if (clip_grad == "value") {

                    nn_utils_clip_grad_value_(

                        sm$nn_model$parameters, clip_params$clip_value)
                }
            }

            optimizer$step()

            total_loss <- total_loss + loss$item()
        })

        if (verbose) {

            avg_loss <- total_loss / n_batches

            cat(sprintf("Epoch %2d/%2d - avg loss: %.4f\n",

                epoch, num_epochs, avg_loss))
        }
    }

    return(sm)
}

#=== END =======================================================================
