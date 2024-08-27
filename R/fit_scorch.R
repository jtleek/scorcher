#===============================================================================
# FUNCTIONS FOR FITTING A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Fit a Scorch Model
#'
#' @param scorch_model A scorch model object
#'
#' @param loss A loss function from the torch package.
#' Default is `nn_mse_loss`.
#'
#' @param loss_params A list of parameters to pass to the loss function.
#' Default is `list(reduction = "mean")`.
#'
#' @param optim An optimizer from the torch package.
#' Default is `optim_adam`.
#'
#' @param optim_params A list of parameters to pass to the optimizer.
#' Default is `list(lr = 0.001)`.
#'
#' @param num_epochs The number of epochs to train for. Default is 10.
#'
#' @param verbose A logical value indicating whether to print loss results at
#' each epoch. Default is TRUE.
#'
#' @param preprocess_fn An optional function with additional, context specific
#' steps to preprocess batches of data before training. The function must
#' return a list containing at least `input` and `output`.
#'
#' @param clip_grad A character string specifying the gradient clipping
#' strategy. Options are `"norm"` or `"value"`. Defaults to NULL (no clipping).
#'
#' @param clip_params A list of parameters for gradient clipping.
#'  - For `"norm"`: should include `max_norm` (a numeric value).
#'  - For `"value"`: should include `clip_value` (a numeric value).
#'
#' @param ... Additional arguments passed to the `preprocess_fn`.
#'
#' @return A trained scorch model.
#'
#' @export
#'
#' @examples
#'
#' input  <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' output <- mtcars |> as.matrix() |> torch::torch_tensor()
#'
#' dl <- scorch_create_dataloader(input,output,batch_size=2)
#'
#' scorch_model <- dl |> initiate_scorch() |>
#'
#'   scorch_layer("linear", 11, 5) |>
#'
#'   scorch_layer("linear", 5, 2) |>
#'
#'   scorch_layer("linear", 2, 5) |>
#'
#'   scorch_layer("linear", 5, 11) |>
#'
#'   compile_scorch() |>
#'
#'   fit_scorch()
#'
#' first_batch <- head(dl)
#'
#' test_output <- scorch_model(first_batch$input)

fit_scorch <- function(scorch_model,

  loss = nn_mse_loss, loss_params = list(reduction = "mean"),

  optim = optim_adam, optim_params = list(lr = 0.001),

  num_epochs = 10, verbose = TRUE, preprocess_fn = NULL,

  clip_grad = NULL, clip_params = list(), ...) {

  loss_fn <- do.call(loss, loss_params)

  optim_params <- append(list(

    params = scorch_model$nn_model$parameters), optim_params)

  optim_fn <- do.call(optim, optim_params)

  length_dl <- length(scorch_model$dl)

  for (epoch in 1:num_epochs) {

    total_loss = 0

    coro::loop(for (batch in scorch_model$dl) {

      if (!is.null(preprocess_fn)) {

        preprocessed <- preprocess_fn(batch, ...)

        inputs <- preprocessed[!names(preprocessed) %in% "output"]

        output <- preprocessed$output

      } else {

        inputs <- list(input = batch$input)

        output <- batch$output
      }

      optim_fn$zero_grad()

      pred <- do.call(scorch_model$nn_model, c(inputs, list()))

      loss <- loss_fn(pred, output)

      loss$backward()

      if (!is.null(clip_grad)) {

        if (clip_grad == "norm") {

          nn_utils_clip_grad_norm_(

            scorch_model$nn_model$parameters, clip_params$max_norm)

        } else if (clip_grad == "value") {

          nn_utils_clip_grad_value_(

            scorch_model$nn_model$parameters, clip_params$clip_value)

        } else {

          stop("Unsupported gradient clipping strategy. Use 'norm' or 'value'.")
        }
      }

      optim_fn$step()

      total_loss <- total_loss + loss$item()
    })

    if(verbose){

      cat(glue::glue("Epoch {crayon::red(epoch)}, ",

        "Loss: {crayon::red(total_loss/length_dl)} \n\n"))
    }
  }

  return(scorch_model$nn_model)
}

#=== END =======================================================================
