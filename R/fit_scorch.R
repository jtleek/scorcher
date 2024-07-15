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
#' @param clip_grad_norm A logical value indicating whether to clip gradient
#' norms. Default is FALSE.
#'
#' @param max_norm The maximum norm value for gradient clipping. Only used if
#' clip_grad_norm is TRUE. Default is 1.
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
#'   scorch_layer(torch::nn_linear(11,5)) |>
#'
#'   scorch_layer(torch::nn_linear(5,2)) |>
#'
#'   scorch_layer(torch::nn_linear(2,5)) |>
#'
#'   scorch_layer(torch::nn_linear(5,11)) |>
#'
#'   compile_scorch() |>
#'
#'   fit_scorch()
#'
#' first_batch <- head(dl)
#'
#' test_output <- scorch_model(first_batch$input)

fit_scorch = function(scorch_model,

  loss = nn_mse_loss, loss_params = list(reduction = "mean"),

  optim = optim_adam, optim_params = list(lr = 0.001),

  num_epochs = 10, verbose = TRUE, preprocess_fn = NULL,

  clip_grad_norm = F, max_norm = 1, ...) {

  device <- if(cuda_is_available()) {

    cat("Using available GPU.\n\n")

    torch_device("cuda")

  } else {

    torch_device("cpu")
  }

  scorch_model <- scorch_model$nn_model$to(device = device)

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

        for (i in 1:length(preprocessed)) {

          assign(names(preprocessed)[i], preprocessed[[i]])
        }

      } else {

        input <- batch$input

        output <- batch$output
      }

      optim_fn$zero_grad()

      ## Diffusion

      if (exists("timesteps")) {

        pred <- scorch_model$nn_model(input, timesteps)

      } else {

        pred <- scorch_model$nn_model(input)
      }

      loss <- loss_fn(pred, output)

      loss$backward()

      if (clip_grad_norm) {

        nn_utils_clip_grad_norm_(scorch_model$parameters, max_norm)
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
