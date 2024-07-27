#===============================================================================
# FUNCTIONS FOR FITTING A SCORCH MODEL
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Fit a Scorch Model
#'
#' @param scorch_model A scorch model object
#'
#' @param loss A loss function from the torch package
#'
#' @param loss_params Parameters to pass to the loss function
#'
#' @param optim An optimizer from the torch package
#'
#' @param optim_params Parameters to pass to the optimizer
#'
#' @param num_epochs The number of epochs to train for
#'
#' @param verbose Whether to print loss results at each epoch
#'
#' @return A trained nn_module
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

  loss_fn <- do.call(loss, loss_params)

  optim_params <- append(list(

    params = scorch_model$nn_model$parameters), optim_params)

  optim_fn <- do.call(optim, optim_params)

  length_dl <- length(scorch_model$dl)

  for (epoch in 1:num_epochs) {

    total_loss = 0

    coro::loop(for (batch in scorch_model$dl) {

      if (!is.null(preprocess_fn)) {

        ## preprocess_fun must return a list with at minimum `input` and `output`

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
