#===============================================================================
# HELPER FUNCTIONS TO IMPLEMENT DIFFUSION MODELS
#===============================================================================

#=== SCORCH PROBABILISTIC DIFFUSION MODEL FOR 2D DATASETS ======================

#--- NOISE SCHEDULER -----------------------------------------------------------

#' Noise Scheduler for Diffusion Process
#'
#' The `NoiseScheduler` module handles the scheduling of noise addition in the
#' diffusion process. It supports linear and quadratic beta schedules and
#' provides methods for reconstructing the original signal, computing posterior
#' means, and adding noise to samples.
#'
#' @param num_timesteps The number of timesteps in the diffusion process.
#' Default is 1000.
#'
#' @param beta_start The starting value of beta for the noise schedul.
#' Default is 0.0001.
#'
#' @param beta_end The ending value of beta for the noise schedule.
#' Default is 0.02.
#'
#' @param beta_schedule The schedule type for beta values, either "linear" or
#' "quadratic". Default is "linear".
#'
#' @return A `NoiseScheduler` object with methods for managing the diffusion
#' process.
#'
#' @references <https://github.com/tanelp/tiny-diffusion>
#'
#' @export

NoiseScheduler <- nn_module(

  initialize = function(num_timesteps = 1000,

    beta_start = 0.0001, beta_end = 0.02, beta_schedule = "linear") {

    self$num_timesteps <- num_timesteps

    if (beta_schedule == "linear") {

      self$betas <- torch_linspace(beta_start, beta_end, num_timesteps,

        dtype = torch_float())

    } else if (beta_schedule == "quadratic") {

      self$betas <- torch_linspace(sqrt(beta_start), sqrt(beta_end),

        num_timesteps, dtype = torch_float())^2
    }

    self$alphas <- 1.0 - self$betas

    self$alphas_cumprod <- torch_cumprod(self$alphas, dim = 1)

    self$alphas_cumprod_prev <- torch_cat(list(torch_tensor(1.0,

      dtype = torch_float()), self$alphas_cumprod[1:(num_timesteps - 1)]))

    ## Required for self$add_noise

    self$sqrt_alphas_cumprod <- torch_sqrt(self$alphas_cumprod)

    self$sqrt_one_minus_alphas_cumprod <- torch_sqrt(1 - self$alphas_cumprod)

    # Required for reconstruct_x0

    self$sqrt_inv_alphas_cumprod <- torch_sqrt(1 / self$alphas_cumprod)

    self$sqrt_inv_alphas_cumprod_minus_one <- torch_sqrt(1 /

      self$alphas_cumprod - 1)

    # Required for q_posterior

    self$posterior_mean_coef1 <- self$betas *

      torch_sqrt(self$alphas_cumprod_prev) / (1.0 - self$alphas_cumprod)

    self$posterior_mean_coef2 <- (1.0 - self$alphas_cumprod_prev) *

      torch_sqrt(self$alphas) / (1.0 - self$alphas_cumprod)
  },

  reconstruct_x0 = function(x_t, t, noise) {

    s1 <- self$sqrt_inv_alphas_cumprod[t + 1]

    s2 <- self$sqrt_inv_alphas_cumprod_minus_one[t + 1]

    s1 <- s1$unsqueeze(-1)

    s2 <- s2$unsqueeze(-1)

    s1 * x_t - s2 * noise
  },

  q_posterior = function(x_0, x_t, t) {

    s1 <- self$posterior_mean_coef1[t + 1]

    s2 <- self$posterior_mean_coef2[t + 1]

    s1 <- s1$unsqueeze(-1)

    s2 <- s2$unsqueeze(-1)

    s1 * x_0 + s2 * x_t
  },

  get_variance = function(t) {

    if (t == 0) {

      return(torch_tensor(0, dtype = torch_float()))
    }

    variance <- self$betas[t] *

      (1.0 - self$alphas_cumprod_prev[t]) / (1.0 - self$alphas_cumprod[t])

    variance$clamp(min = 1e-20)
  },

  step = function(model_output, timestep, sample) {

    t <- timestep

    pred_original_sample <- self$reconstruct_x0(sample, t, model_output)

    pred_prev_sample <- self$q_posterior(pred_original_sample, sample, t)

    variance <- torch_tensor(0, dtype = torch_float())

    if (t > 0) {

      noise <- torch_randn_like(model_output)

      variance <- torch_sqrt(self$get_variance(t)) * noise
    }

    pred_prev_sample + variance
  },

  add_noise = function(x_start, x_noise, timesteps) {

    s1 <- self$sqrt_alphas_cumprod[timesteps + 1L]

    s2 <- self$sqrt_one_minus_alphas_cumprod[timesteps + 1L]

    s1 <- s1$unsqueeze(-1)

    s2 <- s2$unsqueeze(-1)

    s1 * x_start + s2 * x_noise
  }
)

#--- INPUT PREPROCESSING FOR COMPILE STEP --------------------------------------

#' Helper to Initialize Scorch 2D Diffusion Model
#'
#' The `scorch_2d_diffusion_init` function can be passed to `compile_scorch`
#' and provides additional functionality for initializing a Scorch 2D diffusion
#' model by setting up positional embeddings for the input and time steps.
#'
#' @param model The Scorch model to initialize.
#'
#' @param emb_size The size of the embeddings. Default is 128.
#'
#' @param time_emb The type of embedding for time steps, options include
#' "sinusoidal", "linear", "learnable", "zero", and "identity".
#' Default is "sinusoidal".
#'
#' @param input_emb The type of embedding for inputs, options include
#' "sinusoidal", "linear", "learnable", "zero", and "identity".
#' Default is "sinusoidal".
#'
#' @param scale The scaling factor for input embeddings. Default is 1.0.
#'
#' @return None. The function modifies the `model` in place.
#'
#' @export

scorch_2d_diffusion_init <- function(model, emb_size = 128,

  time_emb = "sinusoidal", input_emb = "sinusoidal", scale = 1) {

  model$self$time_mlp <- PositionalEmbedding(emb_size, type = time_emb)

  model$self$input_mlp1 <- PositionalEmbedding(emb_size, type = input_emb,

    scale = scale)

  model$self$input_mlp2 <- PositionalEmbedding(emb_size, type = input_emb,

    scale = scale)
}

#' Helper for Forward Pass for Scorch 2D Diffusion Model
#'
#' The `scorch_2d_diffusion_forward` function defines the additional steps for
#' the forward pass of the Scorch 2D diffusion model. It processes the input
#' data and time step into embeddings and concatenates them for further
#' processing in the model.
#'
#' @param model The Scorch model being used.
#'
#' @param input The input data, expected to be a tensor with two dimensions.
#'
#' @param timesteps The current timesteps in the diffusion process.
#'
#' @return The processed input tensor after embedding and concatenation.
#'
#' @export

scorch_2d_diffusion_forward <- function(model, input, timesteps) {

  x1_emb <- model$self$input_mlp1(input[, 1])

  x2_emb <- model$self$input_mlp2(input[, 2])

  t_emb  <- model$self$time_mlp(timesteps)

  input  <- torch_cat(list(x1_emb, x2_emb, t_emb), dim = -1)

  input
}

#--- BATCH PREPROCESSING FOR TRAINING STEP -------------------------------------

#' Helper for Batch Preprocessing for Training in Scorch 2D Diffusion Model
#'
#' The `scorch_2d_diffusion_train` function prepares a batch of data for
#' training by adding noise to the inputs according to the current timestep and
#' generating the corresponding noisy outputs.
#'
#' @param batch A batch of data containing inputs.
#'
#' @param noise_scheduler The `NoiseScheduler` object used for adding noise.
#'
#' @param ... Additional arguments for customization.
#'
#' @return A list containing the noisy inputs, the noise added, and the
#' timesteps used.
#'
#' @export

scorch_2d_diffusion_train <- function(batch, noise_scheduler, ...) {

  noise <- torch_randn(batch$input$shape)

  timesteps <- torch_randint(0, noise_scheduler$num_timesteps,

    list(batch$input$shape[1])) |> torch_tensor(dtype = torch_long())

  noisy <- noise_scheduler$add_noise(batch$input, noise, timesteps)

  list(input = noisy, output = noise, timesteps = timesteps)
}

#=== END =======================================================================
