#===============================================================================
# HELPER FUNCTIONS TO IMPLEMENT DIFFUSION MODELS
#===============================================================================

#=== SCORCH PROBABILISTIC DIFFUSION MODEL FOR 2D DATASETS ======================

#--- NOISE SCHEDULER -----------------------------------------------------------

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


scorch_2d_diffusion_init <- function(model, emb_size = 128, time_emb = "sinusoidal", input_emb = "sinusoidal", scale = 25) {

  model$self$time_mlp <- PositionalEmbedding(emb_size, type = time_emb)

  model$self$input_mlp1 <- PositionalEmbedding(emb_size, type = input_emb, scale = scale)

  model$self$input_mlp2 <- PositionalEmbedding(emb_size, type = input_emb, scale = scale)
}

scorch_2d_diffusion_forward <- function(model, input, timestep) {

  x1_emb <- model$self$input_mlp1(input[, 1])

  x2_emb <- model$self$input_mlp2(input[, 2])

  t_emb  <- model$self$time_mlp(timestep)

  input  <- torch_cat(list(x1_emb, x2_emb, t_emb), dim = -1)

  input
}

#--- BATCH PREPROCESSING FOR TRAINING STEP -------------------------------------

scorch_2d_diffusion_train <- function(batch, noise_scheduler, ...) {

  noise <- torch_randn(batch$input$shape)

  timesteps <- torch_randint(0, noise_scheduler$num_timesteps,

    list(batch$input$shape[1])) |> torch_tensor(dtype = torch_long())

  noisy <- noise_scheduler$add_noise(batch$input, noise, timesteps)

  list(input = noisy, output = noise, timesteps = timesteps)
}

#=== END =======================================================================
