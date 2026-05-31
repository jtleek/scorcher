test_that("scorch_block accepts custom nn_modules", {
  skip_if_no_torch_backend()

  residual <- torch::nn_module(
    initialize = function(width) {
      self$fc <- torch::nn_linear(width, width)
    },
    forward = function(x) {
      x + self$fc(x)
    }
  )

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_block(residual, width = 2, .name = res) |>
    scorch_output()

  expect_equal(model$graph$name, "res")
  expect_equal(model$graph$node_type, "block")
  expect_silent(validate_scorch_graph(model))
})

test_that("scorch_snapshot and audit create reproducibility metadata", {
  skip_if_no_torch_backend()
  skip_if_not_installed("digest")

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 1) |>
    scorch_output()

  run <- scorch_snapshot(model, data = data.frame(x = 1:3))
  audit <- scorch_audit(run)

  expect_s3_class(run, "scorch_run")
  expect_true(nzchar(run$graph_hash))
  expect_true(all(c("check", "status", "message") %in% names(audit)))
})

test_that("plot methods return ggplot objects", {
  skip_if_no_torch_backend()
  skip_if_not_installed("ggplot2")
  skip_if_not_installed("digest")

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 1) |>
    scorch_output()

  run <- scorch_snapshot(model)

  expect_s3_class(autoplot(model), "ggplot")
  expect_s3_class(autoplot(model, type = "parameters"), "ggplot")
  expect_s3_class(autoplot(run, type = "audit"), "ggplot")
})

test_that("plot.scorch_model returns a DiagrammeR widget", {
  skip_if_no_torch_backend()
  skip_if_not_installed("DiagrammeR")

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 1) |>
    scorch_output()

  plt <- plot(model, input_shapes = list(x = "batch x 2"))

  expect_s3_class(plt, "htmlwidget")
})

test_that("plot.scorch_model supports simple detail view", {
  skip_if_no_torch_backend()
  skip_if_not_installed("DiagrammeR")

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 1) |>
    scorch_output()

  plt <- plot(model, detail = "simple")

  expect_s3_class(plt, "htmlwidget")
})

test_that("diffusion preprocessing returns model inputs and outputs", {
  skip_if_no_torch_backend()

  scheduler <- NoiseScheduler(num_timesteps = 5L)
  x <- torch::torch_randn(4, 2)
  batch <- list(input = list(input = x), output = list(output = x))

  processed <- scorch_2d_diffusion_train(batch, scheduler)

  expect_true(all(c("input", "output") %in% names(processed)))
  expect_true(all(c("input", "timesteps") %in% names(processed$input)))
  expect_true(inherits(processed$output, "torch_tensor"))
})

test_that("PositionalEmbedding accepts size and emb_size aliases", {
  skip_if_no_torch_backend()

  x <- torch::torch_tensor(1:4, dtype = torch::torch_float())

  by_size <- PositionalEmbedding(size = 8, type = "sinusoidal")
  by_alias <- PositionalEmbedding(emb_size = 8, type = "sinusoidal")

  expect_equal(as.numeric(by_size(x)$shape), c(4, 8))
  expect_equal(as.numeric(by_alias(x)$shape), c(4, 8))
})

test_that("diffusion sampling loop produces xy samples", {
  skip_if_no_torch_backend()

  input_column <- function(x, column) x[, column]
  emb_size <- 8
  hidden_size <- 8
  input <- output <- torch::torch_randn(8, 2)
  dl <- scorch_create_dataloader(input, output, batch_size = 4)

  model <- initiate_scorch(dl) |>
    scorch_input(input) |>
    scorch_input(timesteps) |>
    scorch_function(input_column, column = 1, .from = input, .name = input_x) |>
    scorch_function(input_column, column = 2, .from = input, .name = input_y) |>
    scorch_layer(PositionalEmbedding, size = emb_size, type = "sinusoidal",
                 scale = 25, .from = input_x, .name = input_x_emb) |>
    scorch_layer(PositionalEmbedding, size = emb_size, type = "sinusoidal",
                 scale = 25, .from = input_y, .name = input_y_emb) |>
    scorch_layer(PositionalEmbedding, size = emb_size, type = "sinusoidal",
                 .from = timesteps, .name = timestep_emb) |>
    scorch_concat(.from = c(input_x_emb, input_y_emb, timestep_emb),
                  dim = 2, .name = features) |>
    scorch_layer(linear, in_features = emb_size * 3, out_features = hidden_size,
                 .from = features, .name = input_projection) |>
    scorch_layer(gelu, .name = input_activation) |>
    scorch_block(
      linear(in_features = hidden_size, out_features = hidden_size),
      gelu(),
      residual = TRUE,
      repeats = 1,
      .name = residual_mlp,
      .from = input_activation
    ) |>
    scorch_layer(linear, in_features = hidden_size, out_features = 2,
                 .from = residual_mlp,
                 .name = predicted_noise) |>
    scorch_output(predicted_noise) |>
    compile_scorch(loss_fn = nn_mse_loss(), optimizer_fn = optim_adamw)

  noise_scheduler <- NoiseScheduler(num_timesteps = 2L)
  generated <- torch::torch_randn(5, 2)

  for (t in rev(seq_len(noise_scheduler$num_timesteps) - 1L)) {
    t_tensor <- torch::torch_tensor(rep(t, 5), dtype = torch::torch_long())
    predicted_noise <- scorch_predict(
      model,
      input = list(input = generated, timesteps = t_tensor)
    )
    generated <- noise_scheduler$step(predicted_noise, t, generated)
  }

  expect_equal(as.numeric(generated$shape), c(5, 2))
})
