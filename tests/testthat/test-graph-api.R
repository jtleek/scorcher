test_that("old string syntax and tidy syntax both build graph specs", {
  skip_if_no_torch_backend()

  old <- initiate_scorch() |>
    scorch_input("x") |>
    scorch_layer("fc1", "linear", in_features = 2, out_features = 4) |>
    scorch_layer("act1", "relu") |>
    scorch_output("act1")

  tidy <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 4, .name = fc1) |>
    scorch_layer(relu, .name = act1) |>
    scorch_output(act1)

  expect_equal(old$inputs, tidy$inputs)
  expect_equal(old$outputs, tidy$outputs)
  expect_equal(old$graph$name, tidy$graph$name)
  expect_true(all(c("node_type", "constructor", "explicit_name") %in%
                    names(scorch_spec(tidy))))
})

test_that("scorch_output captures unquoted output names", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(features) |>
    scorch_layer(linear, in_features = 4, out_features = 16, .name = hidden) |>
    scorch_block(
      linear(in_features = 16, out_features = 16),
      gelu(),
      residual = TRUE,
      .from = hidden,
      .name = residual
    ) |>
    scorch_layer(linear, in_features = 16, out_features = 1,
                 .name = prediction) |>
    scorch_output(prediction)

  expect_equal(model$outputs, "prediction")
})

test_that("as_torch returns luz-compatible module generator by default", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 1) |>
    scorch_output()

  module <- as_torch(model)
  instance <- as_torch(model, instantiate = TRUE)

  expect_true(is.function(module))
  expect_true(inherits(instance, "nn_module"))
})

test_that("scorch_create_dataloader preserves token index dtypes by default", {
  skip_if_no_torch_backend()

  x <- torch::torch_tensor(matrix(1:12, nrow = 3), dtype = torch::torch_long())
  y <- torch::torch_tensor(matrix(2:13, nrow = 3), dtype = torch::torch_long())

  dl <- scorch_create_dataloader(
    x,
    y,
    batch_size = 2,
    shuffle = FALSE
  )

  batch <- coro::collect(dl, 1)[[1]]

  expect_equal(as.character(batch$input$input$dtype),
               as.character(torch::torch_long()))
  expect_equal(as.character(batch$output$output$dtype),
               as.character(torch::torch_long()))
  expect_equal(as.numeric(batch$input$input$shape), c(2, 4))
  expect_equal(as.numeric(batch$output$output$shape), c(2, 4))
})

test_that("embedding layers accept float token tensors defensively", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(tokens) |>
    scorch_layer(embedding,
                 num_embeddings = 10,
                 embedding_dim = 4,
                 .from = tokens,
                 .name = token_emb) |>
    scorch_output(token_emb) |>
    compile_scorch()

  tokens <- torch::torch_tensor(matrix(c(1, 2, 3, 4), nrow = 2),
                                dtype = torch::torch_float())
  output <- model$nn_model(tokens)

  expect_equal(as.numeric(output$shape), c(2, 2, 4))
})

test_that("scorch_evaluate_predictions computes common metrics", {
  skip_if_no_torch_backend()

  logits <- torch::torch_tensor(rbind(c(0.1, 0.9), c(0.8, 0.2)))
  truth <- torch::torch_tensor(c(2L, 1L), dtype = torch::torch_long())

  result <- scorch_evaluate_predictions(logits, truth, metric = "accuracy")

  expect_equal(result$metric, "accuracy")
  expect_equal(result$value, 1)
})

test_that("auto names are deterministic", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 2) |>
    scorch_layer(relu) |>
    scorch_output()

  expect_equal(model$graph$name, c("linear_1", "relu_1"))
  expect_false(any(model$graph$explicit_name))
  expect_equal(model$outputs, "relu_1")
})

test_that(".from accepts c(...) references", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 2, .name = a) |>
    scorch_layer(linear, in_features = 2, out_features = 2, .from = x,
                 .name = b) |>
    scorch_concat(.from = c(a, b), .name = merged)

  expect_equal(model$graph$inputs[[3]], c("a", "b"))
})

test_that("scorch_layer supports multihead_attention", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(multihead_attention,
                 embed_dim = 4,
                 num_heads = 2,
                 causal = TRUE,
                 batch_first = TRUE,
                 .from = c(x, x, x),
                 .name = attn) |>
    scorch_output(attn) |>
    compile_scorch()

  input <- torch::torch_randn(3, 2, 4)
  output <- model$nn_model(input)

  expect_true(inherits(output, "torch_tensor"))
  expect_equal(as.numeric(output$shape), c(3, 2, 4))
  expect_equal(model$graph$constructor, "multihead_attention")
})

test_that("scorch_block expands graph layer specifications", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 4, .name = hidden) |>
    scorch_block(
      linear(in_features = 4, out_features = 4),
      gelu(),
      residual = TRUE,
      .from = hidden,
      .name = block1
    ) |>
    scorch_output(block1)

  expect_equal(tail(model$graph$name, 1), "block1")
  expect_equal(model$outputs, "block1")
  expect_equal(model$graph$constructor, c("linear", "linear", "gelu", "add_skip"))
  expect_silent(validate_scorch_graph(model))
})

test_that("scorch_block captures minGPT-style unquoted layer calls", {
  skip_if_no_torch_backend()

  n_embd <- 8L
  n_head <- 2L

  model <- initiate_scorch() |>
    scorch_input(tokens) |>
    scorch_layer(embedding,
                 num_embeddings = 16,
                 embedding_dim = n_embd,
                 .from = tokens,
                 .name = embeddings) |>
    scorch_block(
      layer_norm(normalized_shape = n_embd),
      multihead_attention(
        embed_dim = n_embd,
        num_heads = n_head,
        batch_first = TRUE,
        causal = TRUE,
        self_attention = TRUE
      ),
      dropout(p = 0.1),
      residual = TRUE,
      .from = embeddings,
      .name = attention_block
    ) |>
    scorch_output(attention_block)

  expect_equal(
    model$graph$constructor,
    c("embedding", "layer_norm", "multihead_attention", "dropout", "add_skip")
  )
  expect_equal(model$outputs, "attention_block")
  expect_silent(validate_scorch_graph(model))
})

test_that("validation catches invalid outputs", {
  skip_if_no_torch_backend()

  model <- initiate_scorch() |>
    scorch_input(x) |>
    scorch_layer(linear, in_features = 2, out_features = 2)
  model$outputs <- "missing"

  expect_error(validate_scorch_graph(model), "Output node")
  expect_equal(nrow(validate_scorch_graph(model, strict = FALSE)), 1)
})
