#===============================================================================
#
#  PROGRAM: scorcher_dev_ideas.R
#
#  AUTHOR:  Stephen Salerno (ssalerno@fredhutch.org)
#
#  PURPOSE: Playing around with some ideas for updating/adding scorcher
#           functions to handle more complex tasks (e.g., multi-input and/or
#           multi-head models, residual blocks, transformer encoders) and to
#           call down pre-trained models from HuggingFace for inference.
#
#  NOTES: - Three main sections to this script:
#
#            1. New and updated functions
#            2. Some toy examples
#            3. Stuff that isn't working (yet)
#
#         - Main updates to pre-existing functions are to allow for more
#           complex architectures. I am still using the graph-based approach,
#           but rather than storing in an igraph, I'm now storing in a tibble
#           so we can build out more info later, but it still traverses the
#           graph topology correctly.
#
#         - Main new functions are to call models from HuggingFace. Right now,
#           I can call down a model and get inference, but I am still trying to
#           figure out how to save the model checkpoint as an nn_module so we
#           can pipe additional fine-tuning scorch_layer(s) to it.
#
#         - Throughout, I have some pseudo-roxygen2 documentation just to keep
#           track of the arguments and return values, but if you think these
#           are headed in the right direction, I will fill the documentation.
#
#  UPDATED: 2025-04-24
#
#===============================================================================

#===============================================================================
# NEW AND UPDATED FUNCTIONS
#===============================================================================

#=== UPDATED FUNCTIONS FOR DEFINING AND FITTING A SCORCHER MODEL ===============

#--- INITIATE_SCORCH -----------------------------------------------------------

#' Initiate a Scorch Model
#'
#' Notes: - I've still kept this where it takes in a dataloader and outputs a
#'          scorch_model object, but I've:
#'
#'            1. Changed the object holding the architecture from a list to a
#'               tibble, which will store the layers in a graph structure.
#'
#'            2. Added more components to the scorch_model object to help with
#'               the bookkeeping for more complex architectures.
#'
#' @param dl Optional `torch::dataloader` to attach to the model.
#'
#' @return A `scorch_model` object.
#'
#' @export

initiate_scorch <- function(dl = NULL) {

  #- Create the base structure for a scorch_model object

  sm <- list(

    graph     = tibble::tibble(name = character(),
                               module = list(),
                               inputs = list()),
    inputs    = character(),
    outputs   = character(),
    compiled  = FALSE,
    nn_model  = NULL,
    optimizer = NULL,
    loss_fn   = NULL,
    dl        = NULL
  )

  class(sm) <- "scorch_model"

  #- If a dataloader is provided, attach it

  if (!is.null(dl)) {

    if (!inherits(dl, "dataloader")) {

      stop("`dl` must be a torch::dataloader", call. = FALSE)
    }

    sm$dl <- dl
  }

  return(sm)
}

#--- SCORCH_LAYER --------------------------------------------------------------

#' Add a Generic Layer Node
#'
#' Notes: - Big update was was adding "name" and "inputs" arguments to help
#'          with the bookkeeping for more complex architectures
#'        - Also made the layer_fn argument more general, so it can now take
#'          either a string or a torch module constructor.
#'
#' @param scorch_model A `scorch_model`.
#'
#' @param name         Unique name for this layer.
#'
#' @param layer_fn     Either a string (e.g. `"linear"`, `"conv2d"`, `"gelu"`)
#'                     or an actual `torch::nn_*` constructor.
#'
#' @param inputs       Character vector of upstream node names.  If `NULL`,
#'                     uses the *last* node or, if none, the *sole* input.
#'
#' @param ...          Additional args to pass to `layer_fn()`.
#'
#' @return The updated model.
#'
#' @export

scorch_layer <- function(scorch_model,
                         name,
                         layer_fn,
                         inputs = NULL,
                         ...) {

  #- Capture the raw arguments

  mc <- match.call()

  fn_expr <- mc$layer_fn

  if (is.symbol(fn_expr) || is.character(layer_fn)) {

    #- Either unquoted name (symbol) or a string

    fn_name <- if (is.symbol(fn_expr)) as.character(fn_expr) else layer_fn

    #- Ensure it starts with "nn_"

    if (!grepl("^nn_", fn_name)) fn_name <- paste0("nn_", fn_name)

    #- Check existence in torch namespace

    if (!exists(fn_name, envir = asNamespace("torch"), mode = "function")) {

      stop("No torch layer called '", fn_name, "'.", call. = FALSE)
    }

    layer_fn <- get(fn_name, envir = asNamespace("torch"))

  } else if (is.function(layer_fn)) {

    #- User passed e.g. nn_linear directly — keep it

  } else {

    stop("`layer_fn` must be a torch layer name or function", call. = FALSE)
  }

  #- Pick inputs

  if (is.null(inputs)) {

    if (nrow(scorch_model$graph) == 0) {

      if (length(scorch_model$inputs) != 1) {

        stop("Must specify 'inputs' when multiple inputs exist.", call. = FALSE)
      }

      inputs <- scorch_model$inputs

    } else {

      inputs <- tail(scorch_model$graph$name, 1)
    }
  }

  #- Instantiate module

  module <- do.call(layer_fn, list(...))

  #- Append to graph

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name    = name,
    module  = list(module),
    inputs  = list(inputs)
  )

  return(scorch_model)
}

#--- COMPILE_SCORCH ------------------------------------------------------------

#' Compile a Scorch Model
#'
#' Notes: - Main update is that the model now compiles by traversing the
#'          topology defined by the scorch_model graph (tibble), rather than
#'          sequentially adding layers according to their list position.
#'        - Also now handles multiple inputs/outputs.
#'
#' @param sm               A `scorch_model` built with the above.
#'
#' @param loss_fn          A loss (e.g. `nn_mse_loss()`).
#'
#' @param optimizer_fn     An optimizer constructor (e.g. `optim_adam`).
#'
#' @param optimizer_params Named list of optimizer params.
#'
#' @return The same `scorch_model` with `nn_model`, `optimizer`, `loss_fn` set.
#'
#' @export

compile_scorch <- function(sm,
                           loss_fn = nn_mse_loss(),
                           optimizer_fn = optim_adam,
                           optimizer_params = list(lr = 1e-3)) {

  graph   <- sm$graph
  inputs  <- sm$inputs
  outputs <- sm$outputs

  mod <- torch::nn_module(

    initialize = function() {

      for (i in seq_len(nrow(graph))) {

        self[[graph$name[i]]] <- graph$module[[i]]
      }
    },

    forward = function(...) {

      args <- list(...)

      env  <- new.env(parent = emptyenv())

      #- Assign inputs

      if (length(inputs) == 1) {

        env[[inputs]] <- args[[1]]

      } else {

        for (nm in names(args)) env[[nm]] <- args[[nm]]
      }

      #- Compute each node

      for (i in seq_len(nrow(graph))) {

        node    <- graph[i, ]

        in_vals <- lapply(node$inputs[[1]], function(nm) env[[nm]])

        out     <- do.call(self[[node$name]], in_vals)

        env[[node$name]] <- out
      }

      #- Return outputs

      if (length(outputs) == 1) {

        env[[outputs]]

      } else {

        purrr::map(outputs, ~ env[[.x]])
      }
    }
  )

  sm$nn_model  <- mod()

  sm$optimizer <- do.call(optimizer_fn,
                          c(list(params = sm$nn_model$parameters),
                            optimizer_params))

  sm$loss_fn   <- loss_fn

  sm$compiled  <- TRUE

  return(sm)
}

#--- FIT_SCORCH ----------------------------------------------------------------

#' Fit a Scorch Model (Multi‐Head Aware)
#'
#' Notes: - Now supports either:
#'
#'           1. A single loss function (nn_module) for one output, or
#'           2. A named list of loss functions, one per output name
#'
#' @param sm            A compiled `scorch_model`.
#'
#' @param num_epochs    Number of epochs.
#'
#' @param verbose       Print loss each epoch?
#'
#' @param preprocess_fn Optional fn(batch, ...)  to list(input=…, output=…).
#'
#' @param clip_grad     "norm" or "value" or NULL.
#'
#' @param clip_params   Params for clipping (e.g. list(max_norm=1.0)).
#'
#' @param ...           Additional args passed to `preprocess_fn`.
#'
#' @return The trained `scorch_model` (with `nn_model` updated).
#'
#' @export

fit_scorch <- function(sm,
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

      if (n_out == 1) {

        pred_list <- list(preds)

      } else {

        pred_list <- preds
      }

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

          nn_utils_clip_grad_norm_(sm$nn_model$parameters,
                                   clip_params$max_norm)

        } else if (clip_grad == "value") {

          nn_utils_clip_grad_value_(sm$nn_model$parameters,
                                    clip_params$clip_value)
        }
      }

      optimizer$step()

      total_loss <- total_loss + loss$item()
    })

    if (verbose) {

      avg_loss <- total_loss / n_batches

      cat(sprintf("Epoch %2d/%2d — avg loss: %.4f\n",
                  epoch, num_epochs, avg_loss))
    }
  }

  return(sm)
}

#--- SCORCH_CREATE_DATALOADER --------------------------------------------------

#' Create a Scorch DataLoader for Multi‑Input / Multi‑Output
#'
#' Notes: - Main updates were:
#'
#'           1. To handle multiple inputs/outputs
#'           2. To make the function more flexible to the form of the data
#'              (i.e., now do not need to convert to tensor first)
#'
#'        - Separately handles input vs output conversion:
#'
#'           1. Inputs to float, add channel dims as needed
#'           2. Outputs to long (classification) or float (regression)
#'           3. 1‑based slicing preserves all dims except batch
#'
#' @param input       A torch tensor, R array, or named list thereof.
#'
#' @param output      A torch tensor, R array, or named list thereof.
#'
#' @param batch_size  Batch size (default 32).
#'
#' @param shuffle     Shuffle each epoch? (default TRUE).
#'
#' @param num_workers Number of worker processes (default 0).
#'
#' @param pin_memory  Pin memory? (default FALSE).
#'
#' @param ...         Passed to `torch::dataloader()`.
#'
#' @return A `torch::dataloader` yielding
#' `list(input=<named list>, output=<named list>)`.
#'
#' @export
#'
scorch_create_dataloader <- function(input,
                                     output,
                                     batch_size = 32,
                                     shuffle = TRUE,
                                     num_workers = 0,
                                     pin_memory = FALSE,
                                     ...) {

  #- Wrap bare tensors/arrays into named lists

  if (!is.list(input)  || inherits(input,  "torch_tensor")) {

    input  <- list(input = input)
  }

  if (!is.list(output) || inherits(output, "torch_tensor")) {

    output <- list(output = output)
  }

  #- Convert inputs to float, add channels

  make_input_tensor <- function(x) {

    #- To torch_tensor if needed

    t <- if (inherits(x, "torch_tensor")) {

      x

    } else {

      torch::torch_tensor(x, dtype = torch_float())
    }

    #- Cast any non‑float to float

    if (t$dtype != torch_float()) {

      t <- t$to(dtype = torch_float())
    }

    #- Unsqueeze channel dim for images/features

    if (t$dim() == 3L) {

      t <- t$unsqueeze(2) # (N,H,W) to (N,1,H,W)

    } else if (t$dim() == 1L) {

      t <- t$unsqueeze(2) # (N) to (N,1)
    }

    t
  }

  #- Convert outputs to long or float appropriately

  make_output_tensor <- function(x) {

    #- If torch_tensor, keep dtype for classification/regression logic

    if (inherits(x, "torch_tensor")) {

      t <- x
    } else if (is.integer(x)) {

      t <- torch::torch_tensor(x, dtype = torch_long())

    } else {

      t <- torch::torch_tensor(x, dtype = torch_float())
    }

    #- Classification: 1‑D long stays (N)

    if (t$dtype == torch_long() && t$dim() == 1L) {

      return(t)
    }

    #- Regression single‑output: 1‑D float  to (N,1)

    if (t$dtype == torch_float() && t$dim() == 1L) {

      return(t$unsqueeze(2))
    }

    #- Regression multi‑output: multi‑dim float stays

    if (t$dtype == torch_float() && t$dim() > 1L) {

      return(t)
    }

    #- If multi‑dim long (accidental int matrix), cast to float

    if (t$dtype == torch_long() && t$dim() > 1L) {

      return(t$to(dtype = torch_float()))
    }

    t
  }

  #- Apply conversions

  input <- lapply(input, make_input_tensor)

  output <- lapply(output, make_output_tensor)

  #- Define dataset

  scorch_ds <- torch::dataset(

    name = "scorch_dataset",

    initialize = function(input, output) {

      self$input <- input

      self$output <- output

      self$n <- input[[1]]$size()[1]
    },

    .getitem = function(i) {

      slice <- function(x) {

        if (x$dim() == 1L) {

          x[i]  # 1‑D  to scalar or 1‑D

        } else {

          x$narrow(1, i, 1)$squeeze(1)
        }
      }

      inp <- lapply(self$input,  slice)

      out <- lapply(self$output, slice)

      list(input = inp, output = out)
    },

    .length = function() self$n
  )

  ds <- scorch_ds(input = input, output = output)

  #- Build dataloader

  torch::dataloader(
    ds,
    batch_size  = batch_size,
    shuffle     = shuffle,
    num_workers = num_workers,
    pin_memory  = pin_memory,
    ...
  )
}

#=== NEW FUNCTIONS FOR DEFINING AND FITTING A SCORCHER MODEL ===================

#--- SCORCH_INPUT --------------------------------------------------------------

#' Add an Input Node
#'
#' Notes: - This is for bookkeeping
#'        - This is more tensorflow-y, but it made sense in my head
#'
#' @param scorch_model A `scorch_model`.
#'
#' @param name A unique name for this input.
#'
#' @return The updated model.
#'
#' @export

scorch_input <- function(scorch_model,
                         name) {

  if (name %in% scorch_model$inputs) {

    stop("Input '", name, "' already exists.", call. = FALSE)
  }

  scorch_model$inputs <- c(scorch_model$inputs, name)

  return(scorch_model)
}

#--- SCORCH_OUTPUT -------------------------------------------------------------

#' Mark Output Nodes
#'
#' Notes: - Also for bookkeeping
#'        - Also tensorflow-y
#'
#' @param scorch_model A `scorch_model`.
#'
#' @param outputs      Character vector of node names to return.
#'
#' @return The updated model.
#'
#' @export

scorch_output <- function(scorch_model,
                          outputs) {

  scorch_model$outputs <- outputs

  return(scorch_model)
}

#--- SCORCH_CONCAT -------------------------------------------------------------

#' Add a Concatenation Node
#'
#' Notes: - I know we talked about a general scorch_join that can do things
#'          like concatenation, elementwise sum/multiplication. I can keep
#'          working on this, but this was just a proof of concept.
#'
#' @param scorch_model A `scorch_model`.
#'
#' @param name         Unique name.
#'
#' @param inputs       Vector of node names to concat.
#'
#' @param dim          Dimension to concatenate along.
#'
#' @return The updated model.
#'
#' @export

scorch_concat <- function(scorch_model,
                          name,
                          inputs,
                          dim = 1) {

  concat_mod <- torch::nn_module(

    initialize = function() {},

    forward = function(...) {

      torch::torch_cat(list(...), dim = dim)
    }
  )()

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(concat_mod),
    inputs = list(inputs)
  )

  return(scorch_model)
}

#--- SCORCH_ATTENTION ----------------------------------------------------------

#' Add a Multi‑Head Attention Node
#'
#' Notes: - Haven't kicked the tires on this one yet, but thought we'd need it.
#'
#' @param scorch_model A `scorch_model`.
#'
#' @param name         Unique name.
#'
#' @param inputs       Must be c("query", "key", "value").
#'
#' @param embed_dim    Embedding dimension.
#'
#' @param num_heads    Number of heads.
#'
#' @param ...          Extra args (e.g. dropout).
#'
#' @return The updated model.
#'
#' @export

scorch_attention <- function(scorch_model,
                             name,
                             inputs,
                             embed_dim,
                             num_heads, ...) {

  attn_mod <- torch::nn_multihead_attention(embed_dim, num_heads, ...)

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(attn_mod),
    inputs = list(inputs)
  )

  return(scorch_model)
}

#=== SCORCH_TRANSFORMER ========================================================

#' Add a Transformer Encoder Layer
#'
#' Notes: - Haven't kicked the tires on this one yet, but thought we'd need it.
#'
#' @param scorch_model    A `scorch_model`.
#'
#' @param name            Unique name.
#'
#' @param inputs          Single upstream node name.
#'
#' @param embed_dim       d_model
#'
#' @param num_heads       nhead
#'
#' @param dim_feedforward Inner FF dim.
#'
#' @param dropout         Dropout rate.
#'
#' @param ...             Extra args.
#'
#' @return The updated model.
#'
#' @export

scorch_transformer_encoder <- function(scorch_model,
                                       name,
                                       inputs,
                                       embed_dim,
                                       num_heads,
                                       dim_feedforward = 2048,
                                       dropout = 0.1, ...) {

  tr_mod <- torch::nn_transformer_encoder_layer(

    d_model = embed_dim,
    nhead = num_heads,
    dim_feedforward = dim_feedforward,
    dropout = dropout,
    ...
  )

  scorch_model$graph <- tibble::add_row(
    scorch_model$graph,
    name   = name,
    module = list(tr_mod),
    inputs = list(inputs)
  )

  return(scorch_model)
}

#--- SCORCH_ADD_SKIP -----------------------------------------------------------

#' Add a Skip Connection Node
#'
#' Creates a node that sums two upstream tensors (e.g. for residual connections).
#'
#' Notes: - Haven't kicked the tires on this one yet, but thought we'd need it.
#'
#' @param scorch_model A `scorch_model`.
#'
#' @param name         Unique name for this skip node.
#'
#' @param inputs       A length‑2 character vector: c("main_path", "skip_path").
#'
#' @return The updated model.
#'
#' @export

scorch_add_skip <- function(scorch_model,
                            name,
                            inputs) {

  if (length(inputs) != 2) {

    stop("`inputs` must be length 2.", call. = FALSE)
  }

  skip_mod <- torch::nn_module(

    initialize = function() {},
    forward = function(x, skip) {
      x + skip
    }
  )()

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(skip_mod),
    inputs = list(inputs)
  )

  return(scorch_model)
}

#--- SCORCH_BATCHNORM ----------------------------------------------------------

#' Add a BatchNorm1d Node
#'
#' Notes: - Haven't kicked the tires on this one yet, but thought we'd need it.
#'
#' @param scorch_model  A `scorch_model`.
#'
#' @param name          Unique name for this batch‐norm layer.
#'
#' @param inputs        Single upstream node name.
#'
#' @param num_features  Number of features (the `C` in BatchNorm1d).
#'
#' @param ...           Additional args (e.g. momentum, eps).
#'
#' @return The updated model.
#'
#' @export

scorch_batchnorm <- function(scorch_model,
                             name, inputs,
                             num_features,
                             ...) {

  bn_mod <- torch::nn_batch_norm1d(num_features = num_features, ...)

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(bn_mod),
    inputs = list(inputs)
  )

  return(scorch_model)
}

#--- SCORCH_DROPOUT ------------------------------------------------------------

#' Add a Dropout Node
#'
#' Notes: - Haven't kicked the tires on this one yet, but thought we'd need it.
#'
#' @param scorch_model  A `scorch_model`.
#'
#' @param name          Unique name for this dropout layer.
#'
#' @param inputs        Single upstream node name.
#'
#' @param p             Dropout probability.
#'
#' @param ...           Additional args.
#'
#' @return The updated model.
#'
#' @export

scorch_dropout <- function(scorch_model,
                           name,
                           inputs,
                           p = 0.5,
                           ...) {

  do_mod <- torch::nn_dropout(p = p, ...)

  scorch_model$graph <- tibble::add_row(

    scorch_model$graph,
    name   = name,
    module = list(do_mod),
    inputs = list(inputs)
  )

  return(scorch_model)
}

#=== PRINT/PLOT METHODS ========================================================

#--- PRINT ---------------------------------------------------------------------

#' Notes: - This is not nearly as nice as the print method you already had, but
#'          I am just showing here that we can exploit storing the architecture
#'          as a graph in a tibble, so maybe we can add more info later?
#'
#' @export

print.scorch_model <- function(x,
                               detailed = FALSE,
                               ...) {

  cat("<< Scorch Model >>\n")

  cat(" Inputs : ", paste(x$inputs, collapse = ", "), "\n")

  cat(" Outputs: ", paste(x$outputs, collapse = ", "), "\n\n")

  df <- x$graph |>

    dplyr::mutate(

      module_type = purrr::map_chr(name, ~ class(x$nn_model[[.x]])[1]),
      inputs      = purrr::map_chr(inputs, ~ paste(.x, collapse = ", "))
    )

  if (detailed) {

    df <- df |>

      dplyr::mutate(

        dims = purrr::map_chr(name, function(nm) {

          mod <- x$nn_model[[nm]]

          if (!is.null(mod$weight) && inherits(mod$weight, "torch_tensor")) {

            paste(mod$weight$size(), collapse = "×")

          } else {

            ""
          }
        })
      ) |>

      dplyr::select(name, inputs, module_type, dims)

  } else {

    df <- df |>

      dplyr::select(name, inputs, module_type)
  }

  print(df)

  invisible(x)
}

#--- PLOT ----------------------------------------------------------------------

#' Notes: - Again, there is a lot of room for improvement, but this exploits
#'          the fact that we are representing the architecture as a graph.
#'
#' @import DiagrammeR
#'
#' @export

plot_scorch_model <- function(scorch_model,
                              detailed = FALSE) {

  nodes <- scorch_model$graph$name

  inputs <- scorch_model$inputs

  outputs <- scorch_model$outputs

  #- Node definitions

  node_defs <- c(

    #- Input nodes

    vapply(inputs, function(nm) {

      sprintf('%s [shape=oval, style=filled, fillcolor="lightblue", label="%s"]',
              nm, nm)
    }, ""),

    #- Internal/output nodes

    vapply(nodes, function(nm) {

      mod <- scorch_model$nn_model[[nm]]

      type <- class(mod)[1]

      dims_lbl <- ""

      if (detailed && !is.null(mod$weight) && inherits(mod$weight, "torch_tensor")) {

        dims_lbl <- paste0("\n(", paste(mod$weight$size(), collapse="×"), ")")
      }

      shape <- if (nm %in% outputs) "doublecircle" else "box"

      label <- if (detailed) {

        paste0(nm, "\n", type, dims_lbl)

      } else {

        paste0(nm, "\n", type)
      }

      sprintf('%s [shape=%s, label="%s"]', nm, shape, label)

    }, "")
  )

  #- Edge definitions

  edge_defs <- unlist(lapply(seq_len(nrow(scorch_model$graph)), function(i) {

    to    <- scorch_model$graph$name[i]

    froms <- scorch_model$graph$inputs[[i]]

    vapply(froms, function(fr) sprintf("%s -> %s", fr, to), "")
  }))

  dot <- paste(
    "digraph G {",
    "rankdir=LR;",
    paste(node_defs, collapse = ";\n"), ";",
    paste(edge_defs, collapse = ";\n"), ";",
    "}"
  )

  DiagrammeR::grViz(dot)
}

#=== HUGGINGFACE MODEL CALLS ===================================================

#--- SCORCH_HF_CALL ------------------------------------------------------------

#' Call a HuggingFace model via API and cache its config for fine‐tuning
#'
#' Notes: - This works if you have an API key for inference on a pre-trained
#'          model, but I still haven't figured out how to add more layers via
#'          scorcher (I have a couple attempts below).
#'
#' @param model_id A character string of the form `"username/modelname"`
#' pointing to a model repository on huggingface.co.
#'
#' @param input A list or atomic vector representing the JSON payload to send
#' to the inference endpoint (e.g. `list(inputs = "Hello world")`).
#'
#' @param api_key Your HuggingFace API token. By default it reads from the
#' `HF_API_KEY` environment variable.
#'
#' @param config_path Path (including filename) where the downloaded
#' `config.json` will be written. Defaults to a temp‐dir file named
#' `username_modelname_config.json`.
#'
#' @return A list with components:
#'   * `inference`: the parsed JSON response from the inference API
#'   * `config`: the parsed model configuration (or `NULL` if download failed)
#'   * `config_path`: the path where `config.json` was saved (even if `NULL`)
#'
#' @examples
#' \dontrun{
#' # make sure you have set Sys.setenv(HF_API_KEY = "<your token>")
#' out <- scorch_hf_call(
#'   model_id    = "distilbert-base-uncased-finetuned-sst-2-english",
#'   input       = list(inputs = "I love scorcher!"),
#'   config_path = "my_model_config.json"
#' )
#' str(out)
#' }
#'
#' @import httr
#'
#' @import jsonlite
#'
#' @export

scorch_hf_call <- function(model_id,
                           input,
                           api_key = Sys.getenv("HF_API_KEY", unset = NA),
                           config_path = file.path(tempdir(),
                                                   paste0(gsub("[^A-Za-z0-9_]",
                                                               "_", model_id),
                                                          "_config.json"))
                           ) {

  if (is.na(api_key) || nzchar(api_key) == FALSE) {

    stop("Please set your HuggingFace API key via `api_key=` or ",
         "`Sys.setenv(HF_API_KEY = <token>)`.", call. = FALSE)
  }

  #- Inference call

  infer_url <- paste0("https://api-inference.huggingface.co/models/", model_id)

  infer_res <- httr::POST(

    infer_url,
    httr::add_headers(Authorization = paste("Bearer", api_key)),
    body   = input,
    encode = "json"
  )

  if (httr::status_code(infer_res) != 200) {

    stop(
      "Inference API request failed [",
      httr::status_code(infer_res), "]: ",
      httr::content(infer_res, as = "text", encoding = "UTF-8"),
      call. = FALSE
    )
  }

  inference <- httr::content(infer_res, as = "parsed", simplifyVector = TRUE)

  #- Download config.json

  cfg_url <- paste0("https://huggingface.co/", model_id,
                    "/raw/main/config.json")

  cfg_res <- httr::GET(

    cfg_url,
    httr::add_headers(Authorization = paste("Bearer", api_key))
  )

  if (httr::status_code(cfg_res) != 200) {

    warning(
      "Failed to retrieve config.json [",
      httr::status_code(cfg_res), "]; skipping save.",
      call. = FALSE
    )

    config <- NULL

  } else {

    config <- jsonlite::fromJSON(rawToChar(cfg_res$content))

    jsonlite::write_json(config, config_path, pretty = TRUE, auto_unbox = TRUE)
  }

  list(

    inference   = inference,
    config      = config,
    config_path = config_path
  )
}

#===============================================================================
# SOME TOY EXAMPLES
#===============================================================================

library(torch)
library(torchvision)
library(palmerpenguins)
library(tidyverse)
# library(scorcher)

#--- EXAMPLE 1: MTCARS ---------------------------------------------------------

# This is just a random example (the model doesn't mean anything) just to show
# how some of the new functions work/work together and how we can get some more
# complicated architectures now.

# 1) Prepare data

df <- mtcars

engine_feats  <- as.matrix(df |> select(hp, disp, drat))
chassis_feats <- as.matrix(df |> select(wt, qsec, gear))
y_vec         <- df$mpg

a_tensor <- torch_tensor(engine_feats,  dtype = torch_float())
b_tensor <- torch_tensor(chassis_feats, dtype = torch_float())
y_tensor <- torch_tensor(y_vec,         dtype = torch_float())

# 2) Create dataloader

dl <- scorch_create_dataloader(
  input      = list(a = a_tensor, b = b_tensor),
  output     = y_tensor,
  batch_size = 16,
  shuffle    = TRUE
)

# 3) Build & compile model

model <- dl |>
  initiate_scorch() |>
  scorch_input("a") |>
  scorch_input("b") |>
  scorch_layer("dense1_a", "linear", inputs = "a", in_features = 3, out_features = 16) |> # here "linear" is a string
  scorch_batchnorm("bn1_a", inputs = "dense1_a", num_features = 16) |>
  scorch_layer("act1_a",   "relu",   inputs = "bn1_a") |>
  scorch_layer("dense1_b", "nn_linear", inputs = "b", in_features = 3, out_features = 16) |> # here "nn_linear" is a string
  scorch_dropout("drop1_b", inputs = "dense1_b", p = 0.5) |>
  scorch_layer("act1_b",   "gelu",   inputs = "drop1_b") |>
  scorch_concat("merged", inputs = c("act1_a","act1_b"), dim = 2) |>
  scorch_layer("dense2",   linear, inputs = "merged", in_features = 32, out_features = 32) |> # here linear is an unquoted object
  scorch_add_skip("res1",  inputs = c("dense2","merged")) |>
  scorch_layer("act2",     "relu",   inputs = "res1") |>
  scorch_layer("dense3",   nn_linear, inputs = "act2", in_features = 32, out_features = 1) |> # here nn_linear is a torch module
  scorch_output("dense3") |>
  compile_scorch(
    loss_fn          = nn_mse_loss(),
    optimizer_fn     = optim_adam,
    optimizer_params = list(lr = 1e-3)
  )

# 4) Fit the model

trained_model <- fit_scorch(
  model,
  num_epochs = 20,
  verbose    = TRUE
)

# 5) Print summary

print(model)
print(model, detailed = TRUE)

# 6) Plot the architecture

plot_scorch_model(model)
plot_scorch_model(model, detailed = TRUE)

#--- EXAMPLE 2: MNIST CLASSIFICATION -------------------------------------------

# This one takes a while to run but shows off more updates

# 1) Load MNIST

train_data <- torchvision::mnist_dataset(
  root      = tempdir(),
  download  = TRUE,
  transform = torchvision::transform_to_tensor
)

test_data <- torchvision::mnist_dataset(
  root      = tempdir(),
  train     = FALSE,
  download  = FALSE,
  transform = torchvision::transform_to_tensor
)

# 2) Create dataloaders (fast, correct dtypes)

train_dl <- scorch_create_dataloader(
  train_data$data,
  train_data$targets,
  batch_size  = 500,
  shuffle     = TRUE,
  num_workers = 0,
  pin_memory  = TRUE
)

test_dl <- scorch_create_dataloader(
  test_data$data,
  test_data$targets,
  batch_size  = 1000,
  shuffle     = FALSE,
  num_workers = 0,
  pin_memory  = TRUE
)

# 3) Build & compile ConvNet

mnist_model <- train_dl |>
  initiate_scorch() |>
  scorch_input("input") |>
  scorch_layer("conv1", "conv2d",
               inputs      = "input",
               in_channels = 1, out_channels = 32,
               kernel_size = 3, padding = 1) |>
  scorch_layer("act1",  "relu",       inputs = "conv1") |>
  scorch_layer("pool1", "max_pool2d", inputs = "act1", kernel_size = 2) |>
  scorch_layer("conv2", "conv2d",
               inputs      = "pool1",
               in_channels = 32, out_channels = 64,
               kernel_size = 3, padding = 1) |>
  scorch_layer("act2",  "relu",       inputs = "conv2") |>
  scorch_layer("pool2", "max_pool2d", inputs = "act2", kernel_size = 2) |>
  scorch_layer("flatten", "flatten", inputs = "pool2", start_dim = 2) |>
  scorch_layer("fc1",     "linear",
               inputs      = "flatten",
               in_features = 7*7*64, out_features = 128) |>
  scorch_layer("act3",  "relu",    inputs = "fc1") |>
  scorch_layer("drop1", "dropout", inputs = "act3", p = 0.5) |>
  scorch_layer("fc2",   "linear",
               inputs      = "drop1",
               in_features = 128, out_features = 10) |>
  scorch_output("fc2") |>
  compile_scorch(
    loss_fn          = nn_cross_entropy_loss(),
    optimizer_fn     = optim_adam,
    optimizer_params = list(lr = 1e-3)
  )

# 4) Train

mnist_model <- fit_scorch(
  mnist_model,
  num_epochs = 5,
  verbose    = TRUE
)

# 5) Evaluate

# Note: We should have a scorch_evaluate function or something

eval_accuracy <- function(dl, model) {
  model$nn_model$eval()
  correct <- 0; total <- 0
  coro::loop(for (b in dl) {
    x <- b$input$input
    y <- b$output$output
    preds <- model$nn_model(x)$argmax(dim = 2)
    correct <- correct + (preds == y)$sum()$item()
    total   <- total   + y$size()[1]
  })
  correct / total
}

cat("Test accuracy:", eval_accuracy(test_dl, mnist_model), "\n")

# 6) Inspect

print(mnist_model, detailed = TRUE)
plot_scorch_model(mnist_model, detailed = TRUE)

#--- EXAMPLE 3: PALMER PENGUINS CLASSIFICATION ---------------------------------

# 1) Prepare data (drop NAs, one‐hot encode species)

peng <- penguins |>

  filter(!is.na(bill_length_mm), !is.na(bill_depth_mm), !is.na(species)) |>
  mutate(spec_idx = as.integer(species))

X <- as.matrix(select(peng, bill_length_mm, bill_depth_mm))
y <- peng$spec_idx

# 2) DataLoader via scorcher helper

dl_peng <- scorch_create_dataloader(
  input      = list(x = X),
  output     = y,
  batch_size = 16,
  shuffle    = TRUE
)

# 3) Build & compile a small MLP

peng_model <- dl_peng |>
  initiate_scorch() |>
  scorch_input("x") |>
  scorch_layer("dense1", "linear", inputs = "x",
               in_features = 2, out_features = 16) |>
  scorch_layer("act1",   "relu", inputs = "dense1") |>
  scorch_layer("dense2", "linear", inputs = "act1",
               in_features = 16, out_features = 3) |>
  scorch_output("dense2") |>
  compile_scorch(
    loss_fn          = nn_cross_entropy_loss(),
    optimizer_fn     = optim_adam,
    optimizer_params = list(lr = 5e-3)
  )

# 4) Train & evaluate

peng_model <- fit_scorch(peng_model, num_epochs = 30)

# 5) Quick accuracy

preds <- c()
truth <- c()

peng_model$nn_model$eval()

coro::loop(for (b in dl_peng) {
  x <- b$input$x
  y <- b$output$output
  p <- peng_model$nn_model(x)$argmax(dim = 2)
  preds <- c(preds, as.integer(p))
  truth <- c(truth, as.integer(y))
})

mean(preds == truth)

#--- EXAMPLE 4: PALMER PENGUINS MULTIPLE PREDICTIONS ---------------------------

# 1) Prepare data: predict flipper_length_mm & body_mass_g from bill dims

peng2 <- penguins |>

  filter(!is.na(bill_length_mm), !is.na(bill_depth_mm),
         !is.na(flipper_length_mm), !is.na(body_mass_g))

X2 <- as.matrix(select(peng2, bill_length_mm, bill_depth_mm))
Y2 <- as.matrix(select(peng2, flipper_length_mm, body_mass_g))

# 2) Dataloader

dl_reg <- scorch_create_dataloader(
  input      = list(x = X2),
  output     = Y2,
  batch_size = 16,
  shuffle    = TRUE
)

# 3) MLP to 2‑dim output

reg_model <- dl_reg |>
  initiate_scorch() |>
  scorch_input("x") |>
  scorch_layer("h1",    "linear", inputs = "x",
               in_features = 2, out_features = 32) |>
  scorch_layer("act1",  "relu",   inputs = "h1") |>
  scorch_layer("h2",    "linear", inputs = "act1",
               in_features = 32, out_features = 16) |>
  scorch_layer("act2",  "relu",   inputs = "h2") |>
  scorch_layer("out",   "linear", inputs = "act2",
               in_features = 16, out_features = 2) |>
  scorch_output("out") |>
  compile_scorch(
    loss_fn          = nn_mse_loss(),
    optimizer_fn     = optim_adam,
    optimizer_params = list(lr = 1e-3)
  )

# 4) Train & inspect

reg_model <- fit_scorch(reg_model, num_epochs = 50, verbose = TRUE)

print(reg_model, detailed = TRUE)
plot_scorch_model(reg_model, detailed = TRUE)

#--- EXAMPLE 5: MULTI-HEAD (TWO-OUTPUT MODEL) WITH MTCARS ----------------------

# 1) Prepare data & tensors

df <- mtcars |>
  mutate(cyl_class = as.integer(factor(cyl)))  # 1,2,3

X  <- as.matrix(select(df, hp, disp, drat, wt, qsec, gear))

t_mpg <- torch_tensor(df$mpg,       dtype = torch_float())$unsqueeze(2) # (N,1)
t_cyl <- torch_tensor(df$cyl_class, dtype = torch_long())               # (N)

# 2) DataLoader (two outputs as a named list)

dl_mt <- scorch_create_dataloader(
  input  = list(x = X),
  output = list(mpg = t_mpg, cyl = t_cyl),
  batch_size  = 8,
  shuffle     = TRUE
)

# 3) Build the multi‑head architecture

multi_model <- dl_mt |>
  initiate_scorch() |>
  scorch_input("x") |>
  # shared backbone
  scorch_layer("dense1",  "linear", inputs = "x", in_features = 6, out_features = 32) |>
  scorch_layer("act1",    "relu",   inputs = "dense1") |>
  scorch_layer("dense2",  "linear", inputs = "act1", in_features = 32, out_features = 16) |>
  scorch_layer("act2",    "relu",   inputs = "dense2") |>
  # head 1  to regression mpg
  scorch_layer("mpg_head", "linear", inputs = "act2", in_features = 16, out_features = 1) |>
  # head 2  to classification cyl
  scorch_layer("cyl_head", "linear", inputs = "act2", in_features = 16, out_features = 3) |>
  scorch_output(c("mpg_head", "cyl_head")) |>
  compile_scorch(
    loss_fn = list(
      mpg_head = nn_mse_loss(),
      cyl_head = nn_cross_entropy_loss()
    ),
    optimizer_fn     = optim_adam,
    optimizer_params = list(lr = 1e-3)
  )

# 4) Fit model

multi_model <- fit_scorch(
  multi_model,
  num_epochs = 30,
  verbose    = TRUE
)

# 5) Inspect

print(multi_model, detailed = TRUE)
plot_scorch_model(multi_model, detailed = TRUE)

#--- EXAMPLE 6: HUGGINGFACE INFERENCE VIA API ----------------------------------

# make sure HF_API_KEY is set:
# Sys.setenv(HF_API_KEY = "<your_token>")

res <- scorch_hf_call(
  model_id    = "distilbert-base-uncased-finetuned-sst-2-english",
  input       = list(inputs = "Scorcher makes model building in R so easy!"),
  config_path = "hf_sst2_config.json"
)

# View inference output
print(res$inference)

# View downloaded config
str(res$config)

# Path to saved config.json
res$config_path

#===============================================================================
# STUFF THAT ISN'T WORKING (YET)
#===============================================================================

#=== FUNCTIONS =================================================================

#' Create a torch::nn_module wrapping HF feature‑extraction via the Inference API
#'
#' Fetches only config.json at init, then POSTs real text to the
#' `/pipeline/feature-extraction/{model_id}` endpoint in `forward()`.
#'
#' @param model_id Hugging Face model ID (e.g. "distilbert-base-uncased")
#' @param api_key  Your HF token (Sys.getenv("HF_API_KEY"))
#' @export
#' Create a torch::nn_module wrapping an HF feature‑extraction endpoint
#'
#' - GETs only config.json on init (no dummy inference calls)
#' - Falls back from `config$hidden_size` → `config$dim`
#' - POSTs real text to the HF `/pipeline/feature-extraction/{model_id}` API
#'
#' @param model_id Hugging Face model ID (e.g. `"distilbert-base-uncased"`
#'                 or `"distilbert-base-uncased-finetuned-sst-2-english"`)
#' @param api_key  Your HF API token (via `Sys.getenv("HF_API_KEY")`)
#' @export
scorch_hf_base_module <- function(model_id,
                                  api_key = Sys.getenv("HF_API_KEY")) {
  if (!nzchar(api_key)) {
    stop("Please set HF_API_KEY to your Hugging Face token.", call. = FALSE)
  }

  # 1) Download config.json
  cfg_url <- paste0("https://huggingface.co/", model_id, "/raw/main/config.json")
  cfg_res <- httr::GET(cfg_url)
  if (httr::status_code(cfg_res) != 200) {
    stop("Failed to download config.json for ", model_id, call. = FALSE)
  }
  config     <- jsonlite::fromJSON(httr::content(cfg_res, "text", encoding = "UTF-8"))
  hidden_size <- config$hidden_size %||% config$dim
  if (is.null(hidden_size)) {
    stop("Could not find `hidden_size` or `dim` in config.json.", call. = FALSE)
  }

  # 2) Prepare feature‑extraction endpoint
  infer_url <- paste0(
    "https://api-inference.huggingface.co/pipeline/feature-extraction/",
    model_id
  )
  inference_fn <- function(texts) {
    if (!is.character(texts)) {
      stop("`forward(input)` expects a character vector of strings.", call. = FALSE)
    }
    res <- httr::POST(
      infer_url,
      httr::add_headers(Authorization = paste("Bearer", api_key)),
      body   = list(inputs = texts),
      encode = "json"
    )
    if (httr::status_code(res) != 200) {
      msg <- httr::content(res, as = "text", encoding = "UTF-8")
      stop("Inference API error [", httr::status_code(res), "]: ", msg, call. = FALSE)
    }
    httr::content(res, as = "parsed", simplifyVector = TRUE)
  }

  # 3) Build the torch module
  torch::nn_module(
    "hf_base",
    initialize = function() {
      self$config       <- config
      self$hidden_size  <- hidden_size
      self$inference_fn <- inference_fn
    },
    forward = function(input = NULL) {
      # If called with no input (at compile time), return a dummy shape
      if (is.null(input)) {
        return(torch::torch_empty(c(1, self$hidden_size)))
      }
      # Otherwise, call the HF API
      out <- self$inference_fn(input)
      t_list <- lapply(out, function(mat) {
        torch::torch_tensor(do.call(rbind, mat),
                            dtype = torch::torch_float())$unsqueeze(1)
      })
      batch_hs <- torch::torch_cat(t_list, dim = 1)   # (batch, seq, hidden)
      cls_emb  <- batch_hs$slice(2, 1, 1)$squeeze(2)  # (batch, hidden_size)
      cls_emb
    }
  )
}

#' #' Create a torch::nn_module that wraps HuggingFace Inference API
#' #'
#' #' @param model_id  HF repo ID, e.g. "distilbert-base-uncased"
#' #'
#' #' @param api_key   Your HF token
#' #'
#' #' @return A torch nn_module with forward(x)  to embedding tensor
#' #'
#' #' @export
#'
#' scorch_hf_base_module <- function(model_id,
#'                                   api_key = Sys.getenv("HF_API_KEY")) {
#'
#'   res <- scorch_hf_call(
#'
#'     model_id,
#'     input = list(inputs = ""),
#'     api_key = api_key,
#'     config_path = tempfile("hfconfig", fileext = ".json")
#'   )
#'
#'   hidden_size <- res$config$hidden_size
#'
#'   torch::nn_module(
#'
#'     "hf_base",
#'
#'     initialize = function() {
#'       self$config      <- res$config
#'       self$model_id    <- model_id
#'       self$api_key     <- api_key
#'       self$inference_fn <- res$inference
#'     },
#'
#'     forward = function(input_ids) {
#'
#'       # call the HF inference endpoint, returns logits or embeddings
#'       out <- self$inference_fn(inputs = input_ids)
#'       # suppose out is a list with 'last_hidden_state' as a matrix [batch, seq, hidden]
#'       # here we take the CLS token embedding: out$last_hidden_state[,1,]
#'       torch_tensor(do.call(rbind, out$last_hidden_state))$slice(2, 1, 1)  # (batch,hidden_size)
#'     }
#'   )
#' }

#=== EXAMPLES ==================================================================

#--- EXAMPLE 7: BIGGER HUGGINGFACE EXAMPLE -------------------------------------

# 1) Wrap the HF base module (SST‑2 fine‑tuned)
hf_mod <- scorch_hf_base_module(
  model_id = "distilbert-base-uncased-finetuned-sst-2-english"
)()

# 2) Prepare a tiny text dataset
texts  <- c(
  "Scorcher makes model building in R so easy!",
  "I don't like debugging complex torch code..."
)
labels <- torch_tensor(c(1L, 0L), dtype = torch_long())  # 1=positive, 0=negative

text_ds <- torch::dataset(
  name = "text_ds",
  initialize = function(texts, labels) {
    self$texts  <- texts
    self$labels <- labels
    self$n      <- length(texts)
  },
  .getitem = function(i) {
    list(input  = self$texts[i],   # must be named "input"
         output = self$labels[i])
  },
  .length = function() self$n
)

dl_sent <- text_ds(texts, labels) |>
  torch::dataloader(batch_size = 2, shuffle = FALSE)

# 3) Build & compile the fine‑tuning model via scorcher
ft_model <- dl_sent |>
  initiate_scorch() |>
  scorch_input("input") |>
  scorch_layer("hf_embed", hf_mod, inputs = "input") |>
  scorch_layer("dropout", "dropout", inputs = "hf_embed", p = 0.1) |>
  scorch_layer("classifier",
               "linear",
               inputs      = "dropout",
               in_features = hf_mod$hidden_size,
               out_features = 2) |>
  scorch_output("classifier") |>
  compile_scorch(
    loss_fn          = nn_cross_entropy_loss(),
    optimizer_fn     = optim_adam,
    optimizer_params = list(lr = 2e-5)
  )


# 4) Fine‑tune for 2 epochs
ft_model <- fit_scorch(ft_model, num_epochs = 2, verbose = TRUE)

# 5) Predict on the same batch
ft_model$nn_model$eval()
batch  <- iter_next(dl_sent)
# batch$input is a character vector of length 2
logits <- ft_model$nn_model(batch$input)
preds  <- logits$argmax(dim = 2)
cat("Predicted classes:", as.integer(preds), "\n")


#=== END =======================================================================
