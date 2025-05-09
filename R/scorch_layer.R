#===============================================================================
# SCORCH LAYER
#===============================================================================

#=== MAIN FUNCTION =============================================================

#-------------------------------------------------------------------------------
# NOTES:
#
#   1. Big update was was adding "name" and "inputs" arguments to help with
#      the bookkeeping for more complex architectures
#
#   2. Also made the layer_fn argument more general, so it can now take either
#      a string or a torch module constructor.
#-------------------------------------------------------------------------------

#' Add a Layer to a Scorch Model
#'
#' Append a new layer (module) to the `scorch_model` graph. The `layer_fn` can
#' be a string (e.g., "linear", "nn_conv2d"), an unquoted torch constructor
#' (e.g., `linear`), or a `torch::nn_*` function (e.g. `nn_linear`).
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for this layer node in the graph.
#'
#' @param layer_fn Either a string prefix (e.g. "linear"), a full string name
#' (e.g. "nn_linear"), an unquoted torch constructor (e.g., `linear`), or a
#' `torch::nn_*` function (e.g. `nn_linear`).
#'
#' @param inputs Character vector of upstream node names. If `NULL`, uses the
#' sole input (if no layers yet) or the last-built layer.
#'
#' @param ... Additional parameters passed to `layer_fn()` (e.g. `in_features`,
#' `out_features`, `kernel_size`).
#'
#' @return The updated `scorch_model` with the new layer appended.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(mtcars$wt, mtcars$mpg, batch_size=16)
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("wt") |>
#'
#'   scorch_layer(
#'
#'     name = "h1",
#'     layer_fn = "linear",
#'     inputs = "wt",
#'     in_features = 1,
#'     out_features = 8
#'   )
#'
#' print(sm)
#'
#' @export

scorch_layer <- function(

    scorch_model,
    name,
    layer_fn,
    inputs = NULL, ...) {

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

        #- User passed directly (e.g., nn_linear) — keep it

    } else {

        stop("`layer_fn` must be a torch layer name or function", call. = FALSE)
    }

    #- Pick inputs

    if (is.null(inputs)) {

        if (nrow(scorch_model$graph) == 0) {

            if (length(scorch_model$inputs) != 1) {

                stop("Must specify 'inputs' when multiple inputs exist.",

                    call. = FALSE)
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

#=== SPECIFIC LAYERS ===========================================================

#' Add a Concatenation Node to a Scorch Model
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for the concat node.
#'
#' @param inputs Character vector of node names to concatenate.
#'
#' @param dim Integer dimension to concatenate along (default 1).
#'
#' @return The updated `scorch_model`.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(
#'
#'   input  = list(hp = mtcars$hp, disp = mtcars$disp),
#'   output = mtcars$mpg
#' )
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("hp") |>
#'
#'   scorch_input("disp") |>
#'
#'   scorch_concat("merged", c("hp","disp"), dim = 2)
#'
#' print(sm)
#'
#' @export

scorch_concat <- function(

    scorch_model,
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

#' Add a Multi-Head Attention Node
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for the attention node.
#'
#' @param inputs Character vector c("query", "key", "value").
#'
#' @param embed_dim Embedding dimension.
#'
#' @param num_heads Number of attention heads.
#'
#' @param ... Additional args.
#'
#' @return The updated `scorch_model`.
#'
#' @examples
#'
#' # Need
#'
#' @export

scorch_attention <- function(

    scorch_model,
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

#' Add a Transformer Encoder Layer
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for the encoder.
#'
#' @param inputs Single upstream node name.
#'
#' @param embed_dim Model dimension (d_model).
#'
#' @param num_heads Number of heads.
#'
#' @param dim_feedforward Inner feedforward dim (default 2048).
#'
#' @param dropout Dropout rate (default 0.1).
#'
#' @param ... Additional args.
#'
#' @return The updated `scorch_model`.
#'
#' @examples
#'
#' # Need
#'
#' @export

scorch_transformer_encoder <- function(

    scorch_model,
    name,
    inputs,
    embed_dim,
    num_heads,
    dim_feedforward = 2048,
    dropout = 0.1,
    ...) {

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

#' Add a Skip Connection Node
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for the skip node.
#'
#' @param inputs Character vector length 2.
#'
#' @return The updated `scorch_model`.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(
#'
#'   input  = list(x = mtcars$hp),
#'   output = mtcars$mpg
#' )
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("x") |>
#'
#'   scorch_add_skip("skip", c("x","x"))
#'
#' print(sm)
#'
#' @export

scorch_add_skip <- function(

    scorch_model,
    name,
    inputs) {

    if (length(inputs) != 2) {

        stop("`inputs` must be length 2.", call. = FALSE)
    }

    skip_mod <- torch::nn_module(

        initialize = function() {},

        forward = function(x, skip) {x + skip}
    )()

    scorch_model$graph <- tibble::add_row(

        scorch_model$graph,

        name   = name,
        module = list(skip_mod),
        inputs = list(inputs)
    )

    return(scorch_model)
}

#' Add a BatchNorm1d Layer
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for the batchnorm layer.
#'
#' @param inputs Single upstream node name.
#'
#' @param num_features Number of features (channels).
#'
#' @param ... Additional args.
#'
#' @return The updated `scorch_model`.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(mtcars$wt, mtcars$mpg)
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("wt") |>
#'
#'   scorch_batchnorm("bn", "wt", num_features = 1)
#'
#' print(sm)
#'
#' @export

scorch_batchnorm <- function(

    scorch_model,
    name,
    inputs,
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

#' Add a Dropout Layer
#'
#' @param scorch_model A `scorch_model` object.
#'
#' @param name Unique name for the dropout layer.
#'
#' @param inputs Single upstream node name.
#'
#' @param p Dropout probability (default 0.5).
#'
#' @param ... Additional args.
#'
#' @return The updated `scorch_model`.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(mtcars$wt, mtcars$mpg)
#'
#' sm <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_input("wt") |>
#'
#'   scorch_dropout("do", "wt", p = 0.2)
#'
#' print(sm)
#'
#' @export

scorch_dropout <- function(

    scorch_model,
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


#=== END =======================================================================
