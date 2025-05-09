#===============================================================================
# POSITIONAL EMBEDDINGS
#===============================================================================

#=== MAIN FUNCTION =============================================================

#' Add an Embedding Layer to a Scorch Model
#'
#' The `scorch_embedding` function adds a positional embedding layer to the
#' given Scorch model. This embedding can be one of "sinusoidal", "linear",
#' "learnable", "zero", or "identity".
#'
#' @param scorch_model A scorch model object to which an embedding layer will
#' be added.
#'
#' @param size An integer specifying the size of the embedding. Default is 128.
#'
#' @param type A character string specifying the type of positional embedding
#' to use. Options include "sinusoidal", "linear", "learnable", "zero", and
#' "identity". Default is "sinusoidal".
#'
#' @param ... Additional arguments passed to the embedding function. These
#' arguments are specific to the embedding type and can include parameters
#' like `scale` for the sinusoidal embedding.
#'
#' @return The modified Scorch model with the embedding layer added to its
#' architecture.
#'
#' @examples
#'
#' dl <- scorch_create_dataloader(mtcars$wt, mtcars$mpg, batch_size = 16)
#'
#' scorch_model <- dl |>
#'
#'   initiate_scorch() |>
#'
#'   scorch_embedding(size = 128, type = "sinusoidal", scale = 1)
#'
#' @export

scorch_embedding <- function(scorch_model,
                             size = 128,
                             type = "sinusoidal", ...) {

  embed_func <- PositionalEmbedding(size, type, ...)

  class(embed_func) <- c(paste0(type, "_embedding"), class(embed_func))

  scorch_model$scorch_architecture <- append(

    scorch_model$scorch_architecture, list(embed_func, type = "layer"))

  scorch_model
}

#=== POSITIONAL EMBEDDINGS =====================================================

#--- SINUSOIDAL ----------------------------------------------------------------

#' Sinusoidal Positional Embedding
#'
#' The `SinusoidalEmbedding` module implements sinusoidal positional embeddings.
#' It scales the input and applies sine and cosine functions to create
#' embeddings.
#'
#' @param size The dimension of the embedding.
#'
#' @param scale A scaling factor applied to the input before creating the
#' embedding. Default is 1.0.
#'
#' @return A `SinusoidalEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- SinusoidalEmbedding(8)
#'
#' emb(x)
#'
#' @export
#'
SinusoidalEmbedding <- nn_module(

  initialize = function(size, scale = 1.0) {

    self$size <- size

    self$scale <- scale
  },

  forward = function(x) {

    x <- x * self$scale

    half_size <- self$size %/% 2

    emb <- torch_log(torch_tensor(10000.0)) / (half_size - 1)

    emb <- torch_exp(-emb * torch_arange(1, half_size))

    emb <- x$unsqueeze(-1) * emb$unsqueeze(1)

    emb <- torch_cat(list(torch_sin(emb), torch_cos(emb)), dim = -1)

    emb
  }
)

#--- LINEAR --------------------------------------------------------------------

#' Linear Positional Embedding
#'
#' The `LinearEmbedding` module implements a simple linear positional embedding,
#' which scales the input and adds a dimension for the embedding.
#'
#' @param size The dimension of the embedding.
#'
#' @param scale A scaling factor applied to the input. Default is 1.0.
#'
#' @return A `LinearEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- LinearEmbedding(8)
#'
#' emb(x)
#'
#' @export

LinearEmbedding <- nn_module(

  initialize = function(size, scale = 1.0) {

    self$size <- size

    self$scale <- scale
  },

  forward = function(x) {

    x <- x / self$size * self$scale

    x$unsqueeze(-1)
  }
)

#--- LEARNABLE -----------------------------------------------------------------

#' Learnable Positional Embedding
#'
#' The `LearnableEmbedding` module provides a learnable linear layer to generate
#' positional embeddings, which allows the model to learn the best embedding
#' values during training.
#'
#' @param size The dimension of the embedding.
#'
#' @return A `LearnableEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- LearnableEmbedding(8)
#'
#' emb(x)
#'
#' @export

LearnableEmbedding <- nn_module(

  initialize = function(size) {

    self$size <- size

    self$linear <- nn_linear(1, size)
  },

  forward = function(x) {

    self$linear(x$unsqueeze(-1) / self$size)
  }
)

#--- IDENTITY ------------------------------------------------------------------

#' Identity Positional Embedding
#'
#' The `IdentityEmbedding` module applies an identity transformation,
#' effectively passing the input through without modification.
#'
#' @return An `IdentityEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- IdentityEmbedding()
#'
#' emb(x)
#'
#' @export

IdentityEmbedding <- nn_module(

  initialize = function() {},

  forward = function(x) {

    x$unsqueeze(-1)
  }
)

#--- ZERO ----------------------------------------------------------------------

#' Zero Positional Embedding
#'
#' The `ZeroEmbedding` module generates a zero-valued embedding of the same
#' size as the input, essentially nullifying the input data.
#'
#' @return A `ZeroEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- ZeroEmbedding()
#'
#' emb(x)
#'
#' @export

ZeroEmbedding <- nn_module(

  initialize = function() {},

  forward = function(x) {

    x$unsqueeze(-1) * 0
  }
)

#--- WRAPPER -------------------------------------------------------------------

#' Positional Embedding Wrapper
#'
#' The `PositionalEmbedding` module is a wrapper that selects the appropriate
#' type of positional embedding based on the specified type.
#'
#' @param size The dimension of the embedding.
#'
#' @param type The type of embedding to use, such as "sinusoidal", "linear",
#' "learnable", "zero", or "identity".
#'
#' @param ... Additional parameters specific to the chosen embedding type.
#'
#' @return A `PositionalEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- PositionalEmbedding(8, "sinusoidal")
#'
#' emb(x)
#'
#' @references <https://github.com/tanelp/tiny-diffusion>
#'
#' @export

PositionalEmbedding <- nn_module(

  initialize = function(size, type, ...) {

    if (type == "sinusoidal") {

      self$layer <- SinusoidalEmbedding(size, ...)

    } else if (type == "linear") {

      self$layer <- LinearEmbedding(size, ...)

    } else if (type == "learnable") {

      self$layer <- LearnableEmbedding(size)

    } else if (type == "zero") {

      self$layer <- ZeroEmbedding()

    } else if (type == "identity") {

      self$layer <- IdentityEmbedding()

    } else {

      stop("Unknown positional embedding type: ", type)
    }
  },

  forward = function(x) {

    self$layer(x)
  }
)

#=== END =======================================================================
