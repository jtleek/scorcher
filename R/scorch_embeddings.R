#===============================================================================
# POSITIONAL EMBEDDINGS FOR DIFFUSION MODELS
#===============================================================================

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
#' @returns A `SinusoidalEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- SinusoidalEmbedding(8)
#'
#' emb(x)
#'
#' @family embeddings
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
#' @returns A `LinearEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- LinearEmbedding(8)
#'
#' emb(x)
#'
#' @family embeddings
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
#' @returns A `LearnableEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- LearnableEmbedding(8)
#'
#' emb(x)
#'
#' @family embeddings
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
#' @returns An `IdentityEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- IdentityEmbedding()
#'
#' emb(x)
#'
#' @family embeddings
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
#' @returns A `ZeroEmbedding` object.
#'
#' @examples
#'
#' x <- torch::torch_tensor(1:10)
#'
#' emb <- ZeroEmbedding()
#'
#' emb(x)
#'
#' @family embeddings
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
#' @returns A `PositionalEmbedding` object.
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
#' @family embeddings
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

      stop("Unknown positional embedding type: ", type, call. = FALSE)
    }
  },

  forward = function(x) {

    self$layer(x)
  }
)

#=== END =======================================================================
