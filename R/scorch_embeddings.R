#===============================================================================
# POSITIONAL EMBEDDINGS FOR DIFFUSION MODELS
#===============================================================================

#=== POSITIONAL EMBEDDINGS =====================================================

#--- SINUSOIDAL ----------------------------------------------------------------

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

IdentityEmbedding <- nn_module(

  initialize = function() {},

  forward = function(x) {

    x$unsqueeze(-1)
  }
)

#--- ZERO ----------------------------------------------------------------------

ZeroEmbedding <- nn_module(

  initialize = function() {},

  forward = function(x) {

    x$unsqueeze(-1) * 0
  }
)

#--- WRAPPER -------------------------------------------------------------------

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
