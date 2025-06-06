% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/scorch_embeddings.R
\name{ZeroEmbedding}
\alias{ZeroEmbedding}
\title{Zero Positional Embedding}
\usage{
ZeroEmbedding()
}
\value{
A \code{ZeroEmbedding} object.
}
\description{
The \code{ZeroEmbedding} module generates a zero-valued embedding of the same
size as the input, essentially nullifying the input data.
}
\examples{

x <- torch::torch_tensor(1:10)

emb <- ZeroEmbedding()

emb(x)

}
