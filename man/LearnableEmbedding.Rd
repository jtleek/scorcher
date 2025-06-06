% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/scorch_embeddings.R
\name{LearnableEmbedding}
\alias{LearnableEmbedding}
\title{Learnable Positional Embedding}
\usage{
LearnableEmbedding(size)
}
\arguments{
\item{size}{The dimension of the embedding.}
}
\value{
A \code{LearnableEmbedding} object.
}
\description{
The \code{LearnableEmbedding} module provides a learnable linear layer to generate
positional embeddings, which allows the model to learn the best embedding
values during training.
}
\examples{

x <- torch::torch_tensor(1:10)

emb <- LearnableEmbedding(8)

emb(x)

}
