\name{scorch_graph_workflow}
\alias{scorch_block}
\alias{scorch_spec}
\alias{validate_scorch_graph}
\alias{as_torch}
\alias{as_torch_module}
\alias{as_luz_module}
\alias{scorch_graph_data}
\alias{autoplot.scorch_model}
\title{Graph workflow helpers}
\usage{
scorch_block(scorch_model, block = NULL, inputs = NULL, ..., .name = NULL,
  .from = NULL, layers = NULL, residual = FALSE, repeats = 1L)
scorch_spec(scorch_model)
validate_scorch_graph(scorch_model, strict = TRUE)
as_torch(scorch_model, instantiate = FALSE)
as_torch_module(scorch_model)
as_luz_module(scorch_model)
scorch_graph_data(scorch_model)
\method{autoplot}{scorch_model}(object, type = c("architecture", "parameters"), ...)
}
\arguments{
\item{scorch_model, object}{A \code{scorch_model} object.}
\item{block}{Optional advanced escape hatch: a module instance, module
constructor, or function returning a module. If \code{layers} is omitted and
\code{block} is a list, it is treated as \code{layers}.}
\item{inputs}{Optional character vector of upstream nodes.}
\item{...}{In graph block mode, additional unquoted layer calls. Otherwise,
additional arguments.}
\item{.name}{Optional string or unquoted node name.}
\item{.from}{Optional string, unquoted reference, or \code{c(...)} of references.}
\item{layers}{Optional compatibility form for layer specifications. Prefer
unquoted calls in \code{scorch_block()}, for example
\code{scorch_block(linear(in_features = 32, out_features = 32), gelu())}.}
\item{residual}{If \code{TRUE}, add the block input to the block output.}
\item{repeats}{Number of times to repeat \code{layers}.}
\item{strict}{If \code{TRUE}, graph validation errors throw.}
\item{instantiate}{Logical. If \code{FALSE}, return a module generator. If
\code{TRUE}, return an instantiated module.}
\item{type}{Plot type.}
}
\description{
Helpers for adding reusable graph blocks, inspecting and validating graph
specifications, converting Scorch graphs to torch/luz modules, and plotting or
exporting graph data.
}
\value{
Depends on the helper: an updated \code{scorch_model}, a graph specification
tibble, a validation result, a torch module, graph data, or a ggplot object.
}
\keyword{models}
