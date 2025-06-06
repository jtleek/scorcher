% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/scorch_tfrecord.R
\name{head.scorch_tfrecord}
\alias{head.scorch_tfrecord}
\title{Head Method for Scorch TFRecord}
\usage{
\method{head}{scorch_tfrecord}(x, ...)
}
\arguments{
\item{x}{An object of class 'scorch_tfrecord'.}

\item{...}{Additional arguments to be passed to the head function.}
}
\value{
A list containing the first elements of the input and output data
from the tfrecord dataset.
}
\description{
Defines the head method for objects of class 'scorch_tfrecord', returning
the first elements of the input and output data.
}
\examples{
\dontrun{
dataset <- scorch_tfrecord(filepaths, output = labels)

head(dataset)
}
}
