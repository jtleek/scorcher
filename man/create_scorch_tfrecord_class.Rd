% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/scorch_tfrecord.R
\name{create_scorch_tfrecord_class}
\alias{create_scorch_tfrecord_class}
\title{Create a Scorch TFRecord Class}
\usage{
create_scorch_tfrecord_class(tfrecord)
}
\arguments{
\item{tfrecord}{A tfrecord object.}
}
\value{
The input tfrecord object with the class attribute set to include
'scorch_tfrecord'.
}
\description{
This function adds the class 'scorch_tfrecord' to a given tfrecord object.
}
\examples{
\dontrun{
# Create basic tfrecord object

tfrecord <- list(

  input = torch::torch_tensor(matrix(1:12, nrow = 3)),

  output = torch::torch_tensor(c(1, 2, 1)),

  n_samples = 3,

  n_features = 4
)

scorch_tfrecord <- create_scorch_tfrecord_class(tfrecord)

class(scorch_tfrecord)
}

}
