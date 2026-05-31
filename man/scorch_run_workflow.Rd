\name{scorch_run_workflow}
\alias{scorch_snapshot}
\alias{scorch_audit}
\alias{scorch_save_run}
\alias{scorch_load_run}
\alias{scorch_compare_runs}
\alias{autoplot.scorch_run}
\title{Reproducible Scorch runs}
\usage{
scorch_snapshot(scorch_model, data = NULL, config = list())
scorch_audit(run)
scorch_save_run(run, path, overwrite = FALSE)
scorch_load_run(path)
scorch_compare_runs(...)
\method{autoplot}{scorch_run}(object, type = c("history", "audit", "timeline"), ...)
}
\arguments{
\item{scorch_model}{A \code{scorch_model} object.}
\item{data}{Optional data object or dataloader.}
\item{...}{Additional arguments.}
\item{run, object}{A \code{scorch_run} object.}
\item{path}{File path.}
\item{overwrite}{Overwrite existing files?}
\item{config}{Optional named list of run configuration.}
\item{type}{Plot type.}
}
\description{
Create, audit, save, load, compare, and plot reproducible Scorch workflow
snapshots. A \code{scorch_run} records graph metadata, runtime metadata,
training history, package versions, and optional data fingerprints.
}
\value{
A \code{scorch_run}, audit tibble, comparison tibble, saved path, loaded run,
or ggplot object.
}
\keyword{models}
