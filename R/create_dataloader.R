#' Create a new dataloader for building a model
#'
#' @param input A matrix with input data (usually your predictors)
#' @param output A matrix with output data (usually what you are predicting)
#' @param aux An (optional) matrix with auxiliary training data (usually for training/loss)
#' @param batch_size The size of the batch you want to load
#' @param shuffle  Whether to randomly shuffle the samples
#'
#' @return The dataloader with the specified batch size as a scorch_dataloader
#'
#' @export
#'
#' @examples
#'
#' input = mtcars |> as.matrix()
#' output = mtcars |> as.matrix()
#' dl = create_dataloader(input,output,batch_size=2)


create_dataloader = function(input,output,aux=NULL,name="dl",batch_size=32,shuffle=TRUE){


  ## Check input arguments
  stopifnot("`input` must be a matrix" = is.matrix(input))
  stopifnot("`output` must be a matrix" = is.matrix(output))

  if(!is.null(aux)){
    stopifnot("`aux` must be a matrix" = is.matrix(aux))
  }

  ## Set up dataset creator function
  create_dataset = torch::dataset(
    name = name,
    initialize = function(input,output,aux){
      self$input = input |> torch::torch_tensor()
      self$output = output |> torch::torch_tensor()
      self$aux = aux
      if(!is.null(aux)){
        self$aux = aux |> torch::torch_tensor()
      }
    },
    .getitem = function(index){
      if(is.null(self$aux)){
        list(input = self$input[index,],output=self$output[index,])
      }else{
        list(input = self$input[index,],output=self$output[index,],aux=self$aux[index,])
      }
    },
    .length = function(){
      self$input$shape[1]
    })

  ## Create dataset
  ds = create_dataset(input,output,aux)

  ## Create the dataloader
  dl = torch::dataloader(ds,batch_size=batch_size,shuffle=shuffle)

  dl = create_scorch_dataloader_class(dl)

  return(dl)

}


## A utility function to create the scorch_dataloader class

create_scorch_dataloader_class = function(dl) {
  tmp = dl
  class(tmp) = c("scorch_dataloader",class(dl))
  return(tmp)
}


#' Create the dataloader head function
#'
#' @param dl This is the data loader you want to see
#'
#' @export
#'
#' @examples
#'
#' input = mtcars |> as.matrix()
#' output = mtcars |> as.matrix()
#' dl = create_dataloader(input,output,batch_size=2)
#' head(dl)


head.scorch_dataloader = function(dl,...){
  cat(crayon::blue("Head of input:\n\n"))
  print(head(dl$.iter()$.next()$input,...))

  cat("\n\n")

  cat(crayon::blue("Head of output:\n\n"))
  print(head(dl$.iter()$.next()$output,...))

  cat("\n\n")

  if(!is.null(dl$.iter()$.next()$aux)){
    cat(crayon::blue("Head of aux:\n\n"))
    print(head(dl$.iter()$.next()$aux,...))
  }
}



#' Create the dataloader print function
#'
#' @param dl This is the data loader you want to see
#'
#' @export
#'
#' @examples
#'
#' input = mtcars |> as.matrix()
#' output = mtcars |> as.matrix()
#' dl = create_dataloader(input,output)
#' dl


print.scorch_dataloader = function(dl){
  cat("This is a dataloader object with features:\n")
  cat(paste0(" * Batch size: ",
             crayon::red(dl$batch_size)))
  cat("\n")
  cat(paste0(" * Number of batches: ",
             crayon::red(dl$.length())))

  cat("\n")
  cat(paste0(" * Dimension of input vectors: ",
             crayon::red(dl$.iter()$.next()$input$shape[2])))

  cat("\n")
  cat(paste0(" * Dimension of output vectors: ",
             crayon::red(dl$.iter()$.next()$output$shape[2])))

  if(!is.null(dl$.iter()$.next()$aux)){
    cat("\n")
    cat(paste0(" * Dimension of aux vectors: ",
             crayon::red(dl$.iter()$.next()$aux$shape[2])))
  }

}

