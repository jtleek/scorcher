#' Create a new dataloader for building a model
#'
#' @param input An input tensor (usually your predictors, first dimensions is index)
#' @param output An output tensor (usually what you are predicting, first dimension is index)
#' @param batch_size The size of the batch you want to load
#' @param shuffle  Whether to randomly shuffle the samples
#'
#' @return The dataloader with the specified batch size as a scorch_dataloader
#'
#' @export
#'
#' @examples
#'
#' input = mtcars |> as.matrix() |> torch_tensor()
#' output = mtcars |> as.matrix() |> torch_tensor()
#' dl = create_dataloader(input,output,batch_size=2)


create_dataloader = function(input,
                             output,
                             name="dl",
                             batch_size=32,
                             shuffle=TRUE){


  ## Check input arguments
  stopifnot("`input` must be a tensor" = ("torch_tensor" %in% class(input)))
  stopifnot("`output` must be a tensor" = ("torch_tensor" %in% class(output)))


  ## Set up dataset creator function
  create_dataset = torch::dataset(
    name = name,
    initialize = function(input,output,aux){
      self$input = input
      self$output = output
    },
    .getitem = function(index){
        list(input = self$input[index],
             output=self$output[index])
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

  val = dl$.iter()$.next()
  return(list(input = head(val$input,...),
         output = head(val$output,...)))

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
  cat(paste0(" * Dimension of input tensors: ",
             crayon::red(calc_dim(dl$.iter()$.next()$input))))

  cat("\n")
  cat(paste0(" * Dimension of output tensors: ",
             crayon::red(calc_dim(dl$.iter()$.next()$output))))


}

calc_dim = function(tensor){
  ndim = length(dim(tensor))
  if(ndim==1){
    return(1)
  }else{
    return(paste0(dim(tensor)[-1],collapse=" "))
  }
}
