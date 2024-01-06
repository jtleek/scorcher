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
#' input = mtcars
#' output = mtcars
#' dl = create_dataloader(input,output)


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

  return(dl)

}

