scorch_flatten = function(...){
  l = list(...)
  f = function(x){
    torch_flatten(x,l)
  }
  return(f)
}
