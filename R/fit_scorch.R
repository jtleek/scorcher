


fit_scorch = function(scorch_nnm,
                      loss,
                      optimizer = c("adam","sgd"),
                      num_epochs,
                      dl,
                      lr){

  optim = match.arg(optimizer)
  switch(optim,
        adam = optim_adam(scorch_nnm$parameters,lr=lr),
        sgd =  optim_sgd(scorch_nnm$parameters,lr=lr)
  )

  #loss_fn = loss
  #optimizizer = optim_adam(scorch_nnm$parameters, lr = 0.001)


}
