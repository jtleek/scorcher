


fit_scorch = function(scorch_model,
                      loss=nn_mse_loss,
                      loss_params = list(reduction="mean"),
                      optim = optim_adam,
                      optim_params = list(lr=0.001),
                      num_epochs = 10,
                      verbose=TRUE){

  loss_fn = partial(loss,loss_params)()

  ## Problem seems to be here in getting optimization function right
  optim_fn = partial(optim,scorch_model$nn_model$parameters,optim_params)()

  length_dl = length(scorch_model$dl)

  for (epoch in 1:num_epochs) {  # number of epochs
    total_loss  = 0
    coro::loop(for (batch in scorch_model$dl) {
      optim_fn$zero_grad()
      pred = scorch_model$nn_model(batch$input)
      loss = loss_fn(pred, batch$output)
      loss$backward()
      optim_fn$step()
      total_loss = total_loss + loss$item()
    })
    if(verbose){

      cat(glue::glue("Epoch {crayon::red(epoch)}, Loss: {crayon::red(total_loss/length_dl)} \n\n"))
    }
  }

  return(scorch_model$nn_model)

}




