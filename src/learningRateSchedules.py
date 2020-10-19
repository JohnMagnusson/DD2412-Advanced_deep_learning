import flagSettings

def linear_warm_up_lr(epoch):
    warmup_epochs = flagSettings.nr_epochs_warmup
    initial_lr = flagSettings.learning_rate
    lr = initial_lr * ((epoch+1)/warmup_epochs)
    return lr