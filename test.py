import main

# Name of the model's folder in data/models
model_name = "toy-example-128x2-mblur0-lr-5e-04-m4_mincam" 
# Use model checkpoint from this epoch
epoch = 10

test_results = main.eval_model(model_name, epoch)
print("Test RMSE: %.2f" % test_results["test_rmse"])