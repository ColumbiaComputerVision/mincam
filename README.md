# Minimalist Vision with Freeform Pixels
### [[Paper]](https://cave.cs.columbia.edu/Statics/publications/pdfs/Klotz_ECCV24.pdf) [[Project Page]](https://cave.cs.columbia.edu/projects/categories/project?cid=Computational+Imaging&pid=Minimalist+Vision+with+Freeform+Pixels) [[Video]](https://youtu.be/KC8s30clJSY)

Code and data for the paper "Minimalist Vision with Freeform Pixels" at ECCV 2024.

[Jeremy Klotz](https://cs.columbia.edu/~jklotz) and [Shree K. Nayar](https://www.cs.columbia.edu/~nayar/)


## Python Environment

Install the conda environment:
```
conda env create -f environment.yml
```

## Data

By default, the dataset and data (model checkpoints and logs) directories are in the root directory. These paths can be changed in constants.py.

Download the data for the toy example:
```
wget "https://cave.cs.columbia.edu/old/projects/mincam_2024/data/toy-example.tar.gz"
```
Create datasets/ in the root directory, and move toy-example/ into datasets/.

## Training

All of the hyperparameters are set in the experiment configuration file (e.g. data/exp_configs/toy-example.yml). 

The first section of the experiment config specifies the hyperparameters used for all models (mincam and baseline models). Each subsequent section specifies the hyperparameters for specific models. 

When the model type is "mincam", the cam_sizes field specifies the number of freeform pixels in the minimalist camera. In this case, img_sizes should be the size of the images in the dataset, which is 128x128 for the toy example. When the model type is "baseline", cam_sizes and img_sizes should be identicial. 

The results in figure 4 of the main paper (the toy example) are generated using the following hyperparameters:

Each model is trained for 6000 epochs with a batch size of 128.
The minimalist camera uses a realistic sensor gain, read noise, quanitization, and saturation level (exact values in toy-example.yml).
We swept different learning rates and used the following learning rate for each
minimalist camera:

| Freeform Pixels | Learning Rate |
| --------------- | ------------- |
| 1 | 5e-3 |
| 2 | 5e-4 |
| 4 | 5e-4 |
| 8 | 5e-4 |
| 16 | 1e-3 |
| 32 | 1e-3 |
| 64 | 1e-3 |
| 128 | 1e-3 |

A single experiment config file can represent dozens of different models, depending on how many hyperparameters are being swept. Since DEBUG is True in batch_train.py, it only trains a single model on 1 gpu. When DEBUG is False, it trains multiple models using all available gpu's (1 gpu per model).

Use batch_train.py to train all the models in toy-example.yml:
```
python batch_train.py -f toy-example.yml
```

Or you can train a single model using command-line arguments (using default values for various other hyperparameters):
```
python main.py -n 4 --img_r 128 --img_c 128 -e 10 -b 128 --gpu_dataset -d cuda:0 --num_workers 0 --lr 5e-4 --base_exp_name toy_example --model_type mincam
```

Start tensorboard to view the loss curves and masks during training:
```
tensorboard --logdir data/logs
```


## Testing

Once trained, test the model using `main.eval_model`. `test.py` shows a minimal example: it tests the counting performance of a minimalist camera with 4 freeform pixels after 10 epochs of training.

## Citation
```
@InProceedings{klotz2024minimalistvision,
    author="Klotz, Jeremy and Nayar, Shree K.",
    title="Minimalist Vision with Freeform Pixels",
    booktitle="European Conference on Computer Vision (ECCV)",
    year="2024",
}
```