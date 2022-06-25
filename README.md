# DeepONet: Learning nonlinear operators

[![DOI](https://zenodo.org/badge/260069304.svg)](https://zenodo.org/badge/latestdoi/260069304)

The source code for the paper [L. Lu, P. Jin, G. Pang, Z. Zhang, & G. E. Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3, 218-229, 2021](https://doi.org/10.1038/s42256-021-00302-5).

## System requirements

Most code is written in Python 3, and depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde). Some code is written in Matlab (version R2019a).

## Installation guide

1. Install Python 3
2. Install DeepXDE v0.11.2 (https://github.com/lululxvi/deepxde). If you use DeepXDE>0.11.2, you need to rename `OpNN` to `DeepONet` and `OpDataSet` to `Triple` with other modifications. For DeepONet code using a more recent version of DeepXDE, please see https://github.com/lu-group/deeponet-fno.
3. Optional: For CNN, install Matlab and TensorFlow 1; for Seq2Seq, install PyTorch

The installation may take between 10 minutes and one hour.

## Demo

### Case `Antiderivative`

1. Open deeponet_pde.py, and choose the parameters/setup in the functions `main()` and `ode_system()` based on the comments;
2. Run deeponet_pde.py, which will first generate the two datasets (training and test) and then train a DeepONet. The training and test MSE errors will be displayed in the screen.

A standard output is

```
Building operator neural network...
'build' took 0.104784 s

Generating operator data...
'gen_operator_data' took 20.495655 s

Generating operator data...
'gen_operator_data' took 168.944620 s

Compiling model...
'compile' took 0.265885 s

Initializing variables...
Training model...

Step      Train loss    Test loss     Test metric
0         [1.09e+00]    [1.11e+00]    [1.06e+00]
1000      [2.57e-04]    [2.87e-04]    [2.76e-04]
2000      [8.37e-05]    [9.99e-05]    [9.62e-05]
...
50000     [9.98e-07]    [1.39e-06]    [1.09e-06]

Best model at step 46000:
  train loss: 6.30e-07
  test loss: 9.79e-07
  test metric: [7.01e-07]

'train' took 324.343075 s

Saving loss history to loss.dat ...
Saving training data to train.dat ...
Saving test data to test.dat ...
Restoring model from model/model.ckpt-46000 ...

Predicting...
'predict' took 0.056257 s

Predicting...
'predict' took 0.012670 s

Test MSE: 9.269857471315847e-07
Test MSE w/o outliers: 6.972881784590493e-07
```

You can get the training and test errors in the end of the output.

The run time could be between several minutes to several hours depending on the parameters you choose, e.g., the dataset size and the number of iterations for training.

### Case `Stochastic ODE/PDE`

1. Open sde.py, and choose the parameters/setup in the functions `main()`;
2. Run sde.py, which will generate traning and test datasets;
3. Open deeponet_dataset.py, and choose the parameters/setup in the functions `main()`;
4. Run deeponet_dataset.py to train a DeepONet. The training and test MSE errors will be displayed in the screen.

### Case `1D Caputo fractional derivative`

1. Go to the folder `fractional`;
2. Run Caputo1D.m to generate training and test datasets. One can specify the orthongonal polynomial to be Legendre polynomial or poly-fractonomial in Orthogonal_polynomials.m. Expected run time: 20 mins.
3. Run datasets.py to pack and compress the genrated datasets. Expected outputs: compressed .npz files. Expected run time: 5 mins.
4. Run DeepONet_float32_batch.py to train and test DeepONets. Expected outputs: a figure of training and test losses. Expected run time: 1 hour.

### Case `2D fractional Laplacian`

#### Learning a 2D fractional Laplacian using DeepONets

1. Run Fractional_Lap_2D.m to generate training and test datasets. Expected outputs: text files that store the training and test data. Expected run time: 40 mins.
2. Run datasets.py to pack and compress the genrated datasets. Expected outputs: compressed .npz files. Expected run time: 15 mins.
3. Run DeepONet_float32_batch.py to train and test DeepONets. Expected run time: 3 hours.

#### Learning a 2D fractional Laplacian using CNNs

1. Suppose that the text files containing all training and test sets have been generated in the previous step.
2. Run CNN_operator_alpha.py to train and test CNNs. Expected outputs: a figure of training and test losses. Expected run time: 30 mins.

### Seq2Seq

1. Open seq2seq_main.py, choose the problem in the function main(), and change the parameters/setup in the corresponding function (antiderivative()/pendulum()) if needed.
2. Run seq2seq_main.py, which will first generate the dataset and then train the Seq2Seq model on the dataset. The training and test MSE errors will be displayed in the screen. Moreover, the loss history, generated data and trained best model will be saved in the direction ('./outputs/').

A standard output is

```
Training...
0             Train loss: 0.21926558017730713         Test loss: 0.22550159692764282
1000       Train loss: 0.0022761737927794456     Test loss: 0.0024939212016761303
2000       Train loss: 0.0004760705924127251     Test loss: 0.0005566366016864777
...
49000     Train loss: 1.2885914202342974e-06    Test loss: 1.999963387788739e-06
50000     Train loss: 1.1382834372852813e-06    Test loss: 1.8525416862757993e-06
Done!
'run' took 747.5421471595764 s
Best model at iteration 50000:
Train loss: 1.1382834372852813e-06 Test loss: 1.8525416862757993e-06
```

You can get the training and test errors in the end of the output.

The run time could be between several minutes to several hours depending on the parameters you choose, e.g., the dataset size and the number of iterations for training.

## Instructions for use

The instructions for running each case are as follows.

- Legendre transform: The same as `Antiderivative` in Demo. You need to modify the function `main()` in deeponet_pde.py.
- Antiderivative: In Demo.
- Fractional (1D): In Demo.
- Fractional (2D): In Demo.
- Nonlinear ODE: The same as `Antiderivative` in Demo. You need to modify the functions `main()` and `ode_system()` in deeponet_pde.py.
- Gravity pendulum: The same as `Antiderivative` in Demo. You need to modify the functions `main()` and `ode_system()` in deeponet_pde.py.
- Diffusion-reaction: The same as `Antiderivative` in Demo. You need to modify the function `main()` in deeponet_pde.py.
- Advection: The same as `Antiderivative` in Demo. You need to modify the functions `main()` in deeponet_pde.py, `run()` in deeponet_pde.py, `CVCSystem()` in system.py, and `solve_CVC()` in CVC_solver.py to run each case.
- Advection-diffusion: The same as `Antiderivative` in Demo. You need to modify the function `main()` in deeponet_pde.py.
- Stochastic ODE/PDE: In Demo.

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{lu2021learning,
  title   = {Learning nonlinear operators via {DeepONet} based on the universal approximation theorem of operators},
  author  = {Lu, Lu and Jin, Pengzhan and Pang, Guofei and Zhang, Zhongqiang and Karniadakis, George Em},
  journal = {Nature Machine Intelligence},
  volume  = {3},
  number  = {3},
  pages   = {218--229},
  year    = {2021}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
