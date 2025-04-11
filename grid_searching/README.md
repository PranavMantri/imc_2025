# Grid Searching

This folder contains example source code for how you might run grid searching on an algorithm using the `prosperity3bt` backtester that we have added upon

## Installation 

Change directories to `prosperity-3-backtester`. This folder is a git submodule that represents a fork of this [backtester](https://github.com/jmerle/imc-prosperity-3-backtester) provided by an elite IMC participant. If this is your first time using git submodules ensure you do the following

```bash
git submodule init
git submodule update
```

This will effectively clone the repo into the folder and ensure you are on the correct commit. Now all you need to do is install our fork of the backtester into your python env as a CLI tool.

```bash
# If you have already installed the old one get rid of it
pip uninstall prosperity3bt
# Ensure you are in the prosperity-3-backtester directory
pip install -e .
```

To test if the installation worked run `prosperity3bt --help` and in the optional args it should list `--grid-search` and `--param-file`

## File formats

Your algorithm python file now needs to update its `Trader` class to accept a single arguement `'params'` which is a `Dict`. This arguement will be used to set the parameters for your algorithm that you might want to grid search over. Furthermore, you will need `params.json` file that lists out all the parameters you want to grid search over. See an example of both of these in the current folder

## Running

To execute grid search run the backtester in the same way you would a normal backtest, but add the `--grid-search` and `--param-file` optional arguements. The latter expects a file (`'params.json'`) which lists out the parameters you want to search the grid over. For an example that you can run right now do the following.

```bash
# In this current directory...
prosperity3bt example_alg.py 1 --param-file example_params.json --grid-search
```

You should at the end see a pretty print out of the top 5 parameter configurations