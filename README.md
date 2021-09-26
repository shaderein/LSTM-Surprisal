## Repo Structure

To run the notebook and the code, build a conda environment using `environment.yml`

`/data` contains all model, vocab, and text files used

- Currently using 3 of the pretrained models [LSTM_40m](https://zenodo.org/record/3559340#.YQxENLqSkTc)
- To run the notebook, you should download them from [here](https://zenodo.org/record/3559340/files/LSTM_40m.tar.gz?download=1) and unzip the models in the folder `/data/LSTM_40m`

`/utils` contains all the functions used in `surprisal.ipynb`

`model.py` contains language model code originally provided [here](https://github.com/vansky/neural-complexity/blob/master/model.py)

`scripts/main.py` is used for debugging only. Check `surprisal.ipynb` instead to see analysis and results