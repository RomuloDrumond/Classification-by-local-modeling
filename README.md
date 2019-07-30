# Overview

Classic classification by local modeling is a two-step approach for modeling:

1. An unsupervised clustering algorithm is run to find regions in the dataset;
2. For each region, a model is built with the respective data subset.

For inference the procedure is similar:

1. A similarity metric is used to determine the new data point region, e.g. euclidian distance from regions prototypes;
2. The model from that specific region is used to predict the class of the new data point.

To install dependencies run `pip install -r requirements.txt` on the main directory.

## Built with:

* Pandas
* Numpy
* Sklearn
* Plotly

You can see the notebook [here](https://nbviewer.jupyter.org/github/RomuloDrumond/Classification-by-local-modeling/blob/master/Classification%20by%20local%20modeling.ipynb).
