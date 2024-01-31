# Globally Assessing Local Explanations (GALE) with Gudhi

This version of GALE's Python library uses the [GUDHI library](https://gudhi.inria.fr/) to create the mapper and compute the persistence diagram and bottleneck distances. The [original version of GALE](https://github.com/pnxenopoulos/gale) uses GUDHI for computing the persistence diagram and the bottleneck distance, but the mapper is from [kmapper](https://kepler-mapper.scikit-tda.org/en/latest/).

The main advantage of using GUDHI's mapper is the automatic estimation of the distance threshold parameter for the clustering, significantly reducing the execution time for selecting the mapper parameters.

This version also introduces two functions for plotting images relevant to analyzing GALE's results: the mapper (`plot_mapper`) and the persistence diagram (`plot_ext_persistance_diagram`).
