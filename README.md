North Atlantic Bloom
==============================

Investigating the role of vertical nitrate flux on phytoplankton bloom.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for vizualization.
    │   └── raw            <- The original, immutable data dump (model outputs).
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- submesoscale and coarse-resolution Oceananigans simulations
    │   │
    │   ├── 01-submesoscale.jl  <- fine-scale submesoscale simulation
    │   ├── 02-coarse_no_mle.jl <- coarse-resolution simulation with no MLE parameterization
    │   └── 03-coarse_mle.jl    <- coarse-resolution simulation with MLE parameterization
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    │
    └── src                <- Source code for use in this project.
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── 01-initial_conditions.py            <- fine-scale submesoscale simulation
        │   ├── 02-initial_conditions_coarse.py     <- coarse-resolution simulation with no MLE parameterization
        │   ├── 03-analysis.py                      <- analyze the model output and generate processed data        
        │   └── cumulative_vertical_integration.jl  <- function used for initial geostrophic velocities
        │
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
