North Atlantic Bloom
==============================

Investigating the role of vertical nitrate flux on phytoplankton bloom.

Running order (add later to a `MakeFile`):
   - `src/01-models/01-initial_conditions.py`
   - `src/01-models/02-submesoscale.py`
   - `src/01-models/03-initial_conditions_coarse.py`
   - `src/01-models/04-coarse_no_mle.py`
   - `src/01-models/05-coarse_mle.py`

Project Organization
------------

    ├── LICENSE
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
        ├── 01-model
        │   │                 
        │   ├── 01-initial_conditions.py            <- initial conditions for the fine-scale submesoscale simulation
        │   ├── 02-submesoscale.jl                  <- fine-scale submesoscale simulation
        │   ├── 03-initial_conditions_coarse.py     <- initial conditions for the coarse-resolution simulations
        │   ├── 04-coarse_no_mle.jl                 <- coarse-resolution simulation with no MLE parameterization
        │   ├── 05-coarse_mle.jl                    <- coarse-resolution simulation with MLE parameterization
        │   └── cumulative_vertical_integration.jl  <- function used for initial geostrophic velocities
        │   
        ├── 02-analysis
        │   └── 01-analysis.py                      <- analyze the model output and generate processed data        
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
