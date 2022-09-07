North Atlantic Bloom
==============================

Investigating the role of vertical nitrate flux on phytoplankton bloom.

<p align="center">
  <img src="https://github.com/iuryt/NorthAtlanticBloom/blob/main/Unstable_submesoscale_fronts.gif" /></br>
</p>


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
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── src                <- Source code for use in this project.
        │
        ├── 00-preprocess
        │   │                 
        │   └── 01-preprocess_external.py           <- preprocess external data
        │
        ├── 01-model
        │   │                 
        │   ├── 01-initial_conditions.py            <- initial conditions for the fine-scale submesoscale simulation
        │   ├── 02-submesoscale.jl                  <- fine-scale submesoscale simulation
        │   ├── 03-initial_conditions_coarse.py     <- initial conditions for the coarse-resolution simulations
        │   ├── 04-coarse_averaging.jl              <- coarse-resolution simulation that prescribes restratification (NVF)
        │   ├── 05-coarse_mle.jl                    <- coarse-resolution simulation with MLE parameterization (MLE)
        │   ├── cumulative_vertical_integration.jl  <- function used for initial geostrophic velocities
        │   ├── mle_parameterization.jl             <- function for computing the eddy stream function from Fox-Kemper et al. (2011)
        │   ├── Manifest.toml                       <- manifest file with for the Julia packages
        │   └── Project.toml                        <- project file with for the Julia packages
        │   
        └── 02-analysis
            ├── 01-video.py                         <- code for generating frames for the video
            ├── 02-3D_plots.py                      <- 3D box plots for w and Ro
            ├── 03-Ro_restratification.py           <- comparison for the evolution of Ro and restratification
            ├── 04-biogeochemistry.py               <- comparing new production between simulations
            ├── 05-nitrate_flux.py                  <- comparing nitrate flux between simulations
            └── cmaps.py                            <- extra colormaps     



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
