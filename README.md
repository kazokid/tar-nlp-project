# Repository for TAR 2024 project

# Organization:
* `bundle/*` - the data, baselines and scorers given by organizers
* `models/*` - our models codes
* `outputs/*` - our models outputs and scores

# Environment:
It is recommended to use conda environment management tool.
Steps:
 1. Edit the environment variable(s) in the `env.yml` file to match your clone destination
 2. Run `conda env create -f env.yml`
 3. When the dependencies or variables in `env.yml` file change, run `conda env update -f env.yml`