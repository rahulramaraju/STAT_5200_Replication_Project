## Paper Replication


This project implements a computationally smaller version of Bi and Traum’s
“Estimating Fiscal Limits: The Case of Greece” ([Journal of Applied Econometrics, 2014](https://doi.org/10.1002/jae.2401)).

It uses the paper’s published posterior estimates rather than re-estimating the model’s parameters. The scripts solve the model, filter the economic states, and produce a one-year forecast of interest rates and sovereign default risk.

Differences from the paper
- Parameter estimation: I do not run the paper’s Bayesian MCMC estimation. Instead, I use the published posterior estimates in Table 2.
- State grid: I use 21,875 grid points. The paper’s grid has 577,500 points, so this is a substantially coarser approximation.
- Particle filter: I use 5,000 particles rather than the paper’s 60,000.
- Initial state: The paper uses 2000Q3 and 2000Q4 data to infer the 2000Q4 model state. My implementation initializes dispersed particles using the available 2001 data instead. This is an approximation to the paper’s initialization procedure.
- Interest-rate plot: The model produces rate deviations, so I use 5% as a reference level to display them as rates. This is a plotting assumption, not a parameter estimated by the model; it affects the plotted level.

### Run the pipeline

```bash
python3 observed_data_reader.py

python3 posterior_reader.py --case nonlinear

python3 model_solver.py \
  --case nonlinear \
  --n-b 7 --n-c 5 --n-A 5 --n-g 5 --n-z 5 --n-tau 5

python3 state_filter_smoothed.py --case nonlinear --particles 5000

python3 forecast.py --case nonlinear

