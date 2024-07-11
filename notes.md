# Some notes on the code

- `fit_exact`: assumes you have the true q value and probability transition. This is why 'exact' setting is used in the plot 6 to show what is the best hyperparameters assuming you know the true env.
- `fit_sample`: do not have the above assumption, update theta via interactions with the environment.
- `discounting`: `theory`, `jh`, `practice`
    - `theory`: same as the literature
    - `jh`: the balance between on-policy and off-policy state visitation distribution
    - `practice`: gamma value is set to be 1 (undiscounted)