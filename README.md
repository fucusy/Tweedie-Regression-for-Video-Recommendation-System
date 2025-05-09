# Tweedie Regression for Video Recommendation
<img width="993" alt="image" src="https://github.com/user-attachments/assets/03a8e09f-7a0a-4ee7-9365-92b13b14f2c4" />


Paper accepected by ICMI 2025 IEEE 4th International Conference on Computing and Machine Intelligence April 05-06, 2025

Being uploaded to arxiv, will be accessible soon

## Setup
```bash
conda create -n recommender_env python=3.9
conda activate recommender_env
pip install -r requirements.txt
```

## Running one simulation

edit the title_num in agents, and run


```bash
python agent.py
```

You will get output like this
```
Using device:  cpu
title 2: 0.11
title 0: 0.11
title 5: 0.11
title 8: 0.11
title 3: 0.1
title 1: 0.1
title 4: 0.1
title 7: 0.1
title 9: 0.09
title 6: 0.09
reward_when_clicks_maximized:  (np.float64(0.2162792038251148), np.float64(10.986149274377397))
reward_when_watch_duration_maximized:  (np.float64(0.2067680782414789), np.float64(15.269481659881919))
agent name:  TweedieModelAgent
user 0 / 100, time taken: 0.0002758502960205078
Day 0, actual_clicks_reward: 23.0, max_clicks_reward: 21.62792038251146, actual_watch_duration_reward: 896.7794524205063, max_watch_duration_reward: 1526.9481659881894
user 0 / 100, time taken: 0.00010633468627929688
Day 1, actual_clicks_reward: 14.0, max_clicks_reward: 21.62792038251146, actual_watch_duration_reward: 631.2441561329271, max_watch_duration_reward: 1526.9481659881894
user 0 / 100, time taken: 9.608268737792969e-05
Day 2, actual_clicks_reward: 19.0, max_clicks_reward: 21.62792038251146, actual_w
```

setting title number to 1000 might take 2 hours to finish depending on your machine.

## Running multiple simulations

```bash
python run.py
```
will run `agent.py` 10 times using all available cpu cores in parallel. The output will be saved in `output/` folder.
and key data will be saved in logged_date folder.


## Result Visualizing and Analysis

Run the following command to visualize the output of the simulations

```bash
python visualize.py
```
Example output:
![all_agents](./plots/all_agents.png)


Run `python ./sigfinicance_calculator.py` to calculate the significance of the results. 

example output:
```
lift 0.10847954246682129 p value for TweedieModelAgent and RegressionModelAgent: 0.005975222389044577
lift 0.21040501106589973 p value for TweedieModelAgent and WeightedPointWiseModelAgent: 0.0006701759532974361
lift 0.2010559774816251 p value for TweedieModelAgent and PointWiseModelAgent: 1.787095897420876e-05
lift -0.09786336897603888 p value for RegressionModelAgent and TweedieModelAgent: 0.005975222389044577
lift 0.09195069885751117 p value for RegressionModelAgent and WeightedPointWiseModelAgent: 0.06884570297446202
lift 0.08351659319646379 p value for RegressionModelAgent and PointWiseModelAgent: 0.01005134746873515
lift -0.17383025445393202 p value for WeightedPointWiseModelAgent and TweedieModelAgent: 0.0006701759532974361
lift -0.0842077384571644 p value for WeightedPointWiseModelAgent and RegressionModelAgent: 0.06884570297446202
lift -0.007723888697421822 p value for WeightedPointWiseModelAgent and PointWiseModelAgent: 0.8651737653152947
lift -0.16739933962378623 p value for PointWiseModelAgent and TweedieModelAgent: 1.787095897420876e-05
lift -0.0770792009286013 p value for PointWiseModelAgent and RegressionModelAgent: 0.01005134746873515
lift 0.007784011536146464 p value for PointWiseModelAgent and WeightedPointWiseModelAgent: 0.8651737653152947
```
