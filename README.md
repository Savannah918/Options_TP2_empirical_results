# Options_TP2_empirical_results manual

How to rerun the whole process:

Please change directory in each files to your local directory before running.

1. add_forward_price.ipynb -- it adds another column of forward price to the original .csv datafile and save it as a {year}_with_forward_price.pkl.
2. update_yahoo_SET_price.ipynb -- we need to manually update 12 monthly AM forward prices for each year from the price we extracted from yahoo finance, to match the 'SET' price on CBOE.
3. generate_s3.ipynb -- this construct s3 sets, which contains all matching 4-tuples that satisfies the TP2 assumptions, it then save each date's 4-tuples into a pkl file.
4. transition_analysis_2d_1.py -- this will produce the final average transition matrix across a time period. 



Helper files/functions:
1. Violation_rate_time_series.ipynb -- it plots the violation rates (both with and without tolerance), and number of 4-tuples over time, it also can save these data into csv files.
2. delete_files.ipynb -- it delete files with name match a certain pattern, it move files to another directory.
3. Submitting_job_function_parameter_changer.ipynb -- it change parameters of 10 files(each represent a parameter, 7-1 (for example, means 7 days before the expiration date, with transition length = 1), 7-3, 7-5, 7-7, 5-1, 5-3, 5-5, 3-1, 3-3, 1-1) when submitting a batch of jobs.

