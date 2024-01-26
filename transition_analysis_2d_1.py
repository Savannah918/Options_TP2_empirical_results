#!/user/xt2276/.conda/envs/optionEnv/bin/python
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta

# this is a transition matrix analysis with 2 dimentional parameters
def transition_mat_2d(start_date,  end_date, m_days_before_T1 = 7, length = 7, dataset_name = '2021_whole_year.csv', vol=50, cp_flag = 'C', am_flag = 1, entry = [1,0]):
    
    # Directory where files are located
    year = dataset_name[:4]
    directory = './'
    t1_rations = []
    total_corrections = []
    transition_matrix_lists = []
    t2_t1_mats = []
    k2_k1_mats = []
    dates = []
    
    
    spx_index_df = pd.read_csv(f'./SP500_data_{year}_yfinance_updated_am_SET.csv', parse_dates=['Date'], index_col='Date')
    with open(os.path.join(directory, dataset_name), 'rb') as f:
        original_csv_df = pickle.load(f)
        
    directory = f'./{year}_Filtering_condition_3/'
    # Loop through dates
    current_date = start_date

    with open(dataset_name, 'rb') as file:
        df = pickle.load(file)
    # Get all distinct values from the "exdate" column
    distinct_exdates = df['exdate'].unique().tolist()
    distinct_exdates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in distinct_exdates]
    
#     check if current_date is an expiry date in the data set
    while current_date <= end_date:
        if current_date.strftime('%Y-%m-%d') not in distinct_exdates:
            current_date += timedelta(days=1)
            continue
#         Initiate matrices
        transition_matrix = np.zeros((2, 3))
        t2_t1_mat = np.zeros((2, 3))
        k2_k1_mat = np.zeros((2, 3))
#         transit_date_str = (current_date + timedelta(days=length)).strftime("%Y-%m-%d")
    
        start_date_temp = current_date - timedelta(days=m_days_before_T1)
        start_date_temp_str = start_date_temp.strftime("%Y-%m-%d")
        
        cur_end_date = start_date_temp + timedelta(days=length)
        end_date_temp_str = cur_end_date.strftime("%Y-%m-%d")
#         print(start_date_temp_str, end_date_temp_str )
        
        file_name_start = f"{start_date_temp_str}FC3_indexed_{cp_flag}_am={str(am_flag)}_vol{str(vol)}_{dataset_name}.pkl"
        full_path_start = os.path.join(directory, file_name_start)
        file_name_end = f"{end_date_temp_str}FC3_indexed_{cp_flag}_am={str(am_flag)}_vol{str(vol)}_{dataset_name}.pkl"
        full_path_end = os.path.join(directory, file_name_end)
        # Check if the file exists and process it, both start and end file need to exist
#         print(os.path.exists(full_path_start), os.path.exists(full_path_end))
        if os.path.exists(full_path_start) and os.path.exists(full_path_end):
            dates.append(start_date_temp_str)
            with open(full_path_start, 'rb') as f:
                df_start = pickle.load(f)
                df_start.index = pd.MultiIndex.from_tuples(df_start.index)

            with open(full_path_end, 'rb') as f:
                df_end = pickle.load(f)
                df_end.index = pd.MultiIndex.from_tuples(df_end.index)
#             for index in tqdm(df_start.index):
            for index in tqdm(df_start.index):
                start_state = df_start['validate'].loc[index]
                matrix_row_index = abs(start_state - 1)
                df_start['Maturity Dates'] = df_start['Maturity Dates'].apply(lambda x: [pd.to_datetime(date) for date in x])
                t2_t1  = df_start.loc[index, 'Maturity Dates'][1] - df_start.loc[index, 'Maturity Dates'][0]
                k2_k1  = df_start.loc[index, 'Strike Prices'][1] / df_start.loc[index, 'Strike Prices'][0]
                t1 = df_start.loc[index, 'Maturity Dates'][0]
                if t1 != current_date:
                    continue
                if am_flag == 0:
                    # Access the data for the specific date
                    index_price = spx_index_df.loc[t1.strftime('%Y-%m-%d')]['Open']
                else:
                    # Access the data for the specific date
                    index_price = spx_index_df.loc[t1.strftime('%Y-%m-%d')]['Close']

                # if end_date == T1, then we do not need to check if index exists in df_end file, it is expiring anyway
                if t1 != cur_end_date and index in df_end.index:
                    end_state = df_end.loc[index, 'validate']
                    if end_state == 1:
                        transition_matrix[matrix_row_index, 0]+=1
                        t2_t1_mat[matrix_row_index, 0] += t2_t1.days
                        k2_k1_mat[matrix_row_index, 0] += k2_k1
                    else: # end state == 0
                        transition_matrix[matrix_row_index, 1]+=1
                        t2_t1_mat[matrix_row_index, 1] += t2_t1.days
                        k2_k1_mat[matrix_row_index, 1] += k2_k1

                elif t1 == cur_end_date:
                    _,s2,_,s4 = index
                    s2_mid = (original_csv_df.loc[(original_csv_df['symbol'] == s2) & (original_csv_df['date'] == t1), 'best_bid'].iloc[0] + original_csv_df.loc[(original_csv_df['symbol'] == s2) & (original_csv_df['date'] == t1), 'best_offer'].iloc[0]) / 2
                    s4_mid = (original_csv_df.loc[(original_csv_df['symbol'] == s4) & (original_csv_df['date'] == t1), 'best_bid'].iloc[0] + original_csv_df.loc[(original_csv_df['symbol'] == s4) & (original_csv_df['date'] == t1), 'best_offer'].iloc[0]) / 2
                    if cp_flag == 'C':
                        end_state = max(0, index_price - df_start.loc[index, 'Strike Prices'][0]/1000) * s2_mid >= s4_mid * max(0, index_price - df_start.loc[index, 'Strike Prices'][1]/1000)
                    else:
                        end_state = max(0, df_start.loc[index, 'Strike Prices'][0]/1000 - index_price) * s2_mid <= s4_mid * max(0, df_start.loc[index, 'Strike Prices'][1]/1000 - index_price)

                    if end_state == True:
                        transition_matrix[matrix_row_index,0]+=1
                        t2_t1_mat[matrix_row_index,0] += t2_t1.days
                        k2_k1_mat[matrix_row_index,0] += k2_k1
                    else:
                        transition_matrix[matrix_row_index,1]+=1
                        t2_t1_mat[matrix_row_index,1] += t2_t1.days
                        k2_k1_mat[matrix_row_index,1] += k2_k1

#                     invalid    
                else:
                    transition_matrix[matrix_row_index,2]+=1
                    t2_t1_mat[matrix_row_index,2] += t2_t1.days
                    k2_k1_mat[matrix_row_index,2] += k2_k1

            # for t2_t1_mat and k2_k1_mat, ensure it's not dividing by zero
            mask = (transition_matrix == 0)
            print(transition_matrix)

            # Normalize each row to sum up to 100%
            row_sums = transition_matrix.sum(axis=1)
            normalized_matrix = np.divide(transition_matrix.T, row_sums, where=row_sums != 0).T

            # Replace NaNs with 0s if there are rows that sum to 0
            normalized_matrix = np.nan_to_num(normalized_matrix)
            print(normalized_matrix)

            t2_t1_mats.append(t2_t1_mat)
            k2_k1_mats.append(k2_k1_mat)
            transition_matrix_lists.append(normalized_matrix)
        # Move to the next day
        current_date += timedelta(days=1)
    array_3d = np.array(transition_matrix_lists)

    # Calculate the average across the matrices
    average_matrix = np.mean(array_3d, axis=0)
    print("average matrix",average_matrix)
    
    # Sum all the matrices in array_3d
    summed_matrix = np.sum(array_3d, axis=0)

    # Normalize the summed matrix
    row_sums = summed_matrix.sum(axis=1)
    normalized_average_matrix = np.divide(summed_matrix.T, row_sums, where=row_sums != 0).T

    # Replace NaNs with 0s if there are rows that sum to 0
    normalized_average_matrix = np.nan_to_num(normalized_average_matrix)
    print("normalized average matrix", normalized_average_matrix)

    
#     plot_graphs(dates, t2_t1_mats, k2_k1_mats, transition_matrix_lists, start_date, end_date, length, m_days_before_T1, dataset_name, vol, cp_flag, am_flag, entry)
#     plot_graphs(dates, t2_t1_mats, k2_k1_mats, transition_matrix_lists, start_date, end_date, length, m_days_before_T1, dataset_name, vol, cp_flag, am_flag, [1,1])
#     plot_graphs(dates, t2_t1_mats, k2_k1_mats, transition_matrix_lists, start_date, end_date, length, m_days_before_T1, dataset_name, vol, cp_flag, am_flag, [1,2])
#     plot_graphs(dates, t2_t1_mats, k2_k1_mats, transition_matrix_lists, start_date, end_date, length, m_days_before_T1, dataset_name, vol, cp_flag, am_flag, [2,0])
#     plot_graphs(dates, t2_t1_mats, k2_k1_mats, transition_matrix_lists, start_date, end_date, length, m_days_before_T1, dataset_name, vol, cp_flag, am_flag, [2,1])
#     print([i / j for i, j in zip(t1_rations,total_corrections)])
#     return t2_t1_mats, k2_k1_mats, transition_matrix_lists
   
def main():
#     transition_mat_2d(start_date=datetime(2022, 10, 1), end_date=datetime(2022, 12, 31), m_days_before_T1 = 7, length = 5, dataset_name = '2022_with_forward_price.pkl', vol=50, cp_flag = 'P', am_flag = 0, entry = [1,0])
    transition_mat_2d(start_date=datetime(2022, 10, 1), end_date=datetime(2022, 12, 31), m_days_before_T1 = 7, length = 1, dataset_name = '2022_with_forward_price.pkl', vol=50, cp_flag = 'P', am_flag = 0, entry = [1,0])

if __name__ == "__main__":

    main()