import pandas as pd
import numpy as np
import os
from typing import List, Dict
from match import *
import time
from feature_selection import feature_selection
from match import propensity_match,chi_squared_and_t_test
from train import train
import argparse
import json

LOGS = "./logs"  # 依實際情況修改
TEST = "test"
TRAIN = "train_result"

def _ensure_front_cols(df: pd.DataFrame, front=('SUBJECT_ID','HOSPITAL_EXPIRE_FLAG')):
    # 只取存在的欄位，避免 KeyError
    front = [c for c in front if c in df.columns]
    rest  = [c for c in df.columns if c not in front]
    return df.reindex(columns=front + rest)

def prepare_feature_selection_datasets(df: pd.DataFrame, prefix: str, time_range: List[str]):
    """
    根據指定 df 的 SUBJECT_ID, HOSPITAL_EXPIRE_FLAG 和 time_range，
    將三個時間段的 features 各自組合成各自的 features_dataset。
    會自動刪除全為 NaN 的欄位，並確保 SUBJECT_ID 與 HOSPITAL_EXPIRE_FLAG 在最前面。

    Returns:
        List[pd.DataFrame]: 與 time_range 對應的特徵資料集列表
    """
    hrv_folder = os.path.join(LOGS, f"hrv_{prefix}")
    front_cols = ['SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG']

    # 每個時間段收集一堆 (1 x n_features) 的 DataFrame
    per_time_rows = [[] for _ in time_range]

    for _, row in df.iterrows():
        sid = row['SUBJECT_ID']
        hosp_flag = row['HOSPITAL_EXPIRE_FLAG']

        for idx, time in enumerate(time_range):
            file_path = os.path.join(hrv_folder, f"{sid}_{time}_hrv.csv")
            if not os.path.exists(file_path):
                print(f"== Warning ==: {file_path} not found!")
                continue

            hrv_df = pd.read_csv(file_path)
            # 刪掉全 NaN 欄、全 NaN 列
            hrv_df = hrv_df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

            if hrv_df.empty:
                print(f"== Warning ==: {sid}_{time}_hrv.csv have empty dataframe")
                continue
            if len(hrv_df) < 18:
                print(f"== Warning ==: {sid}_{time}_hrv.csv don't have enough data (only {len(hrv_df)})")

            # 只計數值欄平均（會自動跳過 NaN）
            mean_series = hrv_df.select_dtypes(include='number').mean()

            # 轉成 (1 x n) DataFrame，並把兩個識別欄塞到最前面
            mean_df = mean_series.to_frame().T
            mean_df.insert(0, 'HOSPITAL_EXPIRE_FLAG', hosp_flag)
            mean_df.insert(0, 'SUBJECT_ID', sid)

            per_time_rows[idx].append(mean_df)

    # 對每個時間段把多個 (1 x n) DataFrame 沿列合併
    feature_datasets: List[pd.DataFrame] = []
    for dfs in per_time_rows:
        if dfs:
            out = pd.concat(dfs, ignore_index=True)
        else:
            out = pd.DataFrame(columns=['SUBJECT_ID','HOSPITAL_EXPIRE_FLAG'])
        # 無論如何，在回傳前強制把兩個欄位放前面
        out = _ensure_front_cols(out, ('SUBJECT_ID','HOSPITAL_EXPIRE_FLAG'))
        feature_datasets.append(out)

    for i,feature_df in enumerate(feature_datasets):
        feature_df.to_csv(os.path.join(LOGS,f"{prefix}_{time_range[i]}_features.csv"),index=False)

    return feature_datasets


        


def process(k:int, caliper:float, p_threshold:float, model:str):
    os.makedirs(TRAIN, exist_ok=True)
    mort_hrv = pd.read_csv(os.path.join(LOGS,"mort_stage2_filtered_hrv.csv"))
    surv_hrv = pd.read_csv(os.path.join(LOGS,"surv_stage2_filtered_hrv.csv"))

    # 過濾: 3 個時段都要有

    mort_hrv_filtered = mort_hrv[
        mort_hrv['have_1_3_hrv']&
        mort_hrv['have_4_6_hrv']&
        mort_hrv['have_7_9_hrv']
    ]

    surv_hrv_filtered = surv_hrv[
        surv_hrv['have_1_3_hrv']&
        surv_hrv['have_4_6_hrv']&
        surv_hrv['have_7_9_hrv']
    ]
    print(f"========= After Filtering HRV =========\nSurv : {len(surv_hrv_filtered)}\nMort: {len(mort_hrv_filtered)}")


    print(f"========= Start Match =========")
    combined_mort_surv_df = pd.concat([mort_hrv_filtered,surv_hrv_filtered],axis=0).reset_index(drop=True)
    combined_mort_surv_df_with_match_info = catch_target_icd9(combined_mort_surv_df)

    # k_lists = [1,2,3,4,5,6,7,8,9,10]
    # caliper_lists = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]    
    # max_total = 0
    # total_mort = None
    # total_surv = None

    # for k in k_lists:
    #     for caliper in caliper_lists:
    #         matched_mort, matched_surv = propensity_match(combined_mort_surv_df_with_match_info,k = k,caliper=caliper)
    #         match_test, overallPass = chi_squared_and_t_test(matched_mort,matched_surv,alpha=0.05)
    #         if overallPass == False:
    #             print(f"Match Result isn't balanced for k={k}, caliper={caliper}")
    #             continue
    #         if len(matched_surv)  + len(matched_mort)> max_total:
    #             max_total = len(matched_surv)  + len(matched_mort)
    #             best_k = k
    #             best_caliper = caliper
    #             total_mort = len(matched_mort)
    #             total_surv = len(matched_surv)
    # print(f"Best Match Result: k={best_k}, caliper={best_caliper},  total samples : {max_total} , total mort: {total_mort}, total surv: {total_surv}")
    
   
    matched_mort, matched_surv = propensity_match(combined_mort_surv_df_with_match_info,k = k,caliper=caliper)

    match_test, overallPass = chi_squared_and_t_test(matched_mort,matched_surv,alpha=0.05)

  
    print(f"========= After Matching  =========\nSurv : {len(matched_surv)}\nMort: {len(matched_mort)}\nTest Result: {overallPass}")

     

    match_test_path = os.path.join(TRAIN,f"k_{k}_caliper_{caliper}_pthreshold_{p_threshold}_match_result.json")
    with open(match_test_path, "w", encoding="utf-8") as f:
            json.dump(match_test, f, indent=2)
            
    if overallPass == False:
        print("Match Result isn't balanced")
        
        return



    print(f"========= Start Prepare Feature Selection Dataset =========")
    TIME_RANGE = ['1_3', '4_6','7_9']

    surv_out_lists = prepare_feature_selection_datasets(matched_surv,"surv",TIME_RANGE)
    mort_out_lists = prepare_feature_selection_datasets(matched_mort,"mort",TIME_RANGE)

    assert(len(surv_out_lists) == len(mort_out_lists))

    print(f"========= Start Feature Selection and Traning=========")


    for i in range(len(surv_out_lists)):
        start = time.perf_counter()
        surv_feature_df, mort_feature_df = feature_selection(
            surv_out_lists[i],
            mort_out_lists[i],
            TIME_RANGE[i],
            p_threshold=p_threshold
        )
        summary_fileName = f"k_{k}_caliper_{caliper}_pthreshold_{p_threshold}_{model}"
        train(surv_feature_df,mort_feature_df,TIME_RANGE[i],summary_fileName,model=model)
        end = time.perf_counter()
        print(f"訓練{TIME_RANGE[i]} Model 執行時間: {end - start:.6f} 秒")
        
    return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-caliper", type=float, required=True)
    parser.add_argument("-p_threshold", type=float, required=True)
    parser.add_argument("-m",type=str,required=True)


    args = parser.parse_args()

    process(args.k, args.caliper,args.p_threshold,args.m)

if __name__=="__main__":
    main()

