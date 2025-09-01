import pandas as pd
import lightgbm as lgb
import os
from typing import Tuple
LOGS = "logs"
def feature_selection(
        sur_df:pd.DataFrame,
        mort_df:pd.DataFrame,
        time_range:str,
        p_threshold:float = 0.01
        )->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    利用 LightGBM 做特徵篩選
    Args:
    - sur_df: pd.DataFrame, 除了 HRV Features 以外需要攜帶 HOSPITAL_EXPIRE_FLAG
    - p_threshold : 篩選相對重要性（relative gain）需大於此值
    """

    identifier_cols = [ "SUBJECT_ID", "HOSPITAL_EXPIRE_FLAG"]

    candidate_features = [
            "HRV_MeanNN", "HRV_SDNN", "HRV_SDANN1", "HRV_SDNNI1", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD",
        "HRV_MedianNN", "HRV_MadNN", "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD", "HRV_Prc20NN", "HRV_Prc80NN",
        "HRV_pNN50", "HRV_pNN20", "HRV_MinNN", "HRV_MaxNN", "HRV_HTI", "HRV_TINN", "HRV_VLF", "HRV_LF", "HRV_HF",
        "HRV_VHF", "HRV_TP", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_LnHF", "HRV_SD1", "HRV_SD2", "HRV_SD1SD2",
        "HRV_S", "HRV_CSI", "HRV_CVI", "HRV_CSI_Modified", "HRV_PIP", "HRV_IALS", "HRV_PSS", "HRV_PAS", "HRV_GI",
        "HRV_SI", "HRV_AI", "HRV_PI", "HRV_C1d", "HRV_C1a", "HRV_SD1d", "HRV_SD1a", "HRV_C2d", "HRV_C2a",
        "HRV_SD2d", "HRV_SD2a", "HRV_Cd", "HRV_Ca", "HRV_SDNNd", "HRV_SDNNa", "HRV_DFA_alpha1",
        "HRV_MFDFA_alpha1_Width", "HRV_MFDFA_alpha1_Peak", "HRV_MFDFA_alpha1_Mean", "HRV_MFDFA_alpha1_Max",
        "HRV_MFDFA_alpha1_Delta", "HRV_MFDFA_alpha1_Asymmetry", "HRV_MFDFA_alpha1_Fluctuation",
        "HRV_MFDFA_alpha1_Increment", "HRV_DFA_alpha2", "HRV_MFDFA_alpha2_Width", "HRV_MFDFA_alpha2_Peak",
        "HRV_MFDFA_alpha2_Mean", "HRV_MFDFA_alpha2_Max", "HRV_MFDFA_alpha2_Delta", "HRV_MFDFA_alpha2_Asymmetry",
        "HRV_MFDFA_alpha2_Fluctuation", "HRV_MFDFA_alpha2_Increment", "HRV_ApEn", "HRV_SampEn", "HRV_ShanEn",
        "HRV_FuzzyEn", "HRV_MSEn", "HRV_CMSEn", "HRV_RCMSEn", "HRV_CD", "HRV_HFD", "HRV_KFD", "HRV_LZC"
    ]

    sur_candidates = sur_df[candidate_features+["HOSPITAL_EXPIRE_FLAG"]].apply(pd.to_numeric, errors='coerce')
    mort_candidates = mort_df[candidate_features+["HOSPITAL_EXPIRE_FLAG"]].apply(pd.to_numeric, errors='coerce')

    has_na_surv = sur_candidates.isna().to_numpy().any()
    has_na_mort = mort_candidates.isna().to_numpy().any()
    if has_na_surv or has_na_mort:
        print(f"欄位有缺失值存在，請檢察")
        return
    
    # Prepare Training Dataset
    x_train = pd.concat([sur_candidates.drop(labels=['HOSPITAL_EXPIRE_FLAG'],axis=1),mort_candidates.drop(labels=['HOSPITAL_EXPIRE_FLAG'],axis=1)]).reset_index(drop=True)
    y_train = pd.concat([sur_candidates['HOSPITAL_EXPIRE_FLAG'],mort_candidates['HOSPITAL_EXPIRE_FLAG']]).reset_index(drop=True)

    lgb_train = lgb.Dataset(x_train, label=y_train)

    # === 建立 LightGBM Dataset 並訓練模型 ===
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'verbosity': -1
    }
    model = lgb.train(params, lgb_train, num_boost_round=100)

    
    # === 計算特徵重要性 (gain) 並轉為相對重要性 ===
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    total_gain = importance.sum()

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'gain': importance
    })
    df_importance['relative_gain'] = df_importance['gain'] / total_gain

    # ===  輸出所有特徵的重要性結果 ===
    output_df_importance_path = os.path.join(LOGS,f"fetures_importance_{time_range}.csv")
    df_importance.to_csv(output_df_importance_path, index=False)
    print(f"== Info == 所有特徵之 LightGBM gain 與 relative_gain 已輸出至：{output_df_importance_path}")

    # ===  根據 relative_gain 閾值篩選特徵 ===
    selected_features = df_importance.loc[
        df_importance['relative_gain'] >= p_threshold, 'feature'
    ].tolist()

    print(f"\n使用 relative_gain >= {p_threshold} 篩選通過的特徵：\n{selected_features}")

    if selected_features:
        alive_cols_final = [c for c in (identifier_cols + selected_features) if c in sur_df.columns]
        dead_cols_final  = [c for c in (identifier_cols + selected_features) if c in mort_df.columns]
        sur_df_filtered = sur_df[alive_cols_final]
        mort_df_filtered  = mort_df[dead_cols_final]
    else:
        # 若沒有任何特徵被選到，就只保留識別欄位
        sur_df_filtered = sur_df[identifier_cols]
        mort_df_filtered  = mort_df[identifier_cols]

    # === [9] 輸出最終篩選後 CSV ===

    sur_df_filtered.to_csv(os.path.join(LOGS,f"surv_selected_features_{time_range}.csv"), index=False)
    mort_df_filtered.to_csv(os.path.join(LOGS,f"mort_selected_features_{time_range}.csv"),  index=False)

    return sur_df_filtered,mort_df_filtered
