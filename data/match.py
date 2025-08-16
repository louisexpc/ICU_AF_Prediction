import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Set
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import re
from scipy.stats import chi2_contingency, ttest_ind

LOGS = 'logs'
DATA_CSV = "origin_data_csv"

def catch_target_icd9(df: pd.DataFrame) -> pd.DataFrame:
    """
    以 df 中的 SUBJECT_ID, HADM_ID 到 D_ICD_DIAGNOSES 抓取該次住院的所有 ICD9_CODE，
    並檢查該次住院是否存在心臟衰竭（HF, 428）、急性腎損傷（AKI, 584）、慢性腎臟病（CKD, 585）
    (只檢查主碼 = 前三碼)。

    Args:
    - df: 原始 DataFrame，必須包含 'SUBJECT_ID' 與 'HADM_ID' 欄位。
    
    Return:
    - 傳回原始 df 並新增三個欄位：has_HF, has_acute_K, has_chronic_K（值為 0/1）。
    """

    diagnoses_path = os.path.join(DATA_CSV, "DIAGNOSES_ICD.csv")

    # 讀檔（確保使用正確欄位名稱）
    try:
        diagnose_icd_df = pd.read_csv(
            diagnoses_path,
            usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
            dtype={'SUBJECT_ID': object, 'HADM_ID': object, 'ICD9_CODE': object}
        )
    except Exception as e:
        raise ValueError(f"Can't read {diagnoses_path} : {e}")

    # 解析 ICD9 主碼（前三位數字）函式：處理有 '.'、字元等情況
    def extract_main_icd(code):
        if pd.isna(code):
            return None
        s = str(code).strip()
        # 移除小數點
        s = s.replace('.', '')
        # 嘗試找第一個 3 位數字群（例如 "40301" -> "403"，"428.0" -> "428"）
        m = re.search(r'\d{3}', s)
        if m:
            return m.group(0)
        # 若找不到 3 位數字，退而求前三字元
        return s[:3] if s else None

    diagnose_icd_df['main_icd'] = diagnose_icd_df['ICD9_CODE'].apply(extract_main_icd)

    # 目標主碼
    target_icd9 = {'428': 'has_HF', '584': 'has_acute_K', '585': 'has_chronic_K'}

    # 只保留有用到的主碼，提高效能
    filtered = diagnose_icd_df[diagnose_icd_df['main_icd'].isin(target_icd9.keys())].copy()
    if filtered.empty:
        # 若沒有任何命中，直接在 df 加上三個 0 欄位回傳
        df = df.copy()
        df['has_HF'] = 0
        df['has_acute_K'] = 0
        df['has_chronic_K'] = 0
        return df

    # 以 main_icd 產生對應 flag
    filtered['has_HF'] = (filtered['main_icd'] == '428').astype(int)
    filtered['has_acute_K'] = (filtered['main_icd'] == '584').astype(int)
    filtered['has_chronic_K'] = (filtered['main_icd'] == '585').astype(int)

    # groupby (SUBJECT_ID, HADM_ID)，任一 diagnosis 命中就標為 1 → 使用 max 聚合
    agg = (
        filtered.groupby(['SUBJECT_ID', 'HADM_ID'])[
            ['has_HF', 'has_acute_K', 'has_chronic_K']
        ]
        .max()
        .reset_index()
    )

    # 合併回原 df（左合併，沒命中則為 0）
    df_out = df.copy()
    # 確保 SUBJECT_ID、HADM_ID 欄位型別一致（轉成字串比較保險）
    df_out['SUBJECT_ID'] = pd.to_numeric(df_out['SUBJECT_ID'], errors='coerce').astype('Int64')
    df_out['HADM_ID']    = pd.to_numeric(df_out['HADM_ID'],    errors='coerce').astype('Int64')
    agg['SUBJECT_ID'] = agg['SUBJECT_ID'].astype('Int64')
    agg['HADM_ID']    = agg['HADM_ID'].astype('Int64')

    df_out = df_out.merge(agg, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    # agg.to_csv(os.path.join("test","agg.csv"),index=False)
    # df_out.to_csv(os.path.join("test","test.csv"),index=False)
    # 填補沒命中的為 0，並轉為 int
    for col in ['has_HF', 'has_acute_K', 'has_chronic_K']:
        if col not in df_out.columns:
            df_out[col] = 0
        else:
            na_count = df_out[col].isna().sum()
            if na_count > 0:
                print(f"Column '{col}' has {na_count} missing values. Filling with 0. (Target ICD Code 都不具備)")
            df_out[col] = df_out[col].fillna(0).astype(int)
    # df_out.to_csv(os.path.join("test","df_out.csv"),index=False)
    return df_out




def propensity_match(
    df: pd.DataFrame,
    k: int = 5,
    caliper: Optional[float] = 0.1,
    with_replacement: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    covariates: List[str] = [
        "AGE_YEARS",
        "GENDER",
        "has_HF",
        "has_acute_K",
        "has_chronic_K",
    ]
    if df[covariates].isna().any().any():
        na_cols = df[covariates].columns[df[covariates].isna().any()]
        print(f"有缺失值存在，請檢察:\n{df[na_cols].isna().sum()}")
        return
    # 檢查是否有重複 subject id
    assert df.groupby('SUBJECT_ID').size().max() == 1

    

    df_model = df.copy()
    # Logistic regression → propensity
    X = df_model[covariates]
    y = df_model["HOSPITAL_EXPIRE_FLAG"]
    logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
    logreg.fit(X, y)
    df_model["propensity_score"] = logreg.predict_proba(X)[:, 1]

    # treat: 實驗組(死亡)
    # control: 對照組(存活)
    surv_ctrl = df_model[df_model['HOSPITAL_EXPIRE_FLAG']==0].copy()
    mort_treat = df_model[df_model['HOSPITAL_EXPIRE_FLAG']==1].copy() 

    # 檢查兩邊沒有交集
    assert set(surv_ctrl['SUBJECT_ID']).isdisjoint(mort_treat['SUBJECT_ID'])
    # k‑NN in 1‑D
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(surv_ctrl[["propensity_score"]])
    distances, indices = nn.kneighbors(mort_treat[["propensity_score"]])

    matched_pairs: List[Tuple[int, int]] = []
    used_ctrl: Set[int] = set()

    for i, treat_idx in enumerate(mort_treat.index):
        picked = 0
        for dist, ctrl_loc in zip(distances[i], indices[i]):
            if picked >= k:
                break
            if caliper is not None and dist > caliper:
                break
            ctrl_idx = surv_ctrl.index[ctrl_loc]
            if not with_replacement and ctrl_idx in used_ctrl:
                continue
            matched_pairs.append((treat_idx, ctrl_idx))
            picked += 1
            if not with_replacement:
                used_ctrl.add(ctrl_idx)
    mort_idx  = [t for t, _ in matched_pairs]
    surv_idx = [c for _, c in matched_pairs]

    matched_mort  = mort_treat.loc[mort_idx].drop_duplicates("SUBJECT_ID")
    matched_surv = surv_ctrl.loc[surv_idx].drop_duplicates("SUBJECT_ID")

    assert matched_mort['SUBJECT_ID'].is_unique
    assert matched_surv['SUBJECT_ID'].is_unique

    print(
        f"Matched Mort: {len(matched_mort)}\nMatched Surv: {len(matched_surv)}"
    )
    matched_mort.to_csv(os.path.join(LOGS,"mort_final.csv"),index=False)
    matched_surv.to_csv(os.path.join(LOGS,"surv_final.csv"),index=False)
    return matched_mort, matched_surv



def chi_squared_and_t_test(matched_mort: pd.DataFrame, matched_surv: pd.DataFrame, alpha=0.05):
    """
    檢查 Alive 與 Dead 兩組在指定欄位的基線平衡情況 (GENDER : M :0 ; F:1)
    True = 所有欄位 p 值 > alpha → 基線平衡通過
    False = 至少一欄 p 值 <= alpha → 基線平衡不通過
    
    回傳:
        results: dict, 各欄位檢定結果與百分比
        overall_balance: bool, 整體是否通過基線平衡
    """
    results = {}

    # 二元類別欄位
    binary_columns = ['has_acute_K', 'has_chronic_K', 'has_HF',"GENDER"]
    for col in binary_columns:
        alive_counts = matched_surv[col].value_counts().sort_index()
        dead_counts  = matched_mort[col].value_counts().sort_index()
        table = pd.DataFrame({'Alive': alive_counts, 'Dead': dead_counts}) \
                    .reindex([0, 1], fill_value=0)  # 確保有 0 和 1 類別
        
        chi2, p_val, _, _ = chi2_contingency(table)
        
        # 計算 Alive/Dead 中值為 1 的百分比
        alive_pct_1 = matched_surv[col].value_counts(normalize=True).get(1, 0) * 100
        dead_pct_1  = matched_mort[col].value_counts(normalize=True).get(1, 0) * 100
        
        results[col] = {
            "p_value": float(p_val),
            "balanced": bool(p_val > alpha),
            "alive_pct_1": float(alive_pct_1),
            "dead_pct_1": float(dead_pct_1)
        }

    # 年齡欄位
    alive_age = matched_surv['AGE_YEARS'].dropna()
    dead_age  = matched_mort['AGE_YEARS'].dropna()
    t_stat, p_value = ttest_ind(alive_age, dead_age, equal_var=False)
    
    results['AGE_YEARS'] = {
        "p_value": float(p_value),
        "balanced": bool(p_value > alpha),
        "alive_mean": alive_age.mean(),
        "alive_std": alive_age.std(),
        "dead_mean": dead_age.mean(),
        "dead_std": dead_age.std()
    }

    # 判斷整體基線平衡
    overall_balance = all(v["balanced"] for v in results.values())

    return results, overall_balance

# if __name__=="__main__":
#     mort = pd.read_csv(os.path.join(LOGS,"mort_final.csv"))
#     surv = pd.read_csv(os.path.join(LOGS,"surv_final.csv"))
#     result, overall_balance = chi_squared_and_t_test(mort,surv)
#     print(f"Passed : {overall_balance}")
#     print(result)

