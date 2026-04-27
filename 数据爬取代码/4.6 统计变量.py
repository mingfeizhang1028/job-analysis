from __future__ import annotations

# -*- coding: utf-8 -*-
"""
CFPS 元数据扫描 + 变量处理映射模板生成
版本：Python 3.10+
依赖：pandas, pyreadstat, openpyxl, xlsxwriter
作者：ChatGPT（按用户研究框架定制）

功能：
1．扫描指定目录下所有 .dta 文件；
2．提取文件级、变量级、值标签级元数据；
3．更清晰地识别文件来源与数据角色；
4．汇总变量跨年/跨文件出现情况；
5．生成基于当前处理逻辑的变量映射模板。

说明：
1．默认会读取数据内容，以计算每个变量的非缺失数、唯一值数、最小值、最大值；
2．如果文件较大、运行较慢，可将 COMPUTE_COLUMN_STATS = False；
3．所有输出同时保存为 Excel 和 CSV，便于人工核查与版本管理。
"""


import re
import json
import warnings
from pathlib import Path

import pandas as pd
import pyreadstat

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 0．路径配置：你只需要改这里
# =========================================================

CFPS_RAW = Path(r"O:\日常文件\毕业论文\01.数据\CFPS\Stata")

EXPORT_DIR = r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports"

SCAN_DIRS = [
    ("CFPS_RAW", CFPS_RAW)
]

FILE_PATTERN = "*.dta"

# 若为 True，则读取完整数据并计算列统计量；大文件会慢
COMPUTE_COLUMN_STATS = True

# 若不为 None，则当样本过大时用抽样数据估计 unique_n
UNIQUE_COUNT_SAMPLE_LIMIT = None  # 例如 50000

# =========================================================
# 1．基础工具函数
# =========================================================

def safe_json_dumps(x):
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def infer_year_from_name(file_name: str):
    """
    从文件名中提取年份。
    例如：cfps2022person_202410.dta -> 2022
    """
    m = re.search(r"(20\d{2}|19\d{2})", file_name)
    return int(m.group(1)) if m else None


def classify_top_dir(dir_tag: str, file_path: Path):
    """
    按顶层扫描目录识别数据阶段。
    """
    if dir_tag == "CFPS_RAW":
        return "raw_cfps"
    return "other"


def infer_survey_role(file_name: str, parent_dir: str = ""):
    """
    识别文件在调查体系中的角色。
    """
    s = f"{parent_dir.lower()} {file_name.lower()}"

    role_patterns = [
        ("adult", r"adult"),
        ("person", r"person"),
        ("famconf", r"famconf|famcof"),
        ("famecon", r"famecon"),
        ("childproxy", r"childproxy"),
        ("child", r"child"),
        ("comm", r"comm|community"),
        ("pid", r"\bpid\b"),
        ("crossyearid", r"crossyearid"),
        ("fampidlist", r"fampidlist"),
    ]

    for role, pattern in role_patterns:
        if re.search(pattern, s):
            return role

    return "unknown"


def infer_dataset_class(file_name: str, file_path: Path, dir_tag: str):
    """
    识别数据集类别。
    """
    name = file_name.lower()
    path_str = str(file_path).lower()

    if dir_tag == "CFPS_RAW":
        if re.search(r"cfps(20\d{2}|19\d{2})", name):
            return "cfps_raw_wave"
        if "crossyearid" in name:
            return "cfps_id_link"
        if "fampidlist" in name:
            return "cfps_family_link"
        return "cfps_raw_other"

    return "other"


def infer_path_group(file_path: Path, dir_tag: str):
    """
    按路径分组，便于后续汇总。
    """
    p = str(file_path).replace("\\", "/")

    if dir_tag == "CFPS_RAW":
        if "/孩子库/" in p:
            return "cfps_raw_child_subdir"
        return "cfps_raw_main"
    return "other"


def infer_source_stage(dir_tag: str):
    if dir_tag == "CFPS_RAW":
        return "raw"
    return "other"


def infer_source_note(file_name: str, file_path: Path, dir_tag: str):
    """
    给识别结果补充简短说明。
    """
    dataset_class = infer_dataset_class(file_name, file_path, dir_tag)
    survey_role = infer_survey_role(file_name, file_path.parent.name)

    if dataset_class == "cfps_raw_wave":
        return f"CFPS原始波次文件：{survey_role}"
    if dataset_class == "cfps_processed_panel":
        return "CFPS处理中间/最终面板文件"
    if dataset_class == "cfps_processed_yearfile":
        return f"CFPS处理中间年度文件：{survey_role}"
    if dataset_class.startswith("macro_"):
        return "宏观数据库或政策/地区控制数据"
    if dataset_class == "cfps_id_link":
        return "CFPS跨年个人ID链接文件"
    if dataset_class == "cfps_family_link":
        return "CFPS家庭PID映射文件"
    return "未完全识别，请人工核查"


def try_read_dta_metadata(file_path: Path):
    """
    读取 Stata 文件。
    若 COMPUTE_COLUMN_STATS = False，则只读取 metadata。
    """
    df, meta = pyreadstat.read_dta(
        str(file_path),
        metadataonly=False if COMPUTE_COLUMN_STATS else True,
        apply_value_formats=False
    )
    return df, meta


def summarize_series(s: pd.Series):
    """
    汇总变量统计信息。
    """
    non_missing_n = int(s.notna().sum())
    unique_n = None
    min_val = None
    max_val = None

    if UNIQUE_COUNT_SAMPLE_LIMIT is not None and len(s) > UNIQUE_COUNT_SAMPLE_LIMIT:
        s2 = s.sample(UNIQUE_COUNT_SAMPLE_LIMIT, random_state=42)
    else:
        s2 = s

    try:
        unique_n = int(s2.dropna().nunique())
    except Exception:
        unique_n = None

    if pd.api.types.is_numeric_dtype(s):
        try:
            if non_missing_n > 0:
                min_val = s.min(skipna=True)
                max_val = s.max(skipna=True)
        except Exception:
            min_val = None
            max_val = None
    else:
        min_val = None
        max_val = None

    return non_missing_n, unique_n, min_val, max_val


def detect_special_missing_range(min_val, max_val):
    """
    标记是否可能存在 CFPS 常见特殊缺失编码。
    """
    miss_codes = {-10, -9, -8, -2, -1}
    detected = []
    for c in sorted(miss_codes):
        try:
            if min_val is not None and max_val is not None and min_val <= c <= max_val:
                detected.append(c)
        except Exception:
            pass
    return ",".join(map(str, detected)) if detected else ""


# =========================================================
# 2．扫描 dta 文件
# =========================================================

def scan_all_dta_files(scan_dirs):
    """
    扫描多个目录下的所有 .dta 文件。
    """
    files = []
    for dir_tag, root in scan_dirs:
        if not root.exists():
            print(f"[警告] 目录不存在：{root}")
            continue
        for fp in root.rglob(FILE_PATTERN):
            files.append((dir_tag, fp))
    return files


def build_metadata_tables(file_list):
    """
    构建：
    1．文件级清单
    2．变量级字典
    3．值标签字典
    4．变量跨年/跨文件汇总
    """
    file_inventory_rows = []
    variable_rows = []
    value_label_rows = []

    for dir_tag, fp in file_list:
        print(f"正在扫描：{fp}")

        file_name = fp.name
        year = infer_year_from_name(file_name)
        source_stage = infer_source_stage(dir_tag)
        dataset_class = infer_dataset_class(file_name, fp, dir_tag)
        survey_role = infer_survey_role(file_name, fp.parent.name)
        path_group = infer_path_group(fp, dir_tag)
        source_note = infer_source_note(file_name, fp, dir_tag)

        try:
            df, meta = try_read_dta_metadata(fp)

            n_rows = meta.number_rows if hasattr(meta, "number_rows") else (len(df) if df is not None else None)
            n_cols = meta.number_columns if hasattr(meta, "number_columns") else (df.shape[1] if df is not None else None)
            file_label = getattr(meta, "file_label", None)

            file_inventory_rows.append({
                "dir_tag": dir_tag,
                "source_stage": source_stage,
                "dataset_class": dataset_class,
                "survey_role": survey_role,
                "path_group": path_group,
                "source_note": source_note,
                "file_path": str(fp),
                "file_name": file_name,
                "parent_dir": str(fp.parent),
                "year": year,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "file_label": file_label,
            })

            column_names = list(meta.column_names)
            column_labels = list(meta.column_labels) if meta.column_labels is not None else [None] * len(column_names)
            original_types = getattr(meta, "original_variable_types", {}) or {}
            readstat_types = getattr(meta, "readstat_variable_types", {}) or {}
            variable_measure = getattr(meta, "variable_measure", {}) or {}
            variable_to_label = getattr(meta, "variable_to_label", {}) or {}
            value_labels_map = getattr(meta, "value_labels", {}) or {}

            for idx, var in enumerate(column_names):
                var_label = column_labels[idx] if idx < len(column_labels) else None
                value_label_name = variable_to_label.get(var, None)
                dtype = str(df[var].dtype) if (df is not None and var in df.columns) else None

                non_missing_n, unique_n, min_val, max_val = (None, None, None, None)
                special_missing_candidates = ""

                if COMPUTE_COLUMN_STATS and df is not None and var in df.columns:
                    non_missing_n, unique_n, min_val, max_val = summarize_series(df[var])
                    special_missing_candidates = detect_special_missing_range(min_val, max_val)

                variable_rows.append({
                    "dir_tag": dir_tag,
                    "source_stage": source_stage,
                    "dataset_class": dataset_class,
                    "survey_role": survey_role,
                    "path_group": path_group,
                    "source_note": source_note,
                    "file_name": file_name,
                    "file_path": str(fp),
                    "year": year,
                    "var_name": var,
                    "var_label": var_label,
                    "pandas_dtype": dtype,
                    "original_type": original_types.get(var, None),
                    "readstat_type": readstat_types.get(var, None),
                    "measure": variable_measure.get(var, None),
                    "value_label_name": value_label_name,
                    "non_missing_n": non_missing_n,
                    "unique_n": unique_n,
                    "min": min_val,
                    "max": max_val,
                    "special_missing_candidates": special_missing_candidates,
                })

            for lab_name, lab_map in value_labels_map.items():
                if isinstance(lab_map, dict):
                    for code, lab in lab_map.items():
                        value_label_rows.append({
                            "dir_tag": dir_tag,
                            "source_stage": source_stage,
                            "dataset_class": dataset_class,
                            "survey_role": survey_role,
                            "path_group": path_group,
                            "source_note": source_note,
                            "file_name": file_name,
                            "file_path": str(fp),
                            "year": year,
                            "value_label_name": lab_name,
                            "code": code,
                            "label": lab,
                        })

        except Exception as e:
            print(f"[失败] {fp} -> {e}")
            file_inventory_rows.append({
                "dir_tag": dir_tag,
                "source_stage": infer_source_stage(dir_tag),
                "dataset_class": infer_dataset_class(fp.name, fp, dir_tag),
                "survey_role": infer_survey_role(fp.name, fp.parent.name),
                "path_group": infer_path_group(fp, dir_tag),
                "source_note": infer_source_note(fp.name, fp, dir_tag),
                "file_path": str(fp),
                "file_name": fp.name,
                "parent_dir": str(fp.parent),
                "year": infer_year_from_name(fp.name),
                "n_rows": None,
                "n_cols": None,
                "file_label": f"READ_ERROR: {e}",
            })

    file_inventory = pd.DataFrame(file_inventory_rows)
    variable_dictionary = pd.DataFrame(variable_rows)
    value_labels = pd.DataFrame(value_label_rows)

    if not variable_dictionary.empty:
        tmp = variable_dictionary.copy()

        var_year_summary = (
            tmp.groupby(["var_name", "var_label"], dropna=False)
            .agg(
                years_present=("year", lambda x: ",".join(sorted({str(int(v)) for v in x.dropna().tolist()}))),
                files_present=("file_name", lambda x: " | ".join(sorted(set(x.dropna().tolist())))),
                n_files=("file_name", "nunique"),
                first_year=("year", "min"),
                last_year=("year", "max"),
                source_stages=("source_stage", lambda x: ",".join(sorted(set(map(str, x.dropna().tolist()))))),
                dataset_classes=("dataset_class", lambda x: ",".join(sorted(set(map(str, x.dropna().tolist()))))),
                survey_roles=("survey_role", lambda x: ",".join(sorted(set(map(str, x.dropna().tolist()))))),
                path_groups=("path_group", lambda x: ",".join(sorted(set(map(str, x.dropna().tolist()))))),
                dir_tags=("dir_tag", lambda x: ",".join(sorted(set(map(str, x.dropna().tolist()))))),
            )
            .reset_index()
            .sort_values(["var_name", "first_year"])
        )
    else:
        var_year_summary = pd.DataFrame()

    return file_inventory, variable_dictionary, value_labels, var_year_summary


# =========================================================
# 3．变量处理映射模板
# =========================================================

def build_processing_map_template():
    """
    根据当前研究框架预置变量处理映射骨架。
    """
    rows = [
        {
            "final_var": "Wage",
            "raw_var": "emp_income / p_income / income",
            "source_file_pattern": "cfps20xx person/adult",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "rename_priority",
            "transform_detail": "优先级：emp_income -> Wage；若不存在则 p_income -> Wage；再否则 income -> Wage",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "收入水平效应；基准被解释变量候选",
            "comparability_risk": "中",
            "notes": "不同年份来源口径可能不同，需在论文中说明工资性收入与总收入边界。"
        },
        {
            "final_var": "Wage_m",
            "raw_var": "qg11",
            "source_file_pattern": "cfps20xx person/adult",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "rename",
            "transform_detail": "qg11 -> Wage_m",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "收入稳健性检验",
            "comparability_risk": "中",
            "notes": "多年份可能代表月工资，需核对问卷口径。"
        },
        {
            "final_var": "wage",
            "raw_var": "Wage",
            "source_file_pattern": "processed panel",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "log_transform",
            "transform_detail": "wage = ln(Wage + 1)",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "收入水平效应",
            "comparability_risk": "低",
            "notes": "对名义工资取对数。"
        },
        {
            "final_var": "Wage_r",
            "raw_var": "Wage + defl_cpi_2014",
            "source_file_pattern": "processed panel",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "deflation",
            "transform_detail": "Wage_r = Wage * defl_cpi_2014",
            "stata_program": "main do-file",
            "intermediate_var": "defl_cpi_2014",
            "used_in_analysis": 1,
            "analysis_block": "收入水平效应",
            "comparability_risk": "低",
            "notes": "按 2014 年 CPI 不变价平减。"
        },
        {
            "final_var": "wage_r",
            "raw_var": "Wage_r",
            "source_file_pattern": "processed panel",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "log_transform",
            "transform_detail": "wage_r = ln(Wage_r + 1)",
            "stata_program": "main do-file",
            "intermediate_var": "Wage_r",
            "used_in_analysis": 1,
            "analysis_block": "基准被解释变量",
            "comparability_risk": "低",
            "notes": "建议作为主回归核心因变量。"
        },
        {
            "final_var": "internet",
            "raw_var": "ku2 / mobile_internet / pc_internet",
            "source_file_pattern": "cfps2014adult, cfps2016adult, cfps2018person, cfps2020person, cfps2022person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "direct_or_rowmax",
            "transform_detail": "2014：ku2 -> internet；2016/2018/2020/2022：egen internet = rowmax(mobile_internet, pc_internet)",
            "stata_program": "_make_internet_composite",
            "intermediate_var": "mobile_internet, pc_internet",
            "used_in_analysis": 1,
            "analysis_block": "数字接入（DA）",
            "comparability_risk": "中高",
            "notes": "不同年份题项并非完全同口径，需单列可比性说明。"
        },
        {
            "final_var": "finternet",
            "raw_var": "internet",
            "source_file_pattern": "processed panel",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "family_aggregate_max",
            "transform_detail": "bys fid year: egen finternet = max(internet)",
            "stata_program": "main do-file",
            "intermediate_var": "internet",
            "used_in_analysis": 1,
            "analysis_block": "家庭层数字接入",
            "comparability_risk": "中",
            "notes": "家庭中任一成员接入则家庭接入。"
        },
        {
            "final_var": "first_internet_year",
            "raw_var": "internet + year",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "event_timing",
            "transform_detail": "internet_year = year if internet==1；bys pid: egen first_internet_year=min(internet_year)",
            "stata_program": "main do-file",
            "intermediate_var": "internet_year",
            "used_in_analysis": 1,
            "analysis_block": "事件研究／DID",
            "comparability_risk": "低",
            "notes": "个人首次上网年份。"
        },
        {
            "final_var": "ever_treated",
            "raw_var": "internet",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "event_treatment_indicator",
            "transform_detail": "bys pid: egen ever_treated = max(internet)",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "事件研究／DID",
            "comparability_risk": "低",
            "notes": "个体是否曾经接入互联网。"
        },
        {
            "final_var": "event_time",
            "raw_var": "year + first_internet_year",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "event_time_construct",
            "transform_detail": "event_time = year - first_internet_year if ever_treated==1",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "事件研究",
            "comparability_risk": "低",
            "notes": "相对首次上网时期。"
        },
        {
            "final_var": "p_pre6 / p_pre4 / p_pre2 / p_current / p_post2 / p_post4 / p_post6",
            "raw_var": "event_time",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "event_dummy",
            "transform_detail": "根据 event_time 手动生成事件时点虚拟变量；控制组统一置 0",
            "stata_program": "main do-file",
            "intermediate_var": "event_time, ever_treated",
            "used_in_analysis": 1,
            "analysis_block": "事件研究",
            "comparability_risk": "低",
            "notes": "用于平行趋势与动态效应。"
        },
        {
            "final_var": "post",
            "raw_var": "year + first_internet_year",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "did_post",
            "transform_detail": "post = (year >= first_internet_year) if ever_treated==1；控制组置 0",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "传统 DID",
            "comparability_risk": "低",
            "notes": "传统 post 指示变量。"
        },
        {
            "final_var": "did",
            "raw_var": "internet * post",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "did_interaction",
            "transform_detail": "did = internet * post",
            "stata_program": "main do-file",
            "intermediate_var": "internet, post",
            "used_in_analysis": 1,
            "analysis_block": "传统 DID",
            "comparability_risk": "中",
            "notes": "建议核查定义是否符合你的识别设定；此定义在采用状态变量时可能与 treated×post 含义重叠。"
        },
        {
            "final_var": "game_3",
            "raw_var": "game + game_daily",
            "source_file_pattern": "cfps2020person, cfps2022person",
            "source_years": "2020,2022",
            "transform_type": "frequency_to_3level",
            "transform_detail": "gen game_3 = cond(game==0,1, cond(game_daily==1,3,2))",
            "stata_program": "_freq_to3",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "数字使用：娱乐休闲",
            "comparability_risk": "中",
            "notes": "2020/2022 基于二值使用＋daily；与 2016/2018 替代指标不完全同口径。"
        },
        {
            "final_var": "shop_3",
            "raw_var": "shop + shop_daily / shop_fee",
            "source_file_pattern": "cfps2016adult, cfps2018person, cfps2020person, cfps2022person",
            "source_years": "2016,2018,2020,2022",
            "transform_type": "mixed_construct",
            "transform_detail": "2020/2022：_freq_to3；2016/2018：egen cut(shop_fee), group(3) 后 +1",
            "stata_program": "_freq_to3 / main do-file",
            "intermediate_var": "shop_fee",
            "used_in_analysis": 1,
            "analysis_block": "数字使用：消费交易",
            "comparability_risk": "高",
            "notes": "跨年口径差异显著：频率 vs 金额分组。建议仅作稳健性或分期分析。"
        },
        {
            "final_var": "online_learn_3",
            "raw_var": "online_learn + online_learn_daily / online_learn",
            "source_file_pattern": "cfps2016adult, cfps2018person, cfps2020person, cfps2022person",
            "source_years": "2016,2018,2020,2022",
            "transform_type": "mixed_frequency_to_3level",
            "transform_detail": "2020/2022：_freq_to3；2016/2018：_freq7_to3",
            "stata_program": "_freq_to3 / _freq7_to3",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "数字使用：能力发展",
            "comparability_risk": "中",
            "notes": "不同年份问卷等级不同，但方向相近。"
        },
        {
            "final_var": "social_3",
            "raw_var": "wechat_use + wechat_share_freq / social",
            "source_file_pattern": "cfps2016adult, cfps2018person, cfps2020person, cfps2022person",
            "source_years": "2016,2018,2020,2022",
            "transform_type": "mixed_social_construct",
            "transform_detail": "2020/2022：基于微信使用与朋友圈分享频率构造；2016/2018：由 social 的 7 级频率转 3 级",
            "stata_program": "main do-file / _freq7_to3",
            "intermediate_var": "wechat_use, wechat_share_freq, social",
            "used_in_analysis": 1,
            "analysis_block": "数字使用：社交互动",
            "comparability_risk": "高",
            "notes": "跨年口径异质性较大，建议单独报告。"
        },
        {
            "final_var": "ent_final_3",
            "raw_var": "game_3 + short_video_3 + entertainment_3",
            "source_file_pattern": "processed panel",
            "source_years": "2016-2022",
            "transform_type": "proxy_merge",
            "transform_detail": "ent_proxy3 = rowmax(game_3, short_video_3)；若 entertainment_3 缺失则用 ent_proxy3 回填",
            "stata_program": "main do-file",
            "intermediate_var": "ent_proxy3, entertainment_3",
            "used_in_analysis": 1,
            "analysis_block": "数字使用：娱乐休闲综合代理",
            "comparability_risk": "高",
            "notes": "属于跨题型拼接代理变量，建议放稳健性分析。"
        },
        {
            "final_var": "learn_importance / work_importance / social_importance / entertainment_importance / life_importance",
            "raw_var": "qu954/qu951/qu953/qu952/qu955 或 ku301-305, qu301-305",
            "source_file_pattern": "cfps2014adult, cfps2016adult, cfps2018person, cfps2020person, cfps2022person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "rename_harmonize",
            "transform_detail": "不同年份互联网用途重要性题项统一命名",
            "stata_program": "year-specific rename blocks",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "数字参与价值认知",
            "comparability_risk": "中",
            "notes": "建议后续做因子分析或 PCA 构造 DU 维度。"
        },
        {
            "final_var": "marriage",
            "raw_var": "qea0",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "rename",
            "transform_detail": "qea0 -> marriage",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量",
            "comparability_risk": "低",
            "notes": "婚姻变量标准化。"
        },
        {
            "final_var": "hukou",
            "raw_var": "qa301 / pa301",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "recode_harmonize",
            "transform_detail": "qa301 或 pa301 重编码为农业户口=1、非农户口=0",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；先赋条件",
            "comparability_risk": "中",
            "notes": "不同年份变量名前缀不同。"
        },
        {
            "final_var": "minzu",
            "raw_var": "qa701code / pa701code",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "recode_harmonize",
            "transform_detail": "汉族=1，其他=0",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；先赋条件",
            "comparability_risk": "低",
            "notes": "民族二分变量。"
        },
        {
            "final_var": "health",
            "raw_var": "qp201",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "reverse_recode",
            "transform_detail": "原 1-5 反向编码为 5-1：越大越健康",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量",
            "comparability_risk": "低",
            "notes": "健康自评统一方向。"
        },
        {
            "final_var": "unhealth",
            "raw_var": "health",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "binary_construct",
            "transform_detail": "unhealth = (health < 3)",
            "stata_program": "_make_demographic",
            "intermediate_var": "health",
            "used_in_analysis": 1,
            "analysis_block": "控制变量",
            "comparability_risk": "低",
            "notes": "健康不佳二元变量。"
        },
        {
            "final_var": "job",
            "raw_var": "employ",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "binary_construct",
            "transform_detail": "job = (employ==1)",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量",
            "comparability_risk": "中",
            "notes": "需核实 employ 编码在 2014 年是否完全一致。"
        },
        {
            "final_var": "self_employed",
            "raw_var": "jobclass",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "recode",
            "transform_detail": "jobclass 重新编码：创业=1，其他=0",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；就业结构",
            "comparability_risk": "中",
            "notes": "建议进一步核对 jobclass 原始标签。"
        },
        {
            "final_var": "nonfarm",
            "raw_var": "qg101",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "recode",
            "transform_detail": "农业工作=0，非农工作=1",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；行业属性",
            "comparability_risk": "低",
            "notes": ""
        },
        {
            "final_var": "local_work / remote_work",
            "raw_var": "qg301",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "recode",
            "transform_detail": "工作地是否本县/市",
            "stata_program": "_make_demographic",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；空间流动",
            "comparability_risk": "低",
            "notes": ""
        },
        {
            "final_var": "edu / eduy",
            "raw_var": "cfps20xxedu / cfps20xxeduy_im",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "rename_harmonize",
            "transform_detail": "cfps20xxedu -> edu；cfps20xxeduy_im -> eduy",
            "stata_program": "_rename_common_if_exists",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；教育分层匹配",
            "comparability_risk": "低",
            "notes": ""
        },
        {
            "final_var": "educ",
            "raw_var": "eduy / edu",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "years_of_schooling_fill",
            "transform_detail": "优先用 eduy，若缺失则按 edu 对应学年回填",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；分层；异质性",
            "comparability_risk": "低",
            "notes": "标准受教育年限。"
        },
        {
            "final_var": "edu_stage3",
            "raw_var": "educ",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "categorize",
            "transform_detail": "<9, 9-<15, >=15 三分类",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "教育分层匹配；异质性",
            "comparability_risk": "低",
            "notes": "适合教育分层 PSM-DID。"
        },
        {
            "final_var": "edu_group2",
            "raw_var": "educ",
            "source_file_pattern": "processed panel",
            "source_years": "2014-2022",
            "transform_type": "categorize",
            "transform_detail": "<15 vs >=15 二分类",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "异质性",
            "comparability_risk": "低",
            "notes": "高等教育门槛变量。"
        },
        {
            "final_var": "is_party",
            "raw_var": "party / qn4001 / cfps_party / pn401a",
            "source_file_pattern": "cfps2014-2022 adult/person",
            "source_years": "2014,2016,2018,2020,2022",
            "transform_type": "rowmax_merge",
            "transform_detail": "不同党员题项合并：egen rowmax(...)",
            "stata_program": "year-specific blocks",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "控制变量；社会资本",
            "comparability_risk": "中",
            "notes": "需核实不同题项是否均指党员身份。"
        },
        {
            "final_var": "fid",
            "raw_var": "fid22/fid20/fid18/fid16/fid14/fid12/fid10",
            "source_file_pattern": "processed appended panel",
            "source_years": "2010-2022",
            "transform_type": "coalesce_then_group",
            "transform_detail": "先 coalesce 或逐个填充，再 egen group(fid_raw)",
            "stata_program": "main do-file",
            "intermediate_var": "fid_raw",
            "used_in_analysis": 1,
            "analysis_block": "家庭聚合；家庭固定效应候选",
            "comparability_risk": "中",
            "notes": "跨年家庭编号拼接方式要在附录中说明。"
        },
        {
            "final_var": "region / region1",
            "raw_var": "provcd",
            "source_file_pattern": "processed panel",
            "source_years": "2010-2022",
            "transform_type": "manual_region_classification",
            "transform_detail": "按省份编码划分东中西东北",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "地区异质性",
            "comparability_risk": "低",
            "notes": "代码中 region1 似有笔误，应核查。"
        },
        {
            "final_var": "lngdp_rpc",
            "raw_var": "gdp_pc + gdpdef_2014_base",
            "source_file_pattern": "macro merged panel",
            "source_years": "2014-2022",
            "transform_type": "deflate_and_log",
            "transform_detail": "gdp_rpc0 = gdp_pc*(100/gdpdef_2014_base)*0.0001；lngdp_rpc = ln(gdp_rpc0+1)",
            "stata_program": "main do-file",
            "intermediate_var": "gdp_rpc0",
            "used_in_analysis": 1,
            "analysis_block": "宏观控制变量",
            "comparability_risk": "低",
            "notes": ""
        },
        {
            "final_var": "DID",
            "raw_var": "宽带中国.dta 中城市级试点变量",
            "source_file_pattern": "宽带中国.dta",
            "source_years": "政策覆盖年份",
            "transform_type": "merge_city_policy",
            "transform_detail": "merge m:1 year cityname using 宽带中国.dta；未匹配主样本置 DID=0",
            "stata_program": "merge_broadband_china",
            "intermediate_var": "",
            "used_in_analysis": 1,
            "analysis_block": "外生冲击／工具变量候选",
            "comparability_risk": "中",
            "notes": "依赖 cityname 变量质量。"
        },
        {
            "final_var": "sub_station",
            "raw_var": "sub_station_num",
            "source_file_pattern": "各省开通高铁站数量.dta",
            "source_years": "宏观年份",
            "transform_type": "merge_and_fill",
            "transform_detail": "合并后缺失置 0，sub_station_num -> sub_station",
            "stata_program": "main do-file",
            "intermediate_var": "",
            "used_in_analysis": 0,
            "analysis_block": "工具变量／基础设施控制候选",
            "comparability_risk": "低",
            "notes": ""
        },
    ]

    df = pd.DataFrame(rows)

    extra_cols = [
        "source_dataset_exact",
        "raw_value_label",
        "final_value_label",
        "is_constructed",
        "is_imputed",
        "is_family_aggregated",
        "is_macro_merged",
        "is_deflated",
        "needs_appendix_note",
        "paper_table_usage",
        "check_status",
    ]
    for c in extra_cols:
        if c not in df.columns:
            df[c] = ""

    return df


def build_final_variable_list_template(processing_map_df: pd.DataFrame):
    final_vars = (
        processing_map_df[["final_var", "analysis_block", "used_in_analysis", "comparability_risk", "notes"]]
        .drop_duplicates()
        .sort_values(["used_in_analysis", "final_var"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return final_vars


def build_comparability_risk_template(processing_map_df: pd.DataFrame):
    risk_df = processing_map_df[
        ["final_var", "source_years", "transform_type", "comparability_risk", "notes"]
    ].copy()

    risk_df["risk_reason"] = ""
    risk_df["recommended_action"] = ""
    risk_df["paper_statement"] = ""

    def _recommend(row):
        r = str(row.get("comparability_risk", ""))
        if "高" in r:
            return "建议分年份使用、替代定义稳健性、附录解释口径差异。"
        if "中" in r:
            return "建议在主文中说明统一口径过程，并做替代口径稳健性。"
        return "可直接进入基准模型，但仍建议报告基本描述统计。"

    risk_df["recommended_action"] = risk_df.apply(_recommend, axis=1)
    return risk_df


# =========================================================
# 4．导出 Excel
# =========================================================

def export_metadata_excel(file_inventory, variable_dictionary, value_labels, var_year_summary, export_path: Path):
    with pd.ExcelWriter(export_path, engine="xlsxwriter") as writer:
        file_inventory.to_excel(writer, sheet_name="file_inventory", index=False)
        variable_dictionary.to_excel(writer, sheet_name="variable_dictionary", index=False)
        value_labels.to_excel(writer, sheet_name="value_labels", index=False)
        var_year_summary.to_excel(writer, sheet_name="var_year_summary", index=False)


def export_processing_excel(processing_map, final_var_list, risk_template, export_path: Path):
    with pd.ExcelWriter(export_path, engine="xlsxwriter") as writer:
        processing_map.to_excel(writer, sheet_name="processing_map_template", index=False)
        final_var_list.to_excel(writer, sheet_name="final_variable_list", index=False)
        risk_template.to_excel(writer, sheet_name="comparability_risk_template", index=False)


# =========================================================
# 5．主程序
# =========================================================

def main():
    print("=" * 70)
    print("第 1 步：扫描 .dta 文件")
    print("=" * 70)

    files = scan_all_dta_files(SCAN_DIRS)
    print(f"共发现 {len(files)} 个 .dta 文件。")

    file_inventory, variable_dictionary, value_labels, var_year_summary = build_metadata_tables(files)

    metadata_xlsx = r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\cfps_metadata.xlsx"

    export_metadata_excel(file_inventory, variable_dictionary, value_labels, var_year_summary, metadata_xlsx)
    print(f"[完成] 元数据已导出：{metadata_xlsx}")

    print("=" * 70)
    print("第 2 步：生成变量处理映射模板")
    print("=" * 70)

    processing_map = build_processing_map_template()
    final_var_list = build_final_variable_list_template(processing_map)
    risk_template = build_comparability_risk_template(processing_map)

    processing_xlsx = r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\cfps_processing_map.xlsx"
    export_processing_excel(processing_map, final_var_list, risk_template, processing_xlsx)
    print(f"[完成] 处理映射模板已导出：{processing_xlsx}")

    # 额外导出 CSV
    file_inventory.to_csv(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\file_inventory.csv", index=False, encoding="utf-8-sig")
    variable_dictionary.to_csv(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\variable_dictionary.csv", index=False, encoding="utf-8-sig")
    value_labels.to_csv(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\value_labels.csv", index=False, encoding="utf-8-sig")
    var_year_summary.to_csv(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\var_year_summary.csv", index=False, encoding="utf-8-sig")
    processing_map.to_csv(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\processing_map_template.csv", index=False, encoding="utf-8-sig")
    final_var_list.to_csv(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\final_variable_list.csv", index=False, encoding="utf-8-sig")
    risk_template.to_csv(r"O:\日常文件\毕业论文\01.数据\CFPS\metadata_exports\comparability_risk_template.csv", index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("全部完成。")
    print(f"输出目录：{EXPORT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()