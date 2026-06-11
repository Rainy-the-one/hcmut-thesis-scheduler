# -*- coding: utf-8 -*-
import os
import json
import logging

CONFIG_FILE = "config.json"
user_config = {}

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc config.json: {e}. Sẽ dùng cấu hình mặc định.")

def cfg(section, key, default_val):
    return user_config.get(section, {}).get(key, default_val)

# --- THƯ MỤC & FILE ---
LECTURER_EXCEL_PATH = cfg("files", "lecturer_excel", "KHKTMT.xlsx")
LECTURER_VECTOR_PATH = cfg("files", "lecturer_vector", "lecturer_expertise_vectors.pt")
PROJECT_EXCEL_PATH = cfg("files", "project_excel", "Ket_qua_Sap_xep_Hoi_dong_Processed.xlsx")
ROLE_HISTORY_EXCEL_PATH = cfg("files", "role_history_excel", "Tổng hợp Chủ tịch + Thư Ký.xlsx")
SHEET_OUTPUT_DIR = cfg("files", "output_dir", "nicetry_ket_qua_tung_sheet_1")
STATS_OUTPUT_EXCEL_FILE = cfg("files", "stats_output", "thong_ke_giang_vien_final.xlsx")

# --- RÀNG BUỘC CHUNG ---
COMMITTEE_TOPIC_MIN = cfg("constraints", "committee_topic_min", 3)
COMMITTEE_TOPIC_MAX = cfg("constraints", "committee_topic_max", 7)
MIN_MEMBERS_PER_COMMITTEE = cfg("constraints", "min_members", 3)
MAX_MEMBERS_PER_COMMITTEE = cfg("constraints", "max_members", 5)

# --- THAM SỐ THUẬT TOÁN GA ---
PH2_POP = cfg("ga_params", "population_size", 180)
PH2_GEN = cfg("ga_params", "generations", 240)
PH2_TOURN = cfg("ga_params", "tournament_size", 3)
PH2_CR = cfg("ga_params", "crossover_rate", 0.9)
PH2_MR = cfg("ga_params", "mutation_rate", 0.5)
RANDOM_SEED = cfg("ga_params", "random_seed", 42)

# --- DANH SÁCH CONFLICT HARDCODE ---
HARD_CONFLICTS = [
    ("003634", "003332"),
    ("003634", "003048")
]

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("council_scheduling")