# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple
from collections import defaultdict, Counter

from utils import strip_titles_and_normalize, Topic
from config import logger

try:
    from input_processor import InputLoader
except ImportError:
    class InputLoader:
        def __init__(self, path): self.path = path
        def process_excel(self): 
            return []

def load_lecturers(excel_path: str, vector_pt_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(excel_path) if excel_path.endswith('.csv') else pd.read_excel(excel_path, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]
    col_name = next((c for c in df.columns if 'tên' in c.lower()), df.columns[1])
    df['Full name'] = df['Họ và Tên'] if 'Họ và Tên' in df.columns else df[col_name]

    if 'Degree' in df.columns:
        df = df[df['Degree'].notna() & (df['Degree'].astype(str).str.strip() != '')]
    
    df['only_CE'] = df['CE'].astype(str).str.lower().str.strip() == 'x' if 'CE' in df.columns else False
    df['MSCB'] = df['Email'] if 'Email' in df.columns else df.index.astype(str)
    df['knows_english'] = df['English'].astype(str).str.lower().str.strip() == 'x' if 'English' in df.columns else False
    df['is_TS'] = df['Degree'].apply(lambda val: any(k in str(val).lower() for k in ['ts', 'tiến sĩ', 'phd', 'pgs', 'gs', 'prof', 'assoc', 'doctor'])) if 'Degree' in df.columns else False 
    df['Full_name_norm'] = df['Full name'].apply(strip_titles_and_normalize)
    df = df.reset_index(drop=True)

    names = df['Full name'].tolist()
    try: loaded = torch.load(vector_pt_path)
    except Exception: loaded = {}

    vecs = []
    for name in names:
        v = loaded.get(name)
        if v is None:
            v = next((loaded[k] for k in loaded if strip_titles_and_normalize(k) == strip_titles_and_normalize(name)), None)
            
        if v is not None:
             arr = v.cpu().numpy().reshape(-1) if isinstance(v, torch.Tensor) else np.asarray(v).reshape(-1)
             vecs.append(np.resize(arr, 768) if len(arr) != 768 else arr)
        else: vecs.append(np.zeros(768))
            
    lect_np = np.stack(vecs) if vecs else np.zeros((len(names), 768))
    return df, lect_np / (np.linalg.norm(lect_np, axis=1, keepdims=True) + 1e-12)

def load_role_history(file_path: str, lecturer_df: pd.DataFrame) -> Tuple[set, set]:
    historical_chairs = set()
    historical_secretaries = set()
    try:
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path, engine='openpyxl')
        name_to_mscb = {r['Full_name_norm']: str(r['MSCB']) for _, r in lecturer_df.iterrows()}
        
        if 'Chủ tịch' in df.columns:
            for name in df['Chủ tịch'].dropna():
                norm_name = strip_titles_and_normalize(name)
                if not norm_name: continue
                matched_mscb = name_to_mscb.get(norm_name)
                if not matched_mscb:
                    for k, v in name_to_mscb.items():
                        if norm_name in k or k in norm_name:
                            matched_mscb = v
                            break
                if matched_mscb:
                    historical_chairs.add(matched_mscb)
                    
        if 'Thư ký' in df.columns:
            for name in df['Thư ký'].dropna():
                norm_name = strip_titles_and_normalize(name)
                if not norm_name: continue
                matched_mscb = name_to_mscb.get(norm_name)
                if not matched_mscb:
                    for k, v in name_to_mscb.items():
                        if norm_name in k or k in norm_name:
                            matched_mscb = v
                            break
                if matched_mscb:
                    historical_secretaries.add(matched_mscb)
                    
        logger.info(f"Đã load {len(historical_chairs)} người từng làm Chủ tịch và {len(historical_secretaries)} Thư ký từ lịch sử.")
    except Exception as e:
        logger.warning(f"Không thể đọc file Lịch sử Role (sẽ bỏ qua tính năng thưởng role): {e}")
        
    return historical_chairs, historical_secretaries

def load_topics(project_path: str) -> List[Topic]:
    loader = InputLoader(project_path)
    list_of_sheets = loader.process_excel()
    try: sheet_names = pd.ExcelFile(project_path).sheet_names
    except Exception: sheet_names = [f"Sheet_{i}" for i in range(len(list_of_sheets))]
    return [Topic(idx, p, sheet_names[s_idx] if s_idx < len(sheet_names) else f"Sheet_{s_idx}") 
            for s_idx, sheet_data in enumerate(list_of_sheets) for idx, p in enumerate(sheet_data)]

def resolve_mapping(topics: List[Topic], lecturer_df: pd.DataFrame, attr_name: str, target_attr: str):
    mapping = defaultdict(set)
    for _, r in lecturer_df.iterrows(): mapping[r['Full_name_norm']].add(str(r['MSCB']))
    for t in topics:
        raw_list = getattr(t, attr_name) if isinstance(getattr(t, attr_name), list) else [getattr(t, attr_name)]
        found_mscb = set()
        for n in filter(None, raw_list):
            if n in mapping: found_mscb |= mapping[n]
            else:
                for k, m in mapping.items():
                    if n in k or k in n: found_mscb |= m
        setattr(t, target_attr, found_mscb)

def sanitize_sheet_name(name: str) -> str: 
    return "".join(c for c in name if c not in r'[]:*?/\\')[:28] + ("..." if len(name) > 31 else "")

def export_all_sheets_to_single_excel(sheet_to_councils, assignment, lecturer_df, output_file):
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, councils in sheet_to_councils.items():
            rows = []
            for local_id, topics in councils.items():
                mems = [int(x) for x in assignment.get(local_id, []) if x >= 0]
                names = [str(lecturer_df.iloc[m]['Full name']) for m in mems if m < len(lecturer_df)]
                for i, t in enumerate(topics):
                    rows.append({
                        'Hội đồng': f'HĐ {local_id + 1}', 'Số đề tài': len(topics), 'Mã đề tài': t.code,
                        'Tên đề tài (VI)': t.title_vi, 'Tên đề tài (EN)': t.title_en,
                        'GVHD': ', '.join(t.raw_gvhd_list) if t.raw_gvhd_list else '',
                        'Chủ tịch': names[0] if i == 0 and len(names) > 0 else "",
                        'Thư ký': names[1] if i == 0 and len(names) > 1 else "",
                        'Ủy viên 1': names[2] if i == 0 and len(names) > 2 else "",
                        'Ủy viên 2': names[3] if i == 0 and len(names) > 3 else "",
                        'Ủy viên 3': names[4] if i == 0 and len(names) > 4 else "",
                    })
                rows.append({col: "" for col in rows[0].keys() if rows})
            if rows: pd.DataFrame(rows[:-1]).to_excel(writer, sheet_name=sanitize_sheet_name(sheet_name), index=False)

def export_stats(workload: Counter, lecturer_df: pd.DataFrame, filepath: str, english_count: Counter, vietnamese_count: Counter):
    pd.DataFrame([{
        "MSCB": str(row['MSCB']), "Họ tên": row['Full name'],
        "Tổng số hội đồng": workload.get(str(row['MSCB']), 0),
        "HĐ Tiếng Anh": english_count.get(str(row['MSCB']), 0), "HĐ Tiếng Việt": vietnamese_count.get(str(row['MSCB']), 0),
    } for _, row in lecturer_df.iterrows()]).sort_values(by="Tổng số hội đồng", ascending=False).to_excel(filepath, index=False)