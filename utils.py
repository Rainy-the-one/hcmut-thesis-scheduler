# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd

TITLE_PATTERN = re.compile(r'\b(PGS|PGS\.|GS|GS\.|TS|TS\.|ThS|ThS\.|CN|CN\.|Dr|Dr\.|Prof|Prof\.)\b', flags=re.IGNORECASE)

def strip_titles_and_normalize(name: str) -> str:
    if name is None or pd.isna(name): return ""
    s = TITLE_PATTERN.sub('', str(name))
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^0-9a-zA-ZÀ-Ỷà-ỷ\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()

class Topic:
    def __init__(self, idx: int, p, sheet_name: str):
        self.row_index = idx
        self.raw = p.raw_data if hasattr(p, 'raw_data') else {}
        self.sheet_name = sheet_name
        self.code = self.raw.get("Mã đồ án", "")
        self.title_vi = str(self.raw.get("Tên đồ án (Tiếng Việt)", "")).strip()
        self.title_en = str(self.raw.get("Tên đồ án (Tiếng Anh)", "")).strip()
        self.desc_vi = str(self.raw.get("Mô tả (Tiếng Việt)", "")).strip()
        self.desc_en = str(self.raw.get("Mô tả (Tiếng Anh)", "")).strip()
        self.bo_mon = str(self.raw.get("Bộ môn", "")).strip()
        
        self.nganh = str(self.raw.get("Ngành", "")).strip()
        if not self.nganh and "Ngành_2" in self.raw:
            self.nganh = str(self.raw.get("Ngành_2", "")).strip()
        
        nganh_upper = self.nganh.upper()
        sheet_upper = self.sheet_name.upper()
        self.is_CE = any(kw in nganh_upper for kw in ["KỸ THUẬT MÁY TÍNH", "KTMT", "LIÊN NGÀNH", "CE"]) or \
                     any(kw in sheet_upper for kw in ["CE", "LN", "KTMT"])
        
        self.is_CS = not self.is_CE
        
        self.chuong_trinh = self.raw.get("Chương trình đào tạo", "") or ""
        self.mon_hoc = str(self.raw.get("Môn học", "")).strip() if "Môn học" in self.raw else ""
        
        self.raw_gvhd_list = [part.strip() for part in re.split(r',|;|\n', str(self.raw.get('Tập thể hướng dẫn', ''))) if part.strip()]
        self.gvhd_norm = [strip_titles_and_normalize(x) for x in self.raw_gvhd_list]
        self.supervisors_mscb = set()
        
        self.vector = getattr(p, 'vector', np.zeros(768))
        self.raw_gvpb = self.raw.get('GVPB', '')
        self.gvpb_norm = strip_titles_and_normalize(self.raw_gvpb) if self.raw_gvpb else ''
        self.gvpb_mscb = set()
        self.he = self.raw.get('Hệ', '')