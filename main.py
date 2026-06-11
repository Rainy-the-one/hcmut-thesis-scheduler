# -*- coding: utf-8 -*-
import os
import time
import math
import numpy as np
from collections import defaultdict, Counter
from typing import List

from config import (
    LECTURER_EXCEL_PATH, LECTURER_VECTOR_PATH, PROJECT_EXCEL_PATH, 
    ROLE_HISTORY_EXCEL_PATH, SHEET_OUTPUT_DIR, STATS_OUTPUT_EXCEL_FILE, 
    COMMITTEE_TOPIC_MAX, logger
)
from utils import Topic
from data_io import (
    load_lecturers, load_role_history, load_topics, 
    resolve_mapping, export_all_sheets_to_single_excel, export_stats
)
from clustering import greedy_cluster_in_sheet
from optimizer import CommitteeNSGA

def council_requires_english(topics: List[Topic]) -> bool:
    return any(any(kw in (" ".join([str(t.he).upper(), str(t.sheet_name).upper(), str(t.chuong_trinh).upper()])) for kw in ["CC", "TIÊN TIẾN", "ADVANCED", "TIẾNG ANH", "CHƯƠNG TRÌNH TIÊN TIẾN"]) for t in topics)

def main():
    start_time = time.time()
    lecturer_df, lecturer_vectors = load_lecturers(LECTURER_EXCEL_PATH, LECTURER_VECTOR_PATH)
    all_topics = load_topics(PROJECT_EXCEL_PATH)
    
    resolve_mapping(all_topics, lecturer_df, 'gvhd_norm', 'supervisors_mscb')
    resolve_mapping(all_topics, lecturer_df, 'gvpb_norm', 'gvpb_mscb')
    
    historical_chairs, historical_secretaries = load_role_history(ROLE_HISTORY_EXCEL_PATH, lecturer_df)

    sheet_groups = defaultdict(list)
    for t in all_topics: sheet_groups[t.sheet_name].append(t)
    
    all_councils, council_to_sheet, global_council_id = {}, {}, 0
    for sheet_name, topics in sheet_groups.items():
        if not topics: continue
        for _, topic_list in greedy_cluster_in_sheet(topics, max_per=COMMITTEE_TOPIC_MAX).items():
            all_councils[global_council_id] = topic_list
            council_to_sheet[global_council_id] = sheet_name; global_council_id += 1
            
    if not all_councils: return logger.error("Không có hội đồng nào!")

    cc_councils, non_cc_councils, cc_ids, non_cc_ids = {}, {}, [], []
    for gid, tps in all_councils.items():
        if council_requires_english(tps): cc_councils[len(cc_ids)] = tps; cc_ids.append(gid)
        else: non_cc_councils[len(non_cc_ids)] = tps; non_cc_ids.append(gid)
            
    num_english_speakers = max(1, lecturer_df['knows_english'].sum())
    p1_avg_slots = (len(cc_councils) * 3) / num_english_speakers
    p1_target_max = math.ceil(p1_avg_slots)

    global_avg_slots = (len(all_councils) * 3) / len(lecturer_df)
    global_target_min = max(1, math.floor(global_avg_slots))
    global_target_max = math.ceil(global_avg_slots)
    
    logger.info(f"Tổng HĐ: {len(all_councils)} (CC: {len(cc_councils)}, Non-CC: {len(non_cc_councils)})")
    logger.info(f"Target Phase 1 (CC): Max {p1_target_max} HĐ/người cho {num_english_speakers} GV TA (Trung bình {p1_avg_slots:.2f})")
    logger.info(f"Target Toàn Cục (Phase 2): [{global_target_min}, {global_target_max}] HĐ/người (Trung bình {global_avg_slots:.2f})")

    global_workload, assignment = Counter(), {}
    
    # --- PHASE 1: CHỈ HỘI ĐỒNG CC ---
    if cc_councils:
        logger.info("\n" + "=" * 50 + "\n BẮT ĐẦU PHASE 1 (HỘI ĐỒNG CC)\n" + "=" * 50)
        nsga_cc = CommitteeNSGA(cc_councils, lecturer_df, lecturer_vectors, Counter(), True, 0, p1_target_max, historical_chairs, historical_secretaries).evolve()
        if nsga_cc:
            for new_id in range(nsga_cc.councils_members.shape[0]):
                assignment[cc_ids[new_id]] = nsga_cc.councils_members[new_id]
                for m in [x for x in nsga_cc.councils_members[new_id] if x >= 0]: 
                    global_workload[str(lecturer_df.iloc[m]['MSCB'])] += 1

    # --- PHASE 2: CÁC HỘI ĐỒNG NON-CC ---
    if non_cc_councils:
        logger.info("\n" + "=" * 50 + "\n BẮT ĐẦU PHASE 2 (HỘI ĐỒNG NON-CC)\n" + "=" * 50)
        nsga_non_cc = CommitteeNSGA(non_cc_councils, lecturer_df, lecturer_vectors, global_workload, False, global_target_min, global_target_max, historical_chairs, historical_secretaries).evolve()
        if nsga_non_cc:
            for new_id in range(nsga_non_cc.councils_members.shape[0]):
                assignment[non_cc_ids[new_id]] = nsga_non_cc.councils_members[new_id]
                for m in [x for x in nsga_non_cc.councils_members[new_id] if x >= 0]: 
                    global_workload[str(lecturer_df.iloc[m]['MSCB'])] += 1

    if not os.path.exists(SHEET_OUTPUT_DIR): os.makedirs(SHEET_OUTPUT_DIR)
    sheet_to_councils = defaultdict(dict)
    for gid, tps in all_councils.items(): sheet_to_councils[council_to_sheet[gid]][gid] = tps
        
    export_all_sheets_to_single_excel(sheet_to_councils, assignment, lecturer_df, os.path.join(SHEET_OUTPUT_DIR, "Ket_qua_phan_cong_tat_ca_sheet.xlsx"))
    
    english_count, viet_count = Counter(), Counter()
    for cid, mems in assignment.items():
        is_eng = council_requires_english(all_councils[cid])
        for m in [int(x) for x in mems if x >= 0]:
            mscb = str(lecturer_df.iloc[m]['MSCB'])
            if is_eng: english_count[mscb] += 1
            else: viet_count[mscb] += 1
            
    export_stats(global_workload, lecturer_df, STATS_OUTPUT_EXCEL_FILE, english_count, viet_count)
    
    if global_workload:
        vals = [global_workload.get(str(m), 0) for m in lecturer_df['MSCB']]
        logger.info(f"• TÓM TẮT: Time {time.time() - start_time:.2f}s | Min {min(vals)} | Max {max(vals)} | TB {np.mean(vals):.2f}")

if __name__ == "__main__":
    main()