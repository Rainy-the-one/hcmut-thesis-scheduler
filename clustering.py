# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from typing import List

from utils import Topic
from config import COMMITTEE_TOPIC_MAX, COMMITTEE_TOPIC_MIN

def greedy_cluster_in_sheet(topics: List[Topic], max_per=COMMITTEE_TOPIC_MAX, min_per=COMMITTEE_TOPIC_MIN):
    # LỚP 1: GOM NHÓM CƠ BẢN THEO CHUYÊN MÔN
    groups = defaultdict(list)
    for t in topics: 
        key = (t.bo_mon.strip().lower(), t.nganh.strip().lower())
        groups[key].append(t)
        
    councils = {}
    cid = 0
    
    for key in sorted(groups.keys()):
        lst = sorted(groups[key], key=lambda x: (len(x.supervisors_mscb) > 0 or len(x.gvpb_mscb) > 0, x.code), reverse=True)
        bins, means = [], []
        
        for t in lst:
            best_idx, best_score = None, -1e9
            for i, b in enumerate(bins):
                if len(b) >= max_per: continue
                
                sim = float(np.dot(t.vector, means[i]))
                bonus = 0.5 if any((t.supervisors_mscb & ex.supervisors_mscb) for ex in b if t.supervisors_mscb) else 0.0
                if sim + bonus > best_score: 
                    best_score = sim + bonus
                    best_idx = i
                    
            if best_idx is not None and best_score > -0.95:
                bins[best_idx].append(t)
                mnew = np.mean(np.vstack([x.vector for x in bins[best_idx]]), axis=0)
                means[best_idx] = mnew / (np.linalg.norm(mnew) + 1e-12)
            else:
                bins.append([t])
                means.append(t.vector.copy())
                
        for b in bins: 
            councils[cid] = b
            cid += 1

    # LỚP 2: GOM ÉP TỐI ĐA (TIẾT KIỆM CHI PHÍ)
    changed = True
    while changed:
        changed = False
        best_pair = None
        best_sim = -1e9
        
        c_ids = list(councils.keys())
        for i in range(len(c_ids)):
            id1 = c_ids[i]
            if id1 not in councils: continue
            for j in range(i + 1, len(c_ids)):
                id2 = c_ids[j]
                if id2 not in councils: continue
                
                if len(councils[id1]) + len(councils[id2]) <= max_per:
                    mean1 = np.mean([t.vector for t in councils[id1]], axis=0)
                    mean2 = np.mean([t.vector for t in councils[id2]], axis=0)
                    sim = float(np.dot(mean1 / (np.linalg.norm(mean1) + 1e-12), mean2 / (np.linalg.norm(mean2) + 1e-12)))
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (id1, id2)
        
        if best_pair is not None:
            id1, id2 = best_pair
            councils[id1].extend(councils[id2])
            del councils[id2]
            changed = True

    # LỚP 3: SAN PHẲNG TẢI (CHỐNG HỘI ĐỒNG LẮT NHẮT < 3)
    changed = True
    while changed:
        changed = False
        small_cids = [k for k, v in councils.items() if 0 < len(v) < min_per]
        
        if not small_cids: break
            
        for sc in small_cids:
            if sc not in councils: continue
            needed = min_per - len(councils[sc])
            
            for _ in range(needed):
                sc_mean = np.mean([t.vector for t in councils[sc]], axis=0)
                sc_mean /= (np.linalg.norm(sc_mean) + 1e-12)
                
                best_t, best_src, best_sim = None, None, -1e9
                
                for rc, r_topics in councils.items():
                    if rc == sc: continue
                    if len(r_topics) > min_per: 
                        for t in r_topics:
                            sim = float(np.dot(t.vector, sc_mean))
                            if sim > best_sim:
                                best_sim, best_t, best_src = sim, t, rc
                                
                if best_t:
                    councils[sc].append(best_t)
                    councils[best_src].remove(best_t)
                    changed = True
            
            if changed: break 
            
    final_councils = {}
    new_idx = 0
    for k in sorted(councils.keys()):
        if len(councils[k]) > 0:
            final_councils[new_idx] = councils[k]
            new_idx += 1
            
    return final_councils