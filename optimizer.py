# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import Counter, defaultdict

from config import (
    MAX_MEMBERS_PER_COMMITTEE, PH2_POP, PH2_GEN, 
    PH2_TOURN, PH2_CR, PH2_MR, RANDOM_SEED, 
    HARD_CONFLICTS, logger
)

class CommitteeIndividual:
    def __init__(self, n_councils, max_k_members):
        self.n_councils = n_councils
        self.max_k_members = max_k_members
        self.councils_members = np.full((n_councils, max_k_members), -1, dtype=int)
        self.objs = np.zeros(2, dtype=float)
        self.rank, self.crowding = 0, 0.0

    def copy(self):
        ni = CommitteeIndividual(self.n_councils, self.max_k_members)
        ni.councils_members = self.councils_members.copy()  
        ni.objs, ni.rank, ni.crowding = self.objs.copy(), self.rank, self.crowding
        return ni

    def get_num_members(self, cid: int) -> int: return int(np.sum(self.councils_members[cid] >= 0))

class CommitteeNSGA:
    def __init__(self, councils, lecturer_df, lecturer_vectors, previous_workload=None, is_phase_1=False, target_min=0, target_max=0, historical_chairs=None, historical_secretaries=None):
        random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
        self.councils, self.council_ids = councils, sorted(councils.keys())
        self.n_councils, self.max_k_members = len(self.council_ids), MAX_MEMBERS_PER_COMMITTEE
        self.lecturer_df, self.lect_vectors = lecturer_df, lecturer_vectors
        self.n_lecturers = lecturer_vectors.shape[0] if lecturer_vectors is not None else 0
        self.lect_mscb, self.lect_is_ts, self.lect_knows_english, self.lect_only_CE = lecturer_df['MSCB'].to_numpy(), lecturer_df['is_TS'].to_numpy(), lecturer_df['knows_english'].to_numpy(), lecturer_df['only_CE'].to_numpy()
        self.previous_workload = previous_workload if previous_workload is not None else Counter()
        self.is_phase_1, self.target_min, self.target_max = is_phase_1, target_min, target_max
        
        self.historical_chairs = historical_chairs if historical_chairs else set()
        self.historical_secretaries = historical_secretaries if historical_secretaries else set()

        self.council_means, self.council_requires_english, self.council_is_ce, self.council_supervisors, self.council_gvpb = np.zeros((self.n_councils, self.lect_vectors.shape[1])), {}, {}, {}, {}

        # --- KHỞI TẠO CONFLICT GRAPH ---
        self.conflict_graph = defaultdict(set)
        def find_lecturer_idx(ident):
            ident = str(ident).strip().lower()
            for i, mscb in enumerate(self.lect_mscb):
                if ident == str(mscb).strip().lower() or ident in str(mscb).strip().lower():
                    return i
            return None

        for m1, m2 in HARD_CONFLICTS:
            idx1 = find_lecturer_idx(m1)
            idx2 = find_lecturer_idx(m2)
            if idx1 is not None and idx2 is not None:
                self.conflict_graph[idx1].add(idx2)
                self.conflict_graph[idx2].add(idx1)
                logger.info(f"Đã kích hoạt khóa Conflict: {self.lect_mscb[idx1]} <-> {self.lect_mscb[idx2]}")
            else:
                logger.warning(f"Không tìm thấy giảng viên cho cặp Conflict: ({m1}, {m2}). Hãy dùng đúng Email/MSCB.")

        for i, cid in enumerate(self.council_ids):
            tlist = self.councils[cid]
            m = np.mean(np.vstack([t.vector for t in tlist]), axis=0)
            self.council_means[i] = m / (np.linalg.norm(m) + 1e-12)
            sample = " ".join(str(t.he).upper() + " " + str(t.sheet_name).upper() + " " + str(t.chuong_trinh).upper() for t in tlist)
            self.council_requires_english[i] = any(kw in sample for kw in ["CC", "TIÊN TIẾN", "ADVANCED", "TIẾNG ANH", "CHƯƠNG TRÌNH TIÊN TIẾN"])
            self.council_is_ce[i] = any(t.is_CE for t in tlist)
            self.council_supervisors[i] = set().union(*(t.supervisors_mscb for t in tlist))
            self.council_gvpb[i] = set().union(*(t.gvpb_mscb for t in tlist))

        self.pop_size, self.n_gen, self.tourn, self.cr, self.mr = PH2_POP, PH2_GEN, PH2_TOURN, PH2_CR, PH2_MR

    def get_candidate_weights(self, available_indices, current_wl: Counter, is_ce_council: bool = False) -> np.ndarray:
        weights = []
        for idx in available_indices:
            load = current_wl.get(str(self.lect_mscb[idx]), 0)
            if load >= self.target_max: w = 0.001       
            elif load == 0: w = 100.0                   
            elif load < self.target_min: w = 10.0       
            else: w = 1.0
            
            if is_ce_council and self.lect_only_CE[idx] and load < self.target_max:
                w *= 5.0
                
            weights.append(w)
        weights = np.array(weights)
        return weights / weights.sum() if weights.sum() > 0 else np.ones(len(available_indices)) / len(available_indices)

    def validate(self, ind: CommitteeIndividual) -> bool:
        for i in range(ind.councils_members.shape[0]):
            valid = [int(x) for x in ind.councils_members[i] if x >= 0]
            if len(valid) not in (3, 5) or len(set(valid)) != len(valid): return False
            chair = valid[0]
            if not self.lect_is_ts[chair] or (self.council_requires_english[i] and not self.lect_knows_english[chair]) or (not self.council_is_ce[i] and any(self.lect_only_CE[idx] for idx in valid)): return False
            if self.council_requires_english[i] and any(not self.lect_knows_english[idx] for idx in valid): return False
            
            if any(idx in self.conflict_graph[other_idx] for idx in valid for other_idx in valid if idx != other_idx):
                return False

        return True

    def random_individual(self):
        ind = CommitteeIndividual(self.n_councils, self.max_k_members)
        running_wl = Counter(self.previous_workload)
        for i in range(self.n_councils):
            base_mask = self.lect_knows_english.copy() if self.council_requires_english[i] else np.ones(self.n_lecturers, dtype=bool)
            if not self.council_is_ce[i]: base_mask &= ~self.lect_only_CE

            chair_candidates = np.where(self.lect_is_ts & base_mask)[0]
            if len(chair_candidates) == 0: chair_candidates = np.where(self.lect_is_ts)[0]
            chair = int(np.random.choice(chair_candidates, p=self.get_candidate_weights(chair_candidates, running_wl, self.council_is_ce[i])))
            
            members, target = [chair], random.choices([3, 5], weights=[0.9, 0.1])[0]
            running_wl[str(self.lect_mscb[chair])] += 1
            available_mask = base_mask.copy()
            available_mask[chair] = False
            
            for conflict_idx in self.conflict_graph[chair]:
                if conflict_idx < self.n_lecturers:
                    available_mask[conflict_idx] = False

            while len(members) < target and available_mask.sum() > 0:
                avail_idx = np.where(available_mask)[0]
                chosen = int(np.random.choice(avail_idx, p=self.get_candidate_weights(avail_idx, running_wl, self.council_is_ce[i])))
                members.append(chosen)
                running_wl[str(self.lect_mscb[chosen])] += 1
                available_mask[chosen] = False
                
                for conflict_idx in self.conflict_graph[chosen]:
                    if conflict_idx < self.n_lecturers:
                        available_mask[conflict_idx] = False

            ind.councils_members[i, :len(members)] = members
        return ind

    def evaluate(self, ind: CommitteeIndividual) -> np.ndarray:
        similarity, penalty, local_wl = 0.0, 0.0, Counter()
        for i in range(ind.councils_members.shape[0]):
            valid = [int(x) for x in ind.councils_members[i] if x >= 0]
            if not valid: penalty += 10000.0; continue
            
            similarity += np.sum(np.dot(self.lect_vectors[valid], self.council_means[i]))
            mscb_list = [str(self.lect_mscb[m]) for m in valid]
            if not any(m in self.council_supervisors[i] for m in mscb_list): penalty += 300.0
            if not any(m in self.council_gvpb[i] for m in mscb_list): penalty += 300.0
            penalty += -1200.0 if len(valid) == 3 else 300.0
            
            chair_mscb = str(self.lect_mscb[valid[0]])
            if chair_mscb in self.historical_chairs:
                penalty -= 500.0 
                
            if len(valid) > 1:
                sec_mscb = str(self.lect_mscb[valid[1]])
                if sec_mscb in self.historical_secretaries:
                    penalty -= 500.0 

            for m in mscb_list: local_wl[m] += 1
            
            chair = valid[0]
            if not self.lect_is_ts[chair] or (self.council_requires_english[i] and any(not self.lect_knows_english[m] for m in valid)) or (not self.council_is_ce[i] and any(self.lect_only_CE[m] for m in valid)): penalty += 50000.0
            
            if any(m1 in self.conflict_graph[m2] for m1 in valid for m2 in valid if m1 != m2):
                penalty += 100000.0

        total_wl = Counter(self.previous_workload)
        for m, count in local_wl.items(): total_wl[m] += count
        values = [total_wl.get(str(m), 0) for m in self.lect_mscb]
        
        overload_pen = 0.0
        for cnt in values:
            if cnt > self.target_max: 
                overload_pen += 2000.0 * ((cnt - self.target_max) ** 2)
            elif cnt < self.target_min:
                if not self.is_phase_1:
                    overload_pen += 5000.0 if cnt == 0 else 1000.0 * ((self.target_min - cnt) ** 2)
            else:
                if not self.is_phase_1 and cnt > 0: 
                    overload_pen -= 800.0 

        penalty += overload_pen
        penalty += float(np.std(values)) * 500.0 + (max(values) - min(values)) * 100.0
        ind.objs[0], ind.objs[1] = -similarity, penalty
        return ind.objs

    def equalize_workload(self, ind: CommitteeIndividual, max_iters: int = 30):
        mscb_to_idx = {str(self.lect_mscb[idx]): idx for idx in range(self.n_lecturers)}
        for _ in range(max_iters):
            total_wl = Counter(self.previous_workload)
            for mems in ind.councils_members:
                for m in [int(x) for x in mems if x >= 0]: total_wl[str(self.lect_mscb[m])] += 1

            underload_ms = [m for m in self.lect_mscb if total_wl.get(str(m), 0) == 0 and self.lect_knows_english[mscb_to_idx[str(m)]]] if self.is_phase_1 else [m for m in self.lect_mscb if total_wl.get(str(m), 0) < self.target_min]
            if not underload_ms: break

            underload_indices = sorted([mscb_to_idx[str(m)] for m in underload_ms], key=lambda x: total_wl.get(str(self.lect_mscb[x]), 0))
            made_change = False
            
            for cid in range(ind.councils_members.shape[0]):
                mems, current_members = ind.councils_members[cid], set(int(x) for x in ind.councils_members[cid] if x >= 0)
                n_mem = len(current_members)
                if n_mem <= 1: continue

                sorted_pos = sorted(range(1, n_mem), key=lambda p: total_wl.get(str(self.lect_mscb[int(mems[p])]), 0), reverse=True)
                for pos in sorted_pos:
                    cur_idx = int(mems[pos])
                    cur_load = total_wl.get(str(self.lect_mscb[cur_idx]), 0)

                    for uidx in underload_indices:
                        uidx_load = total_wl.get(str(self.lect_mscb[uidx]), 0)
                        
                        if cur_load <= uidx_load + 1: continue
                        if uidx in current_members or (self.council_requires_english[cid] and not self.lect_knows_english[uidx]) or (not self.council_is_ce[cid] and self.lect_only_CE[uidx]): continue
                        
                        has_conflict = False
                        for ex_m in current_members:
                            if ex_m != cur_idx and uidx in self.conflict_graph[ex_m]:
                                has_conflict = True
                                break
                        if has_conflict: continue

                        mems[pos] = uidx
                        total_wl[str(self.lect_mscb[cur_idx])] -= 1
                        total_wl[str(self.lect_mscb[uidx])] += 1
                        current_members.remove(cur_idx); current_members.add(uidx); made_change = True
                        break
            if not made_change: break
        try: ind.objs = self.evaluate(ind)
        except Exception: pass
        return ind

    def mutate(self, ind: CommitteeIndividual):
        if random.random() > self.mr: return ind
        m = ind.copy()
        
        ops = ['swap_member', 'replace_member', 'scramble_councils', 'targeted_insert']
        op = random.choices(ops, weights=[0.25, 0.25, 0.25, 0.25])[0]

        try:
            if op == 'swap_member' and self.n_councils > 1:
                i, j = random.sample(range(self.n_councils), 2)
                n_i, n_j = m.get_num_members(i), m.get_num_members(j)
                if n_i > 0 and n_j > 0:
                    p1, p2 = random.randrange(n_i), random.randrange(n_j)
                    a, b = m.councils_members[i, p1], m.councils_members[j, p2]
                    if (p1 == 0 and not self.lect_is_ts[int(b)]) or (p2 == 0 and not self.lect_is_ts[int(a)]) or (self.council_requires_english[i] and not self.lect_knows_english[int(b)]) or (self.council_requires_english[j] and not self.lect_knows_english[int(a)]) or (not self.council_is_ce[i] and self.lect_only_CE[int(b)]) or (not self.council_is_ce[j] and self.lect_only_CE[int(a)]): return m
                    
                    if any(int(b) in self.conflict_graph[int(m.councils_members[i, p])] for p in range(n_i) if p != p1): return m
                    if any(int(a) in self.conflict_graph[int(m.councils_members[j, p])] for p in range(n_j) if p != p2): return m
                    
                    m.councils_members[i, p1], m.councils_members[j, p2] = b, a

            elif op == 'replace_member':
                i = random.randrange(self.n_councils)
                n_mem = m.get_num_members(i)
                if n_mem > 0:
                    pos = random.randrange(n_mem)
                    base_mask = self.lect_knows_english.copy() if self.council_requires_english[i] else np.ones(self.n_lecturers, dtype=bool)
                    if not self.council_is_ce[i]: base_mask &= ~self.lect_only_CE
                    cand_mask = base_mask & self.lect_is_ts if pos == 0 else base_mask
                    
                    current_members = set(m.councils_members[i][:n_mem])
                    current_members.remove(m.councils_members[i, pos])
                    
                    available = [idx for idx in np.where(cand_mask)[0] if idx not in current_members]
                    available = [idx for idx in available if not any(idx in self.conflict_graph[ex_m] for ex_m in current_members)]
                    
                    if available:
                        curr_wl = Counter(self.previous_workload)
                        for c in m.councils_members:
                            for mem in [x for x in c if x >= 0]: curr_wl[str(self.lect_mscb[mem])] += 1
                        m.councils_members[i, pos] = int(np.random.choice(available, p=self.get_candidate_weights(available, curr_wl, self.council_is_ce[i])))

            elif op == 'scramble_councils' and self.n_councils > 1:
                c1, c2 = random.sample(range(self.n_councils), 2)
                if self.council_requires_english[c1] == self.council_requires_english[c2] and self.council_is_ce[c1] == self.council_is_ce[c2]:
                    mems1 = [x for x in m.councils_members[c1][1:] if x >= 0] 
                    mems2 = [x for x in m.councils_members[c2][1:] if x >= 0]
                    pool = mems1 + mems2
                    random.shuffle(pool)
                    idx = 0
                    for pos in range(1, m.max_k_members):
                        if m.councils_members[c1, pos] >= 0:
                            m.councils_members[c1, pos] = pool[idx]; idx += 1
                    for pos in range(1, m.max_k_members):
                        if m.councils_members[c2, pos] >= 0:
                            m.councils_members[c2, pos] = pool[idx]; idx += 1

            elif op == 'targeted_insert':
                curr_wl = Counter(self.previous_workload)
                for c in m.councils_members:
                    for mem in [x for x in c if x >= 0]: curr_wl[str(self.lect_mscb[mem])] += 1
                
                underloaded = [idx for idx in range(self.n_lecturers) if curr_wl[str(self.lect_mscb[idx])] == 0 and self.lect_knows_english[idx]] if self.is_phase_1 else [idx for idx in range(self.n_lecturers) if curr_wl[str(self.lect_mscb[idx])] < self.target_min]
                
                if underloaded:
                    u_idx = random.choice(underloaded)
                    valid_councils = [cid for cid in range(self.n_councils) if 
                                      (not self.council_requires_english[cid] or self.lect_knows_english[u_idx]) and 
                                      (self.council_is_ce[cid] or not self.lect_only_CE[u_idx]) and 
                                      (u_idx not in m.councils_members[cid])]
                    
                    if valid_councils:
                        cid = random.choice(valid_councils)
                        n_mem = m.get_num_members(cid)
                        if n_mem > 1:
                            sorted_pos = sorted(range(1, n_mem), key=lambda p: curr_wl.get(str(self.lect_mscb[int(m.councils_members[cid, p])]), 0), reverse=True)
                            
                            for pos_to_replace in sorted_pos:
                                has_conflict = False
                                for p in range(n_mem):
                                    if p != pos_to_replace and u_idx in self.conflict_graph[int(m.councils_members[cid, p])]:
                                        has_conflict = True
                                        break
                                
                                if not has_conflict and curr_wl.get(str(self.lect_mscb[int(m.councils_members[cid, pos_to_replace])]), 0) > curr_wl[str(self.lect_mscb[u_idx])] + 1:
                                    m.councils_members[cid, pos_to_replace] = u_idx
                                    break

        except Exception: pass
        return m

    def tournament(self, population):
        cand = random.sample(population, min(self.tourn, len(population)))
        return min(cand, key=lambda ind: (ind.rank, -ind.crowding)).copy()

    def crossover(self, p1, p2):
        if random.random() > self.cr: return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        mask = np.random.rand(self.n_councils) < 0.5
        c1.councils_members[mask], c2.councils_members[mask] = p2.councils_members[mask], p1.councils_members[mask]
        return c1, c2

    def fast_non_dominated_sort(self, population):
        N = len(population)
        if N == 0: return []
        objs = np.vstack([ind.objs for ind in population])
        dominates, dominated_sets = np.zeros(N, dtype=int), [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                le, sl = np.all(objs[i] <= objs[j]), np.any(objs[i] < objs[j])
                if le and sl: dominated_sets[i].append(j); dominates[j] += 1
                elif np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]): dominated_sets[j].append(i); dominates[i] += 1
        fronts = [np.where(dominates == 0)[0].tolist()]
        for idx in fronts[0]: population[idx].rank = 0
        while fronts[-1]:
            next_front = []
            for i in fronts[-1]:
                for j in dominated_sets[i]:
                    dominates[j] -= 1
                    if dominates[j] == 0: population[j].rank = len(fronts); next_front.append(j)
            if next_front: fronts.append(next_front)
            else: break
        return fronts[:-1]

    def compute_crowding(self, front, population):
        if not front: return
        objs = np.array([population[i].objs for i in front])
        dists = np.zeros(len(front))
        for m in range(objs.shape[1] if objs.ndim > 1 else 1):
            order = np.argsort(objs[:, m] if objs.ndim > 1 else objs)
            dists[order[0]], dists[order[-1]] = float('inf'), float('inf')
            if len(order) > 2:
                min_v, max_v = (objs[order[0], m], objs[order[-1], m]) if objs.ndim > 1 else (objs[order[0]], objs[order[-1]])
                if abs(max_v - min_v) > 1e-12:
                    for idx in range(1, len(order) - 1):
                        dists[order[idx]] += ((objs[order[idx + 1], m] - objs[order[idx - 1], m]) if objs.ndim > 1 else (objs[order[idx + 1]] - objs[order[idx - 1]])) / (max_v - min_v)
        for idx, val in zip(front, dists): population[idx].crowding = val

    def evolve(self):
        pop = []
        for _ in range(self.pop_size * 5):
            ind = self.random_individual()
            if self.validate(ind): ind.objs = self.evaluate(ind); pop.append(ind)
            if len(pop) == self.pop_size: break
        if not pop: return None

        for f in self.fast_non_dominated_sort(pop): self.compute_crowding(f, pop)
        best, best_score = min(pop, key=lambda x: sum(x.objs)), sum(min(pop, key=lambda x: sum(x.objs)).objs)

        for g in range(self.n_gen):
            if g % 20 == 0: logger.info(f"Gen {g}/{self.n_gen}  best_score {best_score:.2f}")
            offspring = []
            while len(offspring) < self.pop_size:
                c1, c2 = self.mutate(self.crossover(self.tournament(pop), self.tournament(pop))[0]), self.mutate(self.crossover(self.tournament(pop), self.tournament(pop))[1])
                for c in (c1, c2):
                    if len(offspring) < self.pop_size and self.validate(c): c.objs = self.evaluate(c); offspring.append(c)

            combined = pop + offspring
            new_pop = []
            for f in self.fast_non_dominated_sort(combined):
                self.compute_crowding(f, combined)
                new_pop.extend(combined[idx] for idx in sorted(f, key=lambda i: (combined[i].rank, -combined[i].crowding))[:self.pop_size - len(new_pop)])
                if len(new_pop) >= self.pop_size: break
            
            pop = [self.equalize_workload(ind, 5) if g >= int(self.n_gen * 0.4) else ind for ind in new_pop]
            curr_best = min(pop, key=lambda x: sum(x.objs))
            if sum(curr_best.objs) < best_score - 1e-6: best, best_score = curr_best, sum(curr_best.objs)

        return self.equalize_workload(best, max_iters=50) if best else None