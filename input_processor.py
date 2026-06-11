import os
import logging
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---------------- CONFIGURATION ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("input_processor")

# MODEL SELECTION:
# Option A: "all-mpnet-base-v2" (English Only - Best performance for English)
# Option B: "paraphrase-multilingual-mpnet-base-v2" (Supports Vietnamese - Use if many abstracts are missing English)
MODEL_NAME = "all-mpnet-base-v2" 

CACHE_FILE = "project_embeddings_cache.pt"
MIN_TEXT_LEN = 15  # If English text is shorter than this, we attempt fallback

# Column Mapping (Must match your Excel Headers exactly)
COL_ID = "Mã đồ án"
COL_TITLE_EN = "Tên đồ án (Tiếng Anh)"
COL_DESC_EN = "Mô tả (Tiếng Anh)"
COL_TITLE_VI = "Tên đồ án (Tiếng Việt)"
COL_DESC_VI = "Mô tả (Tiếng Việt)"

class ProjectData:
    """
    Structure to hold project info + vector.
    """
    def __init__(self, row_data, vector):
        # Normalize ID: Uppercase and stripped for consistent keys
        self.id = str(row_data.get(COL_ID, "")).strip().upper() 
        self.raw_data = row_data 
        self.vector = vector
        
    def __repr__(self):
        return f"<Project {self.id}>"

class InputLoader:
    def __init__(self, excel_path, model_name=MODEL_NAME, cache_path=CACHE_FILE):
        self.excel_path = excel_path
        self.cache_path = cache_path
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = None 
        self.embedding_cache = {} 
        
        # Check if model is known to be multilingual
        self.is_multilingual = "multilingual" in model_name.lower()

        self._load_cache()

    def _load_cache(self):
        """Load embeddings from disk if available."""
        if os.path.exists(self.cache_path):
            logger.info(f"Loading embedding cache from {self.cache_path}...")
            try:
                self.embedding_cache = torch.load(self.cache_path)
            except Exception:
                self.embedding_cache = {}

    def _save_cache(self):
        """Save embeddings to disk."""
        torch.save(self.embedding_cache, self.cache_path)

    def _get_model(self):
        """Lazy load the AI model."""
        if self.model is None:
            logger.info(f"Loading Model: {self.model_name} (Multilingual={self.is_multilingual})")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def _get_text_to_embed(self, row):
        """
        Determines the best text to embed.
        1. Prefer English.
        2. If English is missing/too short, try Vietnamese.
        3. Only use Vietnamese if the model supports it (Multilingual).
        """
        # 1. Try English
        t_en = str(row.get(COL_TITLE_EN, "")).strip()
        d_en = str(row.get(COL_DESC_EN, "")).strip()
        if t_en.lower() == 'nan': t_en = ""
        if d_en.lower() == 'nan': d_en = ""
        
        text_en = f"{t_en}. {d_en}".strip()

        # 2. Check Length (threshold for valid English input)
        if len(text_en) >= MIN_TEXT_LEN:
            return text_en
        
        # 3. Fallback logic
        t_vi = str(row.get(COL_TITLE_VI, "")).strip()
        d_vi = str(row.get(COL_DESC_VI, "")).strip()
        if t_vi.lower() == 'nan': t_vi = ""
        if d_vi.lower() == 'nan': d_vi = ""
        text_vi = f"{t_vi}. {d_vi}".strip()

        if len(text_vi) > 5:
            if self.is_multilingual:
                return text_vi
            else:
                # English model can't handle Vietnamese -> Return None (becomes Zero Vector)
                return None 
        
        return None

    def process_excel(self):
        """
        Reads all sheets in the Excel file.
        Returns: List[List[ProjectData]] -> A list of arrays, one per sheet.
        """
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"File not found: {self.excel_path}")

        logger.info(f"Processing Excel: {self.excel_path}")
        # Load all sheets
        xls = pd.read_excel(self.excel_path, sheet_name=None, engine='openpyxl')
        
        all_sheets_data = [] 
        texts_to_embed = []
        ids_to_embed = []
        
        # Maintain order and association with sheets
        sheet_map = {} # sheet_name -> list of row_dicts

        # --- PASS 1: Read Data & Identify Missing Embeddings ---
        for sheet_name, df in xls.items():
            if df.empty: continue
            
            clean_rows = []
            for _, row in df.iterrows():
                # Get ID
                pid = str(row.get(COL_ID, "")).strip().upper()
                # Skip invalid IDs
                if not pid or pid == 'NAN': continue

                row_dict = row.to_dict()
                clean_rows.append(row_dict)

                # If not in cache, we need to prepare it for embedding
                if pid not in self.embedding_cache:
                    text = self._get_text_to_embed(row_dict)
                    if text:
                        texts_to_embed.append(text)
                        ids_to_embed.append(pid)
                    else:
                        # Will be set to 0 later
                        pass

            sheet_map[sheet_name] = clean_rows

        # --- PASS 2: Batch AI Embedding ---
        if texts_to_embed:
            logger.info(f"Computing embeddings for {len(texts_to_embed)} new items...")
            model = self._get_model()
            embeddings = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
            
            # Normalize vectors (Important for Cosine Similarity)
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

            for pid, vec in zip(ids_to_embed, embeddings):
                self.embedding_cache[pid] = vec
            
            self._save_cache()
        else:
            logger.info("All necessary embeddings found in cache.")

        # --- PASS 3: Construct Result Objects ---
        for sheet_name, rows in sheet_map.items():
            sheet_projects = []
            for r in rows:
                pid = str(r.get(COL_ID, "")).strip().upper()
                
                if pid in self.embedding_cache:
                    vec = self.embedding_cache[pid]
                else:
                    # Fallback to Zero Vector
                    vec = np.zeros(768) 
                
                sheet_projects.append(ProjectData(r, vec))
            
            if sheet_projects:
                all_sheets_data.append(sheet_projects)
                logger.info(f"Loaded sheet '{sheet_name}': {len(sheet_projects)} projects")

        return all_sheets_data