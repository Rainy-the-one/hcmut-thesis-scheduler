# 🎓 Thesis Defense Council Scheduling System - HCMUT

This project provides an automated solution for scheduling thesis and capstone project defense councils. The system integrates the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** and **Greedy Clustering** to solve a complex multi-objective optimization problem, specifically tailored to the academic regulations and workflows of the Ho Chi Minh City University of Technology (HCMUT).

## 👨‍💻 Authors & Advisors

**Authors:**
* **Tô Quốc Tài** (K24 Student, Faculty of Computer Science and Engineering, HCMUT)
* **Nguyễn Cảnh Hưng** (K24 Student, Faculty of Computer Science and Engineering, HCMUT)
* **Lê Bảo Nguyên** (K24 Student, Faculty of Computer Science and Engineering, HCMUT)

**Advisors:**
* **Assoc. Prof. Dr. Quản Thành Thơ** (Dean of the Faculty of Computer Science and Engineering, HCMUT)
* **MSc. Nguyễn Thanh Tùng**

---

## ✨ Key Features

* **3-Layer Greedy Clustering:** Automatically groups thesis topics into councils based on academic similarity, merges groups to optimize resources, and balances the load to prevent under-allocated councils.
* **Semantic Vectorization (Sentence Transformers):** Utilizes AI models (`all-mpnet-base-v2` or multilingual variants) to deeply understand the semantic meaning of project titles and abstracts (both in Vietnamese and English) for highly accurate grouping.
* **Optimal Lecturer Assignment (NSGA-II):** Balances the workload across all faculty members while strictly adhering to the minimum/maximum council constraints.
* **Hard Conflict Resolution:** Includes a strict mechanism to prevent incompatible lecturers (based on predefined conflicts) from being assigned to the same council.
* **CC / Non-CC Routing:** Automatically identifies and prioritizes assigning English-proficient lecturers to councils associated with Advanced/High-Quality programs (English-taught).

---

## 🗂️ Module Architecture

The system is designed with a modular architecture to ensure easy maintenance and scalability.

| File Name | Role |
| :--- | :--- |
| **`main.py`** | The entry point of the application that coordinates data flow across all modules. |
| **`config.json`** | The user-facing configuration file for adjusting paths, council size constraints, and GA parameters without altering the code. |
| **`config.py`** | Parses `config.json` and provides global environment variables (e.g., `LECTURER_EXCEL_PATH`, `HARD_CONFLICTS`). |
| **`clustering.py`** | Contains the core Greedy Clustering logic to determine which topics belong to the same defense council. |
| **`optimizer.py`** | Houses the Genetic Algorithm (`CommitteeNSGA`) and Chromosome structure (`CommitteeIndividual`) for assigning lecturers to councils. |
| **`data_io.py`** | Handles all I/O operations: loading lecturer data, parsing role histories, and exporting the final Excel results. |
| **`input_processor.py`** | Cleans raw project data and interfaces with HuggingFace models to generate and cache Vector Embeddings. |
| **`utils.py`** | Defines the core `Topic` data structure and string normalization utilities (e.g., stripping academic titles like Dr., Prof.). |
| **`run.sh`** | Execution script for quick startup on Linux/macOS environments. |
| **`run.bat`** | Execution script for quick startup on Windows environments. |

---

## ⚙️ Installation & Configuration

### 1. System Requirements
Ensure you have Python 3.x installed along with the required dependencies:
```bash
pip install pandas numpy torch openpyxl sentence-transformers