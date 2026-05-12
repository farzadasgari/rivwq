<div align="center">

# River Water Quality Analysis

[![GitHub stars](https://img.shields.io/github/stars/farzadasgari/rivwq)](https://github.com/farzadasgari/rivwq/stargazers)
[![GitHub license](https://img.shields.io/github/license/farzadasgari/rivwq)](https://github.com/farzadasgari/rivwq/blob/main/LICENSE)
</div>

## How to use

### 1) Get the code
```bash
git clone https://github.com/farzadasgari/rivwq.git
cd rivwq
```

---

### 2) Create the conda environment (recommended)
This project provides a pinned conda environment (`environment.yml`) including the scientific stack, geospatial tools, and Jupyter.

```bash
conda env create -f environment.yml
conda activate rivwq
```

If you later update dependencies:
```bash
conda env update -f environment.yml --prune
conda activate rivwq
```

#### Alternative: install with pip
If you prefer a plain Python environment, create/activate a venv and install from `requirements.txt`:

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (PowerShell)
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Notes:
> - Some geospatial packages (e.g., `rasterio`, `cartopy`, `shapely`, `pyproj`) are usually easier via conda-forge, so the conda route is recommended.
> - `earthengine-api` is included (via pip in the conda environment), so you may need to authenticate before using Google Earth Engine.

---

### 3) Register the Jupyter kernel (optional but recommended)
Run this **after activating** the environment:

```bash
python -m ipykernel install --user --name rivwq --display-name "Python (rivwq)"
```

---

### 4) Launch JupyterLab and run the notebooks
```bash
jupyter lab
```

Open the notebooks in the repository and run them top-to-bottom.

---

### Contact
For any inquiries, please contact:
- khufarzadasgari@gmail.com
- std_farzad.asgari@khu.ac.ir

### Links

##### Farzad Asgari
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://farzadasgari.ir/)

[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=Rhue_kkAAAAJ&hl=en)

[![ResearchGate Badge](https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=for-the-badge)](https://www.researchgate.net/profile/Farzad-Asgari)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/farzad-asgari/)

[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--3800--0408-A6CE39?logo=orcid&logoColor=fff&style=for-the-badge)](https://orcid.org/0009-0008-3800-0408)

##### Seyed Hossein Mohajeri
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://khu.ac.ir/cv/1139/Seyed-Hossein-Mohajeri)

[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=E8PFUBEAAAAJ&hl=en)

[![ResearchGate Badge](https://img.shields.io/badge/ResearchGate-0CB?logo=researchgate&logoColor=fff&style=for-the-badge)](https://www.researchgate.net/profile/Seyed-Mohajeri-2)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](
https://ir.linkedin.com/in/hossein-mohajeri)

[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8056--293X-A6CE39?logo=orcid&logoColor=fff&style=for-the-badge)](https://orcid.org/0000-0002-8056-293X)
