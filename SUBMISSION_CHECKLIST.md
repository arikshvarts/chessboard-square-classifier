# Final Project Submission Checklist

**Due Date**: January 24, 2026
**Presentation**: January 20-21, 2026

---

## Code Submission

### Repository Requirements

- [ ] **GitHub repository** is public/accessible
 - URL: _______________________________

- [ ] **README.md** includes:
 - [ ] Environment setup (from `git clone` to running)
 - [ ] Training instructions
 - [ ] Inference/evaluation instructions
 - [ ] Requirements.txt with all dependencies

- [ ] **requirements.txt** is complete
 - [ ] All packages listed with versions
 - [ ] Tested in fresh environment

- [ ] **Code is runnable**
 - [ ] Tested `git clone` → setup → run
 - [ ] No hardcoded paths
 - [ ] Clear error messages

---

## Evaluation API

### Function Implementation

- [ ] **evaluate.py** contains `predict_board()` with EXACT signature:
 ```python
 def predict_board(image: np.ndarray) -> torch.Tensor
 ```

- [ ] **Input specification** is correct:
 - [ ] Type: `numpy.ndarray`
 - [ ] Shape: `(H, W, 3)`
 - [ ] Dtype: `uint8`
 - [ ] Range: `[0, 255]`
 - [ ] Channel order: RGB

- [ ] **Output specification** is correct:
 - [ ] Type: `torch.Tensor`
 - [ ] Shape: `(8, 8)`
 - [ ] Dtype: `torch.int64`
 - [ ] Device: CPU

- [ ] **Class encoding** matches spec:
 - [ ] 0-5: White pieces (P, R, N, B, Q, K)
 - [ ] 6-11: Black pieces (p, r, n, b, q, k)
 - [ ] 12: Empty
 - [ ] 13: OOD/Unknown

- [ ] **Tested** the function:
 ```bash
 python evaluate.py --image test_board.jpg
 ```

---

## Dataset Submission

### Dataset Upload

- [ ] **Dataset uploaded** to shared drive
 - [ ] University drive (up to 2TB), OR
 - [ ] Google Drive (with university account)
- [ ] **Share link included** in report
 - Link: _______________________________

### Dataset Format

- [ ] **Compliant format** created:
 ```
 dataset_root/
 ├── images/
 └── gt.csv
 ```

- [ ] **gt.csv has 3 columns**:
 - [ ] Column 1: image_name (e.g., frame_001234.jpg)
 - [ ] Column 2: FEN string
 - [ ] Column 3: View (white_bottom or black_bottom)

- [ ] **All images referenced** in gt.csv exist in images/ folder

- [ ] **Dataset verified**:
 ```bash
 python create_compliant_dataset.py --output compliant_dataset --verify
 ```

---

## Trained Models

### Model Files

- [ ] **Best model saved** to `checkpoints/best_model.pth`

- [ ] **Model download link** provided
 - Link: _______________________________
 - Instructions in README for where to place it

- [ ] **Model loads correctly**:
 ```bash
 python evaluate.py --image test.jpg
 # Should show: "✓ Model loaded from checkpoints/best_model.pth"
 ```

- [ ] **Performance documented**:
 - [ ] Validation accuracy: ______%
 - [ ] Test accuracy per fold documented

---

## Final Report

### Report Structure (Up to 20 pages)

- [ ] **1. Abstract**
 - [ ] Problem summary
 - [ ] Approach overview
 - [ ] Main results

- [ ] **2. Introduction**
 - [ ] Task description and motivation
 - [ ] Challenges
 - [ ] Main contributions

- [ ] **3. Related Work**
 - [ ] Academic papers cited
 - [ ] GitHub repositories cited
 - [ ] Dataset sources cited
 - [ ] Differences from prior work explained

- [ ] **4. Method**
 - [ ] Model architecture (ResNet50)
 - [ ] Input/output representation
 - [ ] Training procedure (7-fold CV)
 - [ ] Loss functions
 - [ ] Data augmentation
 - [ ] Diagrams included

- [ ] **5. Experiments**
 - [ ] Dataset description (games, frames, splits)
 - [ ] Evaluation metrics (accuracy, per-class)
 - [ ] Quantitative results (tables)
 - [ ] Qualitative results (visualizations)
 - [ ] Cross-game generalization analysis

- [ ] **6. Ablation Study** (REQUIRED)
 - [ ] Component X removed → results
 - [ ] Component Y removed → results
 - [ ] Comparison tables
 - [ ] Clear explanations

- [ ] **7. What Did Not Work** (Optional but Recommended)
 - [ ] Failed approaches documented
 - [ ] Negative results explained
 - [ ] Insights from failures

- [ ] **8. Discussion / Limitations**
 - [ ] Failure cases shown
 - [ ] Limitations discussed
 - [ ] Future improvements suggested

- [ ] **9. References**
 - [ ] All sources cited properly

### Report Format

- [ ] **PDF format**
- [ ] **12pt font** (or standard LaTeX defaults)
- [ ] **Maximum 20 pages**
- [ ] **English language**
- [ ] **No AI-generated filler text**

---

## Presentation (7-10 minutes)

### Presentation Content

- [ ] **1. Introduction** (~1 min)
 - [ ] Names and degree programs
 - [ ] Project choice
 - [ ] Brief motivation

- [ ] **2. Problem Statement** (~1 min)
 - [ ] Short and focused
 - [ ] Key challenge highlighted

- [ ] **3. Method & Solution** (~2-3 min)
 - [ ] High-level approach
 - [ ] Key components
 - [ ] Architecture diagram
 - [ ] NO low-level code details

- [ ] **4. What Makes Your Solution Special** (~1 min)
 - [ ] Differentiating factors
 - [ ] Novel ideas or design choices

- [ ] **5. Results & Ablation** (~2-3 min)
 - [ ] Key quantitative results
 - [ ] Visual examples
 - [ ] Ablation study results
 - [ ] Insights over numbers

- [ ] **6. What You Learned** (~1 min, optional)
 - [ ] Technical insights
 - [ ] Challenges overcome
 - [ ] Surprises encountered

### Presentation Quality

- [ ] **Visual slides** (not text-heavy)
- [ ] **Figures and diagrams** included
- [ ] **Key points focused** (not everything from report)
- [ ] **Time managed** (fits in 7-10 minutes)
- [ ] **Practiced** (rehearsed at least once)

### Registration

- [ ] **Registered** for presentation slot
 - Date: _______________________________
 - Time: _______________________________

---

## Optional (But Strongly Recommended)

### Project Webpage

- [ ] **GitHub Pages created** (username.github.io/project)
 - [ ] Visual results
 - [ ] Demo videos/GIFs
 - [ ] Clear project overview
 - [ ] Links to paper/code
- [ ] **URL**: _______________________________

Examples:
- https://feature-3dgs.github.io/
- https://bgu-cs-vil.github.io/FastJAM/

---

## Pre-Submission Tests

### Final Checks (Do These Right Before Submitting)

```bash
# 1. Fresh environment test
cd /tmp
git clone <your-repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate # or .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Test evaluation API
python evaluate.py --image test_board.jpg
# Should work without errors

# 3. Test web app
python app.py
# Open http://localhost:5000 and upload test image

# 4. Verify dataset
python create_compliant_dataset.py --output compliant_dataset --verify
# Should show VALIDATION PASSED

# 5. Check all files are committed
git status
# Should show: "nothing to commit, working tree clean"
```

---

## 📦 What to Submit (January 24, 2026)

1. **GitHub repository URL** with:
 - All source code
 - README.md with complete instructions
 - requirements.txt
 - evaluate.py with compliant `predict_board()`

2. **Dataset share link** (Google Drive / University drive)

3. **Trained models link** (Google Drive / University drive)

4. **Final report PDF** (up to 20 pages)

5. **(Optional) Project webpage URL**

---

## 🚨 Common Mistakes to Avoid

- Hardcoded absolute paths (use relative paths)
- Missing dependencies in requirements.txt
- Wrong `predict_board()` signature or encoding
- Dataset not in compliant format
- No ablation study in report
- Slides with too much text
- Presentation over 10 minutes
- Dataset not uploaded to shared drive
- No model download link provided
- Code not tested in fresh environment

---

## 📧 Submission Method

**Submission portal**: _______________________________
(Will be announced by instructor)

---

## Questions?

- Check course forum/discussion board
- Email instructor for clarifications
- Review project specification document again

---

** **
