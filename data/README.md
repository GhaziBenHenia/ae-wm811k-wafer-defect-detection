# Dataset

The dataset used in this project is **WM811K Wafer Map**.  
You can download it directly from Kaggle:

🔗 https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

Once downloaded, place the file in `data/LSWMD.pkl` and the rest of the pipeline (extraction, preprocessing, and training) will run automatically using this file.

---

### About the Dataset

The **WM811K** dataset contains over **811,000 wafer maps** collected from real semiconductor manufacturing lines.  
Each wafer map visually represents test results of integrated circuits on a wafer, showing **defective die patterns** such as *Center*, *Edge-Ring*, *Scratch*, and others.  

The dataset originates from the research paper:

> **Wu, Ming-Ju, Jyh-Shing R. Jang, and Jui-Long Chen.**  
> *“Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.”*  
> *IEEE Transactions on Semiconductor Manufacturing 28(1), 2015.*

These wafer maps are widely used for **defect pattern recognition**, **image-based fault diagnosis**, and **machine learning benchmarking** in semiconductor yield analysis.
