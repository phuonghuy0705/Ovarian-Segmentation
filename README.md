# Ovarian-Segmentation
## Cấu trúc thư mục dự án 
Ovarian-Segmentation/
├── create_dataset.py
├── preprocess_dataset.py
├── augment_dataset.py
├── convert2yoloformat.py
├── visualize.py
├── train_YOLOv8seg.ipynb
├── requirements.txt
├── OTU2D_8_layers_splitted/
│   ├── Chocolate_Cyst/
│   └── High-grade_Serous_Cystadenoma/
└── ...

## Bước 1: Cài đặt các package cần thiết 
```bash
pip install -r requirements.txt 
```
## Bước 2: Tạo dataset (train-val-test) từ bộ OTD2D_8layer 
```python
python create_dataset.py
```
## Bước 3: Tiền xử lý dữ liệu (cho cả 3 bộ) 
```python
python preprocess_dataset.py
```
## Bước 4: Tăng cường dữ liệu (cho bộ train) 
```python
python augment_dataset.py
```
## Bước 5: Chuyển về định dạng chuẩn của YOLO để huấn luyện  
```python
python convert2yoloformat.py
```
## Bước 6: Visualize data đã tạo  
```python
python visualize.py
```

## Bước 7: Huấn luyện mô hình với YOLOv8 Instance Segmentation 
```bash
train_YOLOv8seg.ipynb 
```