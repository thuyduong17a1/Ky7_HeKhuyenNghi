# HỆ KHUYẾN NGHỊ DỰA TRÊN TRƯỜNG HỢP (CBR) CHO XE CŨ

- **Học phần :** Hệ khuyến nghị
- **Giáo viên hướng dẫn :** Trần Bảo Trung
- **Sinh viên thực hiện :** Đỗ Thùy Dương
- **Mã sinh viên :** 23174600061
- **Lớp :** DHKL17A1HN

## 1. Mô tả đề tài
Hệ thống tư vấn khuyến nghị xe cũ dựa trên thuật toán Case-Based Reasoning (CBR) kết hợp cơ chế Critiquing (phê bình tương tác) trên tập dữ liệu Car Dekho.

## 2. Dữ liệu & Công nghệ
- **Tập dữ liệu:** Car Dekho Used Cars (4.340 bản ghi gốc, 3.577 bản ghi sau làm sạch) 
- **Ngôn ngữ & Thư viện:** Python 3.12, NumPy, Pandas, Matplotlib.
- **Giao diện Web:** HTML, JavaScript và máy chủ HTTP chuẩn của Python.
- **Kiểm thử tự động:** `unittest` (tích hợp sẵn trong `recommendation.py` để kiểm thử lõi truy hồi, giữ điều kiện và nới lỏng có bảo vệ).

## 3. Cấu trúc thư mục
```text
├── CAR DETAILS FROM CAR DEKHO.csv : Tập dữ liệu gốc xe cũ.
├── Code_DoThuyDuong.ipynb : File code lần 2 (tiền xử lý, CBR/Critiquing và mô phỏng sơ bộ)
├── PhuLucB_BaoCaoTienDo.docx : File báo cáo tiến độ lần 2 (Mốc KT2)
├── BaoCaoLan2_DoThuyDuong.docx : File báo cáo lần 2
├── BaoCaoCuoiKy_DoThuyDuong.pdf : File báo cáo tổng kết cuối kỳ
└── ThucHanh/ : Thư mục mã nguồn và thực nghiệm hoàn thiện
    ├── recommendation.py : Module hợp nhất: lõi CBR (CarRecommender), mô phỏng đánh giá (evaluate), vẽ biểu đồ (figures) và kiểm thử đơn vị (test)
    ├── website/app.py : Máy chủ web HTTP cục bộ kết nối giao diện với recommendation.py
    ├── website/web.html : Giao diện người dùng tương tác 
    ├── Code_DoThuyDuong_Lan3.ipynb : Notebook tổng hợp quy trình và hiển thị kết quả hoàn thiện
    ├── results/ : Thư mục lưu bảng kết quả CSV, tham số JSON và nhật ký phiên
    └── figures/ : Thư mục lưu các biểu đồ trực quan hóa kết quả
