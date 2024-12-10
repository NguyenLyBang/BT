# Hệ thống nhận diện cảm xúc qua khuôn mặt

Dự án này được xây dựng để tạo ra một hệ thống nhận diện cảm xúc từ khuôn mặt con người sử dụng học sâu và các kỹ thuật thị giác máy tính. Hệ thống sử dụng mô hình Mạng Nơ-ron Tích chập (CNN) để phân loại các cảm xúc cơ bản từ khuôn mặt. Mô hình được huấn luyện trên tập dữ liệu FER2013Plus và có thể nhận diện các cảm xúc như "Giận dữ", "Vui vẻ", "Buồn bã",... theo thời gian thực thông qua webcam.

## Tính năng
- **Phân loại cảm xúc**: Hệ thống có thể phân loại 7 cảm xúc cơ bản từ một bức ảnh khuôn mặt.
- **Nhận diện cảm xúc theo thời gian thực**: Mô hình có thể được tích hợp với webcam để nhận diện cảm xúc từ video trực tiếp.
- **Huấn luyện mô hình**: Mô hình được huấn luyện với tập dữ liệu FER2013Plus, bao gồm các ảnh khuôn mặt với nhãn cảm xúc tương ứng.
- **Tăng cường dữ liệu**: Dữ liệu được tăng cường để cải thiện tính đa dạng và giúp mô hình giảm overfitting.
- **Ma trận nhầm lẫn và báo cáo phân loại**: Sau khi đánh giá, hệ thống xuất ra ma trận nhầm lẫn và báo cáo phân loại để đánh giá hiệu quả mô hình.

## Tập dữ liệu
Dự án sử dụng tập dữ liệu **FER2013Plus**, chứa hơn 80.000 ảnh khuôn mặt đen trắng, mỗi ảnh được gán nhãn với một trong các cảm xúc sau:
- Giận dữ (Angry)
- Ghê tởm (Disgust)
- Sợ hãi (Fear)
- Vui vẻ (Happy)
- Buồn bã (Sad)
- Ngạc nhiên (Surprise)
- Trung tính (Neutral)

### Cấu trúc thư mục:
- `train/`: Chứa dữ liệu huấn luyện (80% tập dữ liệu).
- `val/`: Chứa dữ liệu xác thực (20% tập dữ liệu).
- `test/`: Chứa dữ liệu kiểm thử (dùng để đánh giá cuối cùng).

## Yêu cầu

Để chạy được dự án này, bạn cần cài đặt các thư viện sau:

- Python 3.x
- OpenCV
- NumPy
- TensorFlow (Keras)
- Scikit-learn
- Matplotlib

Bạn có thể cài đặt các thư viện cần thiết bằng cách sử dụng:

```bash
pip install opencv-python numpy tensorflow scikit-learn matplotlib
