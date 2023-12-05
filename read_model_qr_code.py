import cv2
from pyzbar.pyzbar import decode

def read_qr_code(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Decode QR codes
    decoded_objects = decode(image)

    # Print the decoded information
    for obj in decoded_objects:
        # print(f'Type: {obj.type}')
        # print(f'Data: {obj.data.decode("utf-8")}')
        print(f'Data: {obj.data}')
        print()

    return obj.data

# Replace 'your_qr_code_image.jpg' with the path to your QR code image
qr_code_path = 'code.png'
data = read_qr_code(qr_code_path)

file_path2 = r"models/test_write.onnx.gz"
with open(file_path2, 'wb') as f:
    f.write(data)

