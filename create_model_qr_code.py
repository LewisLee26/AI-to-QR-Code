import qrcode

file_path = r"models/pruned_model.onnx.gz"

with open(file_path, 'rb') as f:
    file_bytes = f.read()

data = qrcode.util.QRData(file_bytes, mode=qrcode.util.MODE_8BIT_BYTE , check_data=False)

qr = qrcode.QRCode(
    version=40,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=5,
    border=4,
)

qr.add_data(file_bytes, optimize=0)

# print(qrcode.util.create_data(qr.version, qr.error_correction, qr.data_list))

qr.make(fit=False)

img = qr.make_image(fill_color="black", back_color="white")
img.save("model.png")

