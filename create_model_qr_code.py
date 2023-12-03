import qrcode
import base64

file_path = r"models/pruned_model.onnx.gz"

x = b""
with open(file_path, 'rb') as f:
    for chunk in iter(lambda: f.read(8), b''):
        x += chunk

print(x)

qr = qrcode.QRCode(
    version=40,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

qr.add_data(x, optimize=0)
# qr.add_data("Hello world!", optimize=20)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("model.png")