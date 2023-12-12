import qrcode

file_path = r"models/pruned_model.onnx.gz"

with open(file_path, 'rb') as f:
    file_bytes = f.read()

print(file_bytes)
print()

# byte_numbers = list(file_bytes)

# print(byte_numbers)

# data_list = []
# for i in byte_numbers:
#     data_list.append(str(i).encode('utf-8'))

# print(data_list)

# print(data_str)

# print(len(byte_numbers))

# print(bytes(byte_numbers))

data = qrcode.util.QRData(file_bytes, mode=qrcode.util.MODE_8BIT_BYTE , check_data=False)

qr = qrcode.QRCode(
    version=40,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=5,
    border=4,
)

# print(qr.get_matrix()) # returns a 2d matrix of true and false values

qr.add_data(file_bytes, optimize=0)
    


# print()
# print(type(qr.data_list[0].data))

# qr.data_list[0].data = qr.data_list[0].data[:10]  
# # print()
# print(qrcode.util.create_data(qr.version, qr.error_correction, qr.data_list))
# qr.data_cache = qrcode.util.create_data(qr.version, qr.error_correction, qr.data_list)
# qr.mask_pattern = None
# qr.map_data(qr.data_cache, qr.mask_pattern)

qr.make(fit=False)

# # qr.make_image()

img = qr.make_image(fill_color="black", back_color="white")
img.save("model.png")