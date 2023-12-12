import qrcode

file_path = r"models/pruned_model.onnx.gz"

with open(file_path, 'rb') as f:
    file_bytes = f.read()

print(file_bytes)

byte_numbers = list(file_bytes)

print(byte_numbers)


def create_qrcode_from_numbers(numbers):
    # Convert the list of numbers to a string
    data = numbers

    # Generate a QR code instance
    qr = qrcode.QRCode(
        version=40,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=5,
        border=4,
    )

    # Add the data to the QR code
    qr.add_data(data)
    qr.make(fit=True)

    # Create an image from the QR code
    img = qr.make_image(fill_color="black", back_color="white")

    # Save or display the image
    img.save("qrcode.png")
    img.show()

create_qrcode_from_numbers(byte_numbers)