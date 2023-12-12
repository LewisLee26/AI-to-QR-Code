from PIL import Image
from pyzbar.pyzbar import ZBarSymbol, decode
img = Image.open('model.png')
output = decode(img, symbols=[ZBarSymbol.QRCODE])
print(output[0].data)