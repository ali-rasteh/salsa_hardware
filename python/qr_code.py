import qrcode

# Replace with your GitHub repo URL
# url = "https://github.com/ali-rasteh/salsa_hardware"
url = "https://arxiv.org/abs/2504.09342"

# Create QR code
img = qrcode.make(url)

# Save the QR code as an image
file_path = "./figs/qr.png"
img.save(file_path)

print(f"QR code saved in {file_path}")
