
import cv2
import matplotlib.pyplot as plt

# Step 1: Load original document image
image_path = "dataset/original/document.png"
image = cv2.imread(image_path)

# Step 2: Check image
if image is None:
    raise FileNotFoundError("Document image not found!")

# Step 3: Get image information
height, width, channels = image.shape

print("===== PaperTrail =====")
print("Image loaded successfully!")
print("Height:", height)
print("Width:", width)
print("Channels:", channels)
print("Total Pixels:", height * width)

# Step 4: Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 5: Convert RGB image to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 6: Display images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title("Original Document")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Document")
plt.axis("off")

plt.tight_layout()
plt.show()

# Step 7: Save grayscale image
cv2.imwrite("dataset/processed/grayscale.png", gray_image)

print("Grayscale image saved successfully!")
