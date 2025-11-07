import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(image):
    # Convert to float in [0,1]         Vi dividere med 255 fordi billeder er skrevet med en værdi fra 0 - 255 [255, 0, 128]
    # Fordi de fleste billede behandings formler bedre kan lide at have værdier fra 0-1 bruger vi dette til at lave det om så [1.0, 0,0, 0.5]
    img = image.astype(np.float32) / 255.0
    # Vi skriver dette fordi, OpenCV giver os billedet som BGR og vi vil gerne have det i RGB så vi omskriver det til RBG ved at gøre dette.
    R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]  # OpenCV loads BGR

    # Intensity             Dette er formlen for Intensity som står på vores slides
    I = (R + G + B) / 3.0

    # Saturation            Dette er formlen for Saturation som står op vores slides
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_val
    S[I == 0] = 0

    # Hue                   Dette er formlen for Hue som står på vors slides
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))

    # Vi skriver dette statement fordi vi har to betingelser for Hue
    # Så hvis B er mindre eller lig G får vi værdier theta ud, ellers siger vi 360 - theta
    #Dette er en måde at skrive if statement på via OpenCV
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    # Vi laver Hue værdier om til [0,1] så det hænger sammen med de andre værdier S og I
    H = H / (2 * np.pi)  # normalize to [0,1]

    return cv2.merge((H, S, I))

# --- Load image (BGR in OpenCV) ---
img_bgr = cv2.imread("Billeder/IMG_20251104_094219.jpg")

# Convert BGR → RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Convert to HSI
hsi_img = rgb_to_hsi(img_bgr)
H, S, I = cv2.split(hsi_img)

# --- Plot ---
plt.figure(figsize=(12,6))

#Her gør vi det at vi plotter vores 4 billeder i en matrix [1, 4, x] ved plt.imshow skriver vi vors værdier H, S og I
# og fortæller matplotlib hvilken farve vi gerne vil have billedet til at være med prompten cmap = "gray" for grayscale f.eks
# cmap="hsv" for hvs farvnerne og img_rgb for billedet som der ser ud i RGB.

# RGB image
plt.subplot(1,5,1)
plt.imshow(img_rgb)
plt.title("RGB")
plt.axis("off")

# H channel
plt.subplot(1,5,2)
plt.imshow(H, cmap="hsv")
plt.title("Hue")
plt.axis("off")

# S channel
plt.subplot(1,5,3)
plt.imshow(S, cmap="gray")
plt.title("Saturation")
plt.axis("off")

# I channel
plt.subplot(1,5,4)
plt.imshow(I, cmap="gray")
plt.title("Intensity")
plt.axis("off")

# HSI Picture
plt.subplot(1,5,5)
plt.imshow(hsi_img, cmap="hsv")
plt.title("HSI")
plt.axis("off")

plt.tight_layout()
plt.show()
