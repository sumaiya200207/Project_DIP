import cv2
import numpy as np
#st_cascade = cv2.CascadeClassifier('C:\\Users\\USER\\AppData\\Local\\Programs\\Lib\\site-packages\\cv2\data\\haarcascade_frontalface_default.xml')
R=0.4
G=0.3
B=0.3

img = cv2.imread('C:\\Users\\USER\\Pictures\\chuck.jpg', 1)
bl, gr, re = cv2.split(img)

print("Red Matrix:")
print(re)
print("Green Matrix:")
print(gr)
print("Blue Matrix:")
print(bl)

new_r = np.clip(R * img[:, :, 2], 0, 255).astype(np.uint8)
new_g = np.clip(G * img[:, :, 1], 0, 255).astype(np.uint8)
new_b = np.clip(B * img[:, :, 0], 0, 255).astype(np.uint8)

    # Merge the new RGB matrices to form the new image
new_image = cv2.merge([new_b, new_g, new_r])

img = cv2.resize(img, (400, 300))
red_arr = np.zeros(img.shape, dtype="uint8")
red_arr[:,:,2] = img[:,:,2]


green_arr = np.zeros(img.shape, dtype="uint8")
green_arr[:,:,1] = img[:,:,1]


blue_arr = np.zeros(img.shape, dtype="uint8")
blue_arr[:,:,0] = img[:,:,0]

ht1 = np.hstack((red_arr, green_arr))
ht2 = np.hstack((blue_arr, img))
img_pop = np.vstack((ht1, ht2))

cv2.imshow("update img", img_pop)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("New Image", new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()






