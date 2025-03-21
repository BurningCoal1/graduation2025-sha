import cv2
import easyocr
from ultralytics import YOLO

# importing the .pt files
model_license = YOLO("best2.pt")
# image input and putput paths as variables
image_path = "inputs\\"
output_path = "outputs\\"
# image's name
image_name = "test19.jpg"
# reading the image in CV2
image = cv2.imread(image_path + image_name)
# reading the image for license plates using YOLO
result1 = model_license.predict(image_path + image_name, save=False)
# variable for naming outputs
number = 0
# intializing the OCR using EasyOCR
OCR = easyocr.Reader(['ar'])
# for every detected plate do the following
for results1 in result1:
    for box2 in results1.boxes:
        number += 1
        # map corners of the plate in the image to crop it
        x1, y1, x2, y2 = map(int, box2.xyxy[0].tolist())
        # Crop the CV2 image using the boxes in the YOLO image
        cropped_image = image[y1:y2, x1:x2]
        # save the cropped image in color, greyscale, and binary
        final_image = cv2.resize(cropped_image, (cropped_image.shape[1] * 2, cropped_image.shape[0] * 2))
        cv2.imwrite(output_path + "cropped image" + str(number) + ".jpg", final_image)
        # resizing, de-noising and enhancing the color image
        resized_image = cv2.resize(final_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        de_noise_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
        cv2.imwrite(output_path + "cropped image final" + str(number) + ".jpg", de_noise_image)
        # applying OCR on all four images(colored, grey, binary and enhanced)
        color_image_output = OCR.readtext(output_path + "cropped image" + str(number) + ".jpg")
        enhanced_image_output = OCR.readtext(output_path + "cropped image final" + str(number) + ".jpg")
        # only show the best results found , only apply if any result is found
        if enhanced_image_output:
            best_match_found4 = max(enhanced_image_output, key=lambda x: x[2])
            enhanced_image_output = best_match_found4[1]
            print("easyocr enhanced = " + enhanced_image_output)
            # enhanced_image_output is the variable containing license letters

