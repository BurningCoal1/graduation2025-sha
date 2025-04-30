import cv2
# import easyocr
# import pytesseract
from ultralytics import YOLO
# importing the .pt files
model_license = YOLO("E:\\graduation 2024\\best2.pt")
model_letters = YOLO("E:\\graduation 2024\\letters_best.pt")
# model.train(data="E:\\graduation 2024\\data.yaml", epochs=10, imgsz=512, batch=16)
image_name = "test (12).jpg"
image_path = "E:\\graduation 2024\\inputs\\"
output_path = "E:\\graduation 2024\\outputs"
image = cv2.imread(image_path + image_name)
result1 = model_license.predict(image_path + image_name, save=False)
# all the letters and numbers mapped to their arabic equivalent
letters_map = {
    "alif": "أ", "baa": "ب", "taa": "ت", "thaa": "ث", "jeem": "ج", "7aa": "ح", "khaa": "خ", "daal": "د",
    "zaal": "ذ", "raa": "ر", "zay": "ز","seen": "س","sheen": "ش", "saad": "ص", "daad": "ض", "Taa": "ط",
    "Thaa": "ظ", "ain": "ع", "ghayn": "غ", "faa": "ف", "qaaf": "ق", "kaaf": "ك", "laam": "ل", "meem": "م",
    "noon": "ن", "haa": "هـ", "waw": "و", "yaa": "ى", "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
    "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩"}
number = 0
# OCR = easyocr.Reader(['ar'])
for results1 in result1:
    for box2 in results1.boxes:
        number += 1
        x1, y1, x2, y2 = map(int, box2.xyxy[0].tolist())
        cropped_image = image[y1:y2, x1:x2]
        final_image = cv2.resize(cropped_image, (cropped_image.shape[1]*2, cropped_image.shape[0]*2))
        # final_image_bw = cv2.threshold(final_image, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output_path + "\\cropped image" + str(number) + ".jpg", final_image)
        final_image_grey = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path + "\\cropped image grey" + str(number) + ".jpg", final_image_grey)
        _, final_image_bw = cv2.threshold(final_image_grey, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output_path + "\\cropped image binary" + str(number) + ".jpg", final_image_bw)
        resized_image = cv2.resize(final_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        de_noise_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
        cv2.imwrite(output_path + "\\cropped image final" + str(number) + ".jpg", de_noise_image)
        readpath = output_path + "\\cropped image" + str(number) + ".jpg"
        cv2.imread(readpath)
        result2 = model_letters.predict(readpath, save=False)
        for results2 in result2:
            detected_letters = []
            detected_numbers = []
            for box3 in results2.boxes:
                cls = int(box3.cls)
                class_name = results2.names[cls]
                arabic_class_name = letters_map.get(class_name, class_name)
                conf = float(box3.conf)
                x_min = box3.xyxy[0][0].item()
                if class_name.isdigit():
                    detected_numbers.append((arabic_class_name, conf, x_min))
                else:
                    detected_letters.append((arabic_class_name, conf, x_min))

                # detecions.append((arabic_class_name, conf, box3.xyxy[0].tolist()))
            sorted_letters = sorted(detected_letters, key=lambda x:x[2], reverse=True)
            sorted_numbers = sorted(detected_numbers, key=lambda x:x[2], reverse=True)
            # detecions_sorted = sorted(detecions, key=lambda x:x[2][0], reverse=True)
            license_plate_text = " ".join([char[0] for char in sorted_letters + sorted_numbers])
            print(license_plate_text)
        # text = OCR.readtext(output_path + "\\cropped image binary" + str(number) + ".jpg")
        # text2 = OCR.readtext(output_path + "\\cropped image" + str(number) + ".jpg")
        # text3 = OCR.readtext(output_path + "\\cropped image grey" + str(number) + ".jpg")
        # text4 = OCR.readtext(output_path + "\\cropped image final" + str(number) + ".jpg")
        # if text:
        #     best_match_found = max(text, key=lambda x: x[2])
        #     text = best_match_found[1]
        #     print("easyocr bw = " + text)
        # if text2:
        #     best_match_found2 = max(text2, key=lambda x: x[2])
        #     text2 = best_match_found2[1]
        #     print("easyocr color = " + text2)
        # if text3:
        #     best_match_found3 = max(text3, key=lambda x: x[2])
        #     text3 = best_match_found3[1]
        #     print("easyocr grey = " + text3)
        # if text4:
        #     best_match_found4 = max(text4, key=lambda x: x[2])
        #     text4 = best_match_found4[1]
        #     print("easyocr enhanced = " + text4)
        # for letters in text:
        #      i = 0
        #      for every_letter in letters[1]:
        #          if letters[1]:
        #              the_output = every_letter[i]
        #              i += 1
        #          else:
        #              break
        #     print("easyOCR bw = " + str(letters[1]))
        # for letters2 in text2:
        #     print("easyOCR color = " + str(letters2[1]))
print("Text read successful")
# results = model.val(data="E:\\graduation 2024\\data.yaml")
#
# precision = results.metrics['precision']  # Precision
# recall = results.metrics['recall']        # Recall
# map50 = results.metrics['map50']          # mAP@0.5
# map5095 = results.metrics['map']          # mAP@0.5:0.95
#
# # Print the metrics
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"mAP@0.5: {map50:.4f}")
# print(f"mAP@0.5:0.95: {map5095:.4f}")
