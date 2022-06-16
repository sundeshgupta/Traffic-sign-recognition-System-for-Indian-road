import os

DATA_DIR = "./All_annotation/"
for filename in os.listdir(DATA_DIR):
#     break
    if (os.path.splitext(filename)[-1]).lower() == '.txt':
        file_path = os.path.join(DATA_DIR, filename)
#         img = cv2.imread(img_path)
#         # img = cv2.resize(img, (256, 256))
#         # show(img, "original")
#         enhanced = dehaze(img, reduce = True)
#         # show(enhanced, "enhanced")
#         cv2.imwrite('../nighttime_itsd/' + filename, enhanced)
#         # cv2.destroyAllWindows()