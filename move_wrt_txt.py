import os
import shutil
from glob import glob

txt_file = "/home/faurecia/FAQT-retinanet/combined_annotated_dont_delete/gloss/big_defects.txt"
dest = "/home/faurecia/FAQT-retinanet/combined_annotated_dont_delete/gloss/images_bigdefects/"

src1 = "/home/faurecia/FAQT-retinanet/combined_annotated_dont_delete/gloss/images/"
src2 = ""
src3 = ""

with open(txt_file, "r") as txt :
	data = txt.readlines()

cnt = 0

for i in data:

	if not os.path.isfile(dest+i.strip()):

		if os.path.isfile(src1+i.strip()):

			shutil.copyfile(src1+i.strip(), dest+i.strip())
			cnt += 1
			print(cnt)

		if os.path.isfile(src2+i.strip()):

			shutil.copyfile(src2+i.strip(), dest+i.strip())
			cnt += 1
			print(cnt)

		if os.path.isfile(src3+i.strip()):

			shutil.copyfile(src3+i.strip(), dest+i.strip())
			cnt += 1
			print(cnt)

"""

with open(txt_file, "r") as txt :
	data = txt.readlines()

cnt = 0

for i in data:

	if not os.path.isfile(dest+i.strip()):

		if os.path.isfile(src1+i.strip()):

			shutil.copyfile(src1+i.strip(), dest+i.strip())
			cnt += 1
			print(cnt)

		if os.path.isfile(src2+i.strip()):

			shutil.copyfile(src2+i.strip(), dest+i.strip())
			cnt += 1
			print(cnt)

		if os.path.isfile(src3+i.strip()):

			shutil.copyfile(src3+i.strip(), dest+i.strip())
			cnt += 1
			print(cnt)
"""