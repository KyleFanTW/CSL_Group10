import cv2
import tkinter
from PIL import Image

'''
n = NumberRecognizer()
n.load()
'''

window_main = tkinter.Tk(className='Evaluate', )
window_main.geometry("400x200")
 
def evaluate_image():
    # Read the image path_image.png
    image = Image.open('path_image.png')
    num = pytesseract.image_to_string(image)
    print(f'Number: {num}')
	# Remove the saved image
#, config='-c tessedit_char_whitelist=0123456789'

button_submit = tkinter.Button(window_main, text ="Evaluate", command=evaluate_image)
button_submit.config(width=20, height=2)
 
button_submit.pack()
window_main.mainloop()
