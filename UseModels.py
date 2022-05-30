from main import CNN, RESIZED
import torch
from PIL import Image
import torchvision.transforms as transforms

import tkinter as tk


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Image Classifier')
        #self.geometry("300x280")
        
        Modelframe = tk.LabelFrame(self, text='Model', relief=tk.GROOVE, bd=4)
        Modelframe.pack(padx=10, pady=10, fill=tk.X)
        Modelframe.pack()
        
        self.selectModelButton = tk.Button(Modelframe, text='Select Model', command=self.handleModelPath)
        self.selectModelButton.pack(padx=5, pady=7, side=tk.LEFT, ipadx=3)
        
        self.ModelPath = tk.StringVar()
        self.ModelPath.set(' None Selected')
        self.selectModelLabel = tk.Label(Modelframe, textvariable=self.ModelPath, borderwidth=2, relief=tk.GROOVE, anchor=tk.W)
        self.selectModelLabel.pack(padx=5, pady=7, ipady=3, fill=tk.X)
        
        
        Imageframe = tk.LabelFrame(self, text='Image', relief=tk.GROOVE, bd=4)
        Imageframe.pack(padx=10, pady=10, fill=tk.X)
        Imageframe.pack()
        
        ImagePathFrame = tk.Frame(Imageframe)
        ImagePathFrame.pack(fill=tk.X)
        
        self.selectImageButton = tk.Button(ImagePathFrame, text='Select Image', command=self.handleModelPath)
        self.selectImageButton.pack(padx=5, pady=7, side=tk.LEFT, ipadx=3)
        
        self.ImagePath = tk.StringVar()
        self.ImagePath.set(' None Selected')
        self.selectImageLabel = tk.Label(ImagePathFrame, textvariable=self.ImagePath, borderwidth=2, relief=tk.GROOVE, anchor=tk.W)
        self.selectImageLabel.pack(padx=5, pady=7, ipady=3, fill=tk.X)
        
        self.ImageCanvas = tk.Canvas(Imageframe, background="white", borderwidth=2, relief=tk.GROOVE,
                                     width=RESIZED[1], height=RESIZED[0])
        self.ImageCanvas.pack(padx=10, pady=7)


        Outframe = tk.LabelFrame(self, text='Output', relief=tk.GROOVE, bd=4)
        Outframe.pack(padx=10, pady=10, fill=tk.X)
        Outframe.pack()
        
        self.OutClass = tk.StringVar()
        self.OutClass.set(' Class: Undetermined')
        
        self.OutLabel = tk.Label(Outframe, textvariable=self.OutClass, borderwidth=2, relief=tk.GROOVE, anchor=tk.W)
        self.OutLabel.pack(padx=5, pady=7, ipady=3, fill=tk.X)
        
        self.classifyButton = tk.Button(Outframe, text='Classify')
        self.classifyButton.pack(padx=10, pady=(10, 7), fill=tk.X)

        self.resizable(0, 0)


    def handleModelPath(self):
        self.ModelPath.set(' ABC')


def getImage():
    # Read a PIL image
    image = Image.open('PATH')
    
    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([transforms.Resize(RESIZED), transforms.ToTensor()])
    
    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)


def main():
    app = App()
    app.mainloop()
    

if __name__ == '__main__':
    main()
