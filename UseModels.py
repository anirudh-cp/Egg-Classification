from main import CNN, RESIZED, CLASSES
import torch
from PIL import Image
import torchvision.transforms as transforms

import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from PIL import Image,ImageTk
from pathlib import Path


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Image Classifier')
        #self.geometry("300x280")
        
        Modelframe = tk.LabelFrame(self, text='Model', relief=tk.GROOVE, bd=4)
        Modelframe.pack(padx=10, pady=10, fill=tk.X)
        Modelframe.pack()
        
        self.ModelPath = tk.StringVar()
        self.ModelPath.set(' None Selected')
        self.ModelPathText = ''
        
        self.selectModelButton = tk.Button(Modelframe, text='Select Model', 
                                           command=lambda : 
                                               self.handleModelLoad('Open Model',
                                                                    [('PyTorch Machine Learning Model (*.pth)', '*.pth')],
                                                                    self.ModelPath))
        self.selectModelButton.pack(padx=5, pady=7, side=tk.LEFT, ipadx=3)
        
        self.selectModelLabel = tk.Label(Modelframe, textvariable=self.ModelPath, borderwidth=2, relief=tk.GROOVE, anchor=tk.W)
        self.selectModelLabel.pack(padx=5, pady=7, ipady=3, fill=tk.X)
        
        
        Imageframe = tk.LabelFrame(self, text='Image', relief=tk.GROOVE, bd=4)
        Imageframe.pack(padx=10, pady=10, fill=tk.X)
        Imageframe.pack()
        
        ImagePathFrame = tk.Frame(Imageframe)
        ImagePathFrame.pack(fill=tk.X)
        
        self.ImagePath = tk.StringVar()
        self.ImagePath.set(' None Selected')
        self.ImagePathText = ''
        
        self.selectImageButton = tk.Button(ImagePathFrame, text='Select Image', 
                                           command=lambda : 
                                               self.handleImageLoad('Open Image', 
                                                                    [('PNG (*.png)', '*.png')],
                                                                    self.ImagePath))
        self.selectImageButton.pack(padx=5, pady=7, side=tk.LEFT, ipadx=3)

        self.selectImageLabel = tk.Label(ImagePathFrame, textvariable=self.ImagePath, borderwidth=2, relief=tk.GROOVE, anchor=tk.W)
        self.selectImageLabel.pack(padx=5, pady=7, ipady=3, fill=tk.X)
        
        self.ImageCanvas = tk.Canvas(Imageframe, borderwidth=2, relief=tk.GROOVE,
                                     width=RESIZED[1], height=RESIZED[0])    
        self.ImageCanvas.pack(padx=10, pady=7)

        Outframe = tk.LabelFrame(self, text='Output', relief=tk.GROOVE, bd=4)
        Outframe.pack(padx=10, pady=10, fill=tk.X)
        Outframe.pack()
        
        self.OutClass = tk.StringVar()
        self.OutClass.set(' Class: Undetermined')
        
        self.OutLabel = tk.Label(Outframe, textvariable=self.OutClass, borderwidth=2, relief=tk.GROOVE, anchor=tk.W)
        self.OutLabel.pack(padx=5, pady=7, ipady=3, fill=tk.X)
        
        self.classifyButton = tk.Button(Outframe, text='Classify', command=self.handleClassification)
        self.classifyButton.pack(padx=10, pady=(10, 7), fill=tk.X)

        self.resizable(1, 0)


    def __browse(self, title, fileTypes, Stringvar):
        path = fd.askopenfilename(parent=self, title=title, filetypes=fileTypes)
        path_ = Path(path)
        fileName = path_.name
        if fileName == '':
            fileName = 'None Selected'
        Stringvar.set(' ' + fileName)
        return path
    
    
    def handleModelLoad(self, title, fileTypes, Stringvar):
        self.ModelPathText = self.__browse(title, fileTypes, Stringvar)
        self.OutClass.set(' Class: Undetermined')
        
        
    def handleImageLoad(self, title, fileTypes, Stringvar):
        self.ImagePathText = self.__browse(title, fileTypes, Stringvar)
        self.OutClass.set(' Class: Undetermined')
        if self.ImagePathText != '':
            self.image = ImageTk.PhotoImage(Image.open(self.ImagePathText).resize((RESIZED[1], RESIZED[0])))     
            self.ImageCanvas.create_image(3, 3, anchor=tk.NW, image=self.image)    
            self.ImageCanvas.image = self.image  


    def handleClassification(self):
        # Read a PIL image
        
        if self.ImagePathText == '' or self.ModelPathText == '':
            mb.showerror(title='Input Error', message='Please specify both model and image file.')
            return
            
        image = Image.open(self.ImagePathText)
        transform = transforms.Compose([transforms.Resize(RESIZED), transforms.ToTensor()])
        img_tensor = transform(image)
        
        model = CNN()
        model.load_state_dict(torch.load(self.ModelPathText))
    
        out = model(img_tensor)
        # Pick index with highest probability
        _, preds  = torch.max(out, dim=1)
        # Retrieve the class label
        predLabel = CLASSES[preds[0].item()]
        
        self.OutClass.set(f' Class: {predLabel}')


def main():
    app = App()
    app.mainloop()
    

if __name__ == '__main__':
    main()
