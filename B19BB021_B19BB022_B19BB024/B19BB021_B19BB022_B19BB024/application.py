from tkinter import *
from tkinter.colorchooser import askcolor
import cv2
import numpy as np
from PIL import ImageTk,Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tkinter import *  
from PIL import ImageTk,Image
from tkinter import filedialog
import tkinter as tk
from tkinter import Message,Text
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageTk,Image
import PIL
import os
global add

add=0
class Main(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    
    def __init__(self):
        
        self.root = Tk()
        self.root.title("Interactive Segmentation Tool")
        self.lb1=Label(self.root,text="Welcome To Self-Adapting Interactive Segmentation Tool",width=100,height=2,fg='black' ,font=('times',20,' bold'))
        self.lb1.place(x=-100,y=0)
        self.cordinates=Label(self.root,text="",width=100,height=2,fg='black' ,font=('times',20,' bold'))
        self.cordinates.place(x=-100,y=700)
        self.lb2=Label(self.root,text="",width=100,height=2,fg='black' ,font=('times',20,' bold'))
        self.lb2.place(x=-140,y=100)
        
        self.choose_image = Button(self.root, text='Select Image', command=self.choose, bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
        self.choose_image.place(x=10,y=100)
        self.save_button = Button(self.root, text='Save', command=self.save_bu, bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
        self.save_button.place(x=1100,y=400)
        self.segmentation = Button(self.root, text='Segmentation', command=self.seg, bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
        self.segmentation.place(x=10,y=200)
        self.pen_button = Button(self.root, text='Pen', command=self.use_pen, bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))

        self.pen_button.place(x=1100,y=100)
        
        
        
        self.eraser_button = Button(self.root, text='Eraser', command=self.use_eraser,  bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
        self.eraser_button.place(x=1100,y=200)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.place(x=1200,y=50)
        
        
        self.c = Canvas(self.root, width=650, height=650)
        self.c.place(x=450,y=150)
        #self.rect = self.c.create_oval(0,0,0,0, outline = 'white', fill = 'red')
        
        self.setup()
        self.root.mainloop()
    
    def choose(self):
        global address
        global ifile
        
        
        self.ifile = filedialog.askopenfile(parent=self.root,mode='rb',title='Choose a file')
        add=self.ifile
        print("l",add)
        self.address = os.path.abspath(self.ifile.name)
        global path
        self.path = Image.open(self.ifile)
        print(self.path)
        self.img=self.path
        add=self.img
        global main_image
        self.main_image = ImageTk.PhotoImage(self.path)
        self.lb2.configure(text="Input Image")
        self.original=self.img.resize((450,450), Image.ANTIALIAS)
        self.original_image=ImageTk.PhotoImage(self.original)
        self.c.create_image(0,0,anchor = NW, image = self.original_image)
       
    def preprocess(self,imagedata_original):
    
        kernel = np.ones((2,2), np.uint8)

        opening_op = cv2.morphologyEx(imagedata_original, cv2.MORPH_GRADIENT, kernel)
        m = cv2.resize(opening_op, (960,540))
        return opening_op
        cv2.imshow('Opening Operation', m)
        #cv2.waitKey(0)
        
        final_image = cv2.add(imagedata_original,opening_op)
        #return final_image
        n = cv2.resize(final_image, (540,540))
        cv2.imshow('Image', n)
        #cv2.imshow('Image', final_image)
        cv2.waitKey(0)
    def predict_image(self,p_img):
        p_img = cv2.resize(p_img,(120,120))
        p_img = cv2.resize(p_img,(128,128))
        model=load_model('D:/Users/Krishna Jajoo/Downloads/new_with_0_wbc_segmentation_model.h5', compile=False)
        model1=load_model('D:/Users/Krishna Jajoo/Downloads/new_with_255_wbc_segmentation_model.h5', compile=False)
        pred = model.predict(p_img.reshape(1,128,128,3))
        pred1=model1.predict(p_img.reshape(1,128,128,3))
        pred[pred>0.5] = 1
        pred[pred<0.6] = 0
        pred1[pred1>0.5] = 1
        pred1[pred1<0.6] = 0

        pred1 = pred1[:,:,:,1].reshape((128,128))
        pred = pred[:,:,:,1].reshape((128,128))


        #plt.imshow(pred)
        pred = np.uint8(pred)
        pred1=np.uint8(pred1)
        output=pred+pred1
        output=cv2.resize(output,(120,120))
        #model=load_model('D:/Users/Krishna Jajoo/Downloads/WBC_Display/wbc_segmentation_model_updated.h5', compile=False)
        #pred = model.predict(p_img.reshape(1,128,128,3))
        #pred[pred>0.5] = 1
        #pred[pred<0.6] = 0

        #pred = pred[:,:,:,1].reshape((128,128))
        #pred = np.uint8(pred)
        #output=pred
        #output=cv2.resize(output,(120,120))
        return output
    def seg(self):
        image = cv2.imread(self.address)
        image2=self.preprocess(self.predict_image(image))
        
        cv2.imwrite("pr1.png",image2)
        img=cv2.imread("pr1.png")
        #img = Image.open("pr1.png")
        
        #img = image2
        for i in range(len(img)):
          for j in range(len(img[i])):
            for k in range(0,3):
              if(img[i][j][k]==0):
                img[i][j][k]=255
              else:
                img[i][j][k]=0
        #plt.imshow(img)
        #cv2.imwrite("/content/testttttt/file2.png",img)
        #img

        #img = Image.open("file2.png")
        img=cv2.resize(img,(450,450))
        ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("black_white_output.png",bw_img)
        self.i = Image.open("black_white_output.png").convert("RGBA")
        self.draw = ImageDraw.Draw(self.i)
        img = Image.open("black_white_output.png")
        img = img.convert("RGBA")
      
        datas = img.getdata()
      
        newData = []
      
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
      
        img.putdata(newData)
        img.save("./transparent_output.png", "PNG")
        print("Successful")
        '''label.configure(image=image)
        label.image=image
        label.place(x=200,y=60)
        file.place(x=0,y=0)
        label.configure(image=img)
        label.image=img
        label.place(x=200,y=60)'''
        global main_image
        global boundary_image
        main_image=ImageTk.PhotoImage(Image.open(self.address))
        main=cv2.imread(self.address)
        bound=cv2.imread("transparent_output.png")
        
            
        boundary_image = ImageTk.PhotoImage(Image.open("transparent_output.png"))
        
        self.lb2.configure(text="Segmented Image")
        
        self.cat=self.c.create_image(0,0,anchor = NW, image = boundary_image)
    
    def save_bu(self):
        i_1 = Image.open(self.address).resize((450,450), Image.ANTIALIAS)
        i_2 = Image.open("final_trans.png")
        i_1.paste(i_2,(0, 0),i_2)
        i_1.show()
        
    def save(self):
        global image_number
        filename = "final_black_white_output.png"   # image_number increments by 1 at every save
        self.i.save(filename)
        
        
        

    def error_solved(self):
        self.save()
        img = Image.open("final_black_white_output.png").convert("RGBA")
        
        datas = img.getdata()

        newData = []

        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save("final_trans.png")
        self.imgoriginal = ImageTk.PhotoImage(Image.open(self.address))
        self.final = ImageTk.PhotoImage(Image.open("final_trans.png"))
        
        self.update()
        
        
    def update(self):
        self.c.itemconfig(self.cat,image= self.imgoriginal)
        self.c.itemconfig(self.cat,image= self.final)
        
        
    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)
        

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode
        
    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        
        if self.active_button == self.eraser_button :
            paint_color = 'white'
            

        if self.active_button == self.pen_button :
            paint_color = self.color

       
        
        if (self.old_x and self.old_y):
            #self.c.coords(self.rect, self.old_x,self.old_y,(self.old_x+5),(self.old_y+5))
            self.hat = self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill=paint_color , width=self.line_width)
            self.error_solved()
            self.c.delete(self.hat)
            
            
        self.old_x = event.x
        self.old_y = event.y
            

        

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def edit(self):
        print("n",self.ifile)
        New_window(self.ifile,self.address)   
    

if __name__ == '__main__':
    Main()
