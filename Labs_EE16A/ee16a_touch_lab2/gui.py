from tkinter import *
import time

on  = "#00ff00"
off = "#a6a6a6"
vec = {(0,0):0, (0,1):1, (0,2):2, (1,0):3, (1,1):4, (1,2):5, (2,0):6, (2,1):7, (2,2):8}
color = [off, off, off, off, off, off, off, off, off]

# TKINTER ROOT
root = Tk()

# DRAW FRAMES
main = Frame(root)

toprow = Frame(root)
toprow.pack(side=BOTTOM)

midrow = Frame(root)
midrow.pack(side=BOTTOM)

botrow = Frame(root)
botrow.pack(side=BOTTOM)

# TOP ROW
pixel00 = Frame(toprow)
pixel00.pack(side=LEFT)
button00 = Button(pixel00, text="0,0", fg="grey", bg=color[0], state=DISABLED)
button00.pack()

pixel01 = Frame(toprow)
pixel01.pack(side=LEFT)
button01 = Button(pixel01, text="0,1", fg="grey", bg=color[1], state=DISABLED)
button01.pack()

pixel02 = Frame(toprow)
pixel02.pack(side=LEFT)
button02 = Button(pixel02, text="0,2", fg="grey", bg=color[2], state=DISABLED)
button02.pack()

# MID ROW
pixel10 = Frame(midrow)
pixel10.pack(side=LEFT)
button10 = Button(pixel10, text="1,0", fg="green", bg=color[3], state=DISABLED)
button10.pack()

pixel11 = Frame(midrow)
pixel11.pack(side=LEFT)
button11 = Button(pixel11, text="1,1", fg="green", bg=color[4], state=DISABLED)
button11.pack()

pixel12 = Frame(midrow)
pixel12.pack(side=LEFT)
button12 = Button(pixel12, text="1,2", fg="grey", bg=color[5], state=DISABLED)
button12.pack()

# BOT ROW
pixel20 = Frame(botrow)
pixel20.pack(side=LEFT)
button20 = Button(pixel20, text="2,0", fg="grey", bg=color[6], state=DISABLED)
button20.pack()

pixel21 = Frame(botrow)
pixel21.pack(side=LEFT)
button21 = Button(pixel21, text="2,1", fg="grey", bg=color[7], state=DISABLED)
button21.pack()

pixel22 = Frame(botrow)
pixel22.pack(side=LEFT)
button22 = Button(pixel22, text="2,2", fg="grey", bg=color[8], state=DISABLED)
button22.pack()

i = 0
while True:
    try:
        color = [off, off, off, off, off, off, off, off, off]
        color[i] = on
        i += 1
        i = i % 8
        button00.configure(bg=color[0])
        button01.configure(bg=color[1])
        button02.configure(bg=color[2])
        button10.configure(bg=color[3])
        button11.configure(bg=color[4])
        button12.configure(bg=color[5])
        button20.configure(bg=color[6])
        button21.configure(bg=color[7])
        button22.configure(bg=color[8])
        root.update()
        root.update_idletasks()
    except KeyboardInterrupt:
        break
