from tkinter import *

import pandas as pd

from src.stereo_seq import data_analysis


def cluster():
    df = pd.read_csv("../data/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin_merge.tsv", sep="\t")
    data_analysis(df, w1.get(), w2.get(), w3.get()/100)


window = Tk()
window.geometry('400x400')
window.title('Clustering of cells by their position and gene expressions')

w1 = Scale(window, from_=50, to=150, length=400, tickinterval=10, orient=HORIZONTAL)
w1.set(70)
w1.pack()
Label(window, text='Epsilon').pack()

w2 = Scale(window, from_=5, to=100, length=400, tickinterval=5, orient=HORIZONTAL)
w2.set(10)
w2.pack()
Label(window, text='Minimum points').pack()

w3 = Scale(window, from_=0, to=100, length=400, tickinterval=10, orient=HORIZONTAL)
w3.set(70)
w3.pack()
Label(window, text='Similarity index').pack()

Label(window, text='').pack()
Label(window, text='').pack()

Button(window, text='Cluster', command=cluster).pack()

mainloop()
