# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 23:13:01 2021

@author: Dell
"""
import streamlit as st
import numpy as np
from PIL import Image 
import os
from main import main
path = os.path.dirname(__file__)
def load_image(image_file):
    print(image_file)
    img = Image.open(image_file)
    return img 

def app():
    st.title("Hand drawn circuit recognition")
    menu = []
    menu.append('Upload image')
    for i in range(20):
        menu.append("Sample circuit "+str(i+1))
    choice = st.sidebar.radio("Sample Images",menu)
    
    if choice == "Upload image":
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    else:
        no = choice.split()
        image_file = path + '/Test_set/'+ no[2] + '.PNG'

    if image_file is not None :
        # file_type = "FileType:"+image_file.type
        # file_size = "FileSize:" +str(image_file.size)
        #st.write(file_type)
        #st.write(file_size)
        st.header("Different stages in recognition of the circuit")
        
        img = load_image(image_file)
        inp = np.array(img)
        rebuilt, comp, nodes, comp_list, jns_list, conn_list = main(inp,st)
        cols = st.columns(4)
        cols[0].image(img, width=150, caption= 'Scanned circuit')
        cols[1].image(comp, width=150, caption= 'Detected components')
        cols[2].image(nodes, width=150, caption= 'Nodes and terminals')
        cols[3].image(rebuilt, width=150, caption= 'Rebuilt circuit')
        
        st.header("Description of the circuit")
        st.subheader("Components in the circuit are: ")
        for i in range(len(comp_list)):
            st.write(comp_list[i])
        st.subheader("Nodes in the circuit are: ")
        for i in range(len(jns_list)):
            st.write(jns_list[i])
        st.subheader("Connections in the circuit are: ")
        for i in range(len(conn_list)):
            st.write(conn_list[i])
        
if __name__ == '__main__':
	app()
