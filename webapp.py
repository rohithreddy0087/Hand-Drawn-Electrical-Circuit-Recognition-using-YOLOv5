"""
Web app on streamlit
"""
from PIL import Image 
import os

import streamlit as st
import numpy as np
from main import main

path = os.path.dirname(__file__)

def app():
    """
    Streamlit page to showcase the implementation
    """
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
        image_file = 'Test_set/'+ no[2] + '.PNG'

    if image_file is not None :
        st.header("Different stages in recognition of the circuit")
        
        img = Image.open(image_file)
        inp = np.array(img)
        rebuilt, comp, nodes, comp_list, jns_list, conn_list = main(inp)
        cols = st.columns(4)
        cols[0].image(img, width=150, caption= 'Scanned circuit')
        cols[1].image(comp, width=150, caption= 'Detected components')
        cols[2].image(nodes, width=150, caption= 'Nodes and terminals')
        cols[3].image(rebuilt, width=150, caption= 'Rebuilt circuit')
        
        st.header("Description of the circuit")
        st.subheader("Components in the circuit are: ")
        for i,_ in enumerate(comp_list):
            st.write(comp_list[i])
        st.subheader("Nodes in the circuit are: ")
        for i,_ in enumerate(jns_list):
            st.write(jns_list[i])
        st.subheader("Connections in the circuit are: ")
        for i,_ in enumerate(conn_list):
            st.write(conn_list[i])
        
if __name__ == '__main__':
	app()
