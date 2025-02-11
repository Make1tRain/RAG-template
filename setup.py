from src.constants import * 
import os 

if not os.path.isdir(CHROMA_PATH): 
    os.makedirs(CHROMA_PATH)

if not os.path.isdir(DATA_PATH): 
    os.makedirs(DATA_PATH)
