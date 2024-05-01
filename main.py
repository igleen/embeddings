import os, time, traceback

from sentence_transformers import SentenceTransformer, util
from torch import topk

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder.encode(["warmup"], convert_to_tensor=True)

f_path = "embeddings.py"

while True:
    command = input(":")
    command = command.split()
    if 'clear' in command:
        os.system('cls' if os.name == 'nt' else 'clear')
        continue
    if '-f' in command:
        if len(command) > command.index('-f') + 1:
            f_path = command[command.index('-f')+1] 
        else:
            print("Please provide a file path.")
            continue 
        
    if not os.path.exists(f_path):
        print("File not found")
        continue
    
    with open(f_path, "r") as f:
        try: exec(f.read())
        except Exception:
            print(traceback.format_exc())
            continue
