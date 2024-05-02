from typing import TYPE_CHECKING
if TYPE_CHECKING: #always False; initialized in warmup.py
    import os, time
    from sentence_transformers import SentenceTransformer, util
    import torch
    model: SentenceTransformer 

corpus = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

queries = ["llama weight", "how long llamas live", "what llamas eat"]

f = open("data/t2_data", "r")
pages = f.read().lower().split('===')
f.close()
filtered = [page for page in pages if len(page) <= 64]
pages = [page for page in pages if len(page) > 64]
# for f_page in filtered: print(f_page, '='*20)
print('pages_num:',len(pages))

corpus = pages
queries = ['nell brother']

# check if tensor exists
if os.path.exists('data/t2_data.pt'):
    corpus_embeddings = torch.load('data/t2_data.pt')
else: # encode the corpus
  start_time = time.time()
  corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True) 
  end_time = time.time()
  print(f"Time taken to encode the corpus: {end_time - start_time:.4f} seconds")
  torch.save(corpus_embeddings, "data/t2_data.pt")

# Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 3 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print('='*16, query, '='*16)
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"{score:.4f}=={corpus[idx]}")

