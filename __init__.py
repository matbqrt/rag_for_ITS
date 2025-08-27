# import os
# import glob
# import importlib

# # Instantiate retrievers class
# retriever_dir = os.path.join(os.path.dirname(__file__), "retrieval", "retrievers")

# modules = glob.glob(os.path.join(retriever_dir, "*.py"))

# for path in modules:
#     name = os.path.basename(path)
    
#     if name.startswith("__") or name in ("base_retriever.py", "retriever.py"):
#         continue 
    
#     module_name = f"retrieval.retrievers.{name[:-3]}"
#     importlib.import_module(module_name)
