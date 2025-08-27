# A RAG system to help exploring the reference architectures of Intelligent Transportation Systems

This system is based on a rag framework and aims to help users exploring ARC-IT and FRAME, two common frameworks for planning and integrating ITS.

## References architectures
- [ARC-IT](https://www.arc-it.net/)
- [FRAME](https://frame-online.eu/)

## The rag framework 
This system leverages both the generation capabilities of an LLM and the tools provided by the two reference architectures.

We divided the system's implementation into three steps :
- Data extraction and indexing : extract all necessary data from the reference architectures and storing them in an index.
- Retrieval : retrieve most query-relevant documents from the index.
- Generation : provide the retrieved context to a LLM so he can base its answer on verified and domain-specific data. 

## The folder structure
We organized this folder into 6 sections :
- templates : were you can find the prompts for the different tasks
- retrieval : it contains the different retrieval methods we implemented and the tools necessary to execute them
- generation : were you can find a summarization method, a function history for the interface and a generation method
- evaluation : we used different metrics (bleurt, bert, meteor and llm as judge) to evaluate the quality of the retrieved context
- examples : demo notebooks to understand the main functions used by the system

<!-- TO DO 
bleurt checkpoint
bert model type
stocker online les db
expliquer différence entre rag.py et main.py -->