---
license: mit
language:
- en
tags:
- ColBERT
---
<p align="center">
  <img align="center" src="https://github.com/stanford-futuredata/ColBERT/blob/main/docs/images/colbertofficial.png?raw=true" width="430px" />
</p>
<p align="left">


# ColBERT-v1-TripClick

A late-interaction-retriever model trained with [Tripclick Dataset triplets](https://tripdatabase.github.io/tripclick/) (click logs from user interactions in Health search engine) in style of [ColBERTv1](https://arxiv.org/abs/2004.12832). The model is compatible with official [stanford-futuredata ColBERT Repo](https://github.com/stanford-futuredata/ColBERT/tree/main/colbert) and it's derivative model-repo : 
- https://huggingface.co/colbert-ir/colbertv2.0


## How to Use for End-to-End Retrieval

Refer to official ColBERT repo [stanford-futuredata ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/main/colbert) to install the dependencies first.

### 1. Index on your corpus i.e list of documents

```python
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig

def main():
    nbits = 1 
    dataset_name = "test"
    index_name = f'{dataset_name}.{nbits}bits'

    # llm generated ... 
    passages = [
    # Healthcare/Medical (0-9)
    "High blood pressure can lead to heart disease and stroke. Regular exercise and a healthy diet help manage hypertension.",
    "Diabetes is a condition where blood sugar levels are too high. Patients need to monitor glucose and may require insulin therapy.",
    "The flu vaccine is recommended annually to protect against influenza virus. Common side effects include soreness at injection site.",
    "Antibiotics treat bacterial infections but do not work on viruses. Overuse can lead to antibiotic resistance.",
    "Asthma is a chronic lung condition causing difficulty breathing. Inhalers help open airways during asthma attacks.",
    "Regular dental checkups prevent cavities and gum disease. Brushing twice daily and flossing are essential for oral health.",
    "Depression is a mental health disorder affecting mood and daily functioning. Treatment includes therapy and antidepressant medications.",
    "Broken bones require immediate medical attention and x-rays. Casts or splints immobilize the fracture during healing.",
    "Allergies occur when the immune system overreacts to harmless substances. Antihistamines help reduce allergy symptoms.",
    "Heart attack symptoms include chest pain, shortness of breath, and arm pain. Call emergency services immediately if suspected.",
    
    # Finance (10-14)
    "Mortgage rates have increased this quarter affecting home buyers. Fixed rate loans offer stability compared to variable rates.",
    "The stock market showed volatility due to inflation concerns. Investors are moving towards safer bond investments.",
    "Retirement planning should start early to maximize compound interest. 401k contributions reduce taxable income.",
    "Credit scores impact loan approval and interest rates. Paying bills on time improves creditworthiness.",
    "Cryptocurrency markets are highly volatile and risky. Bitcoin and Ethereum dominate the digital currency space.",
    
    # Construction (15-19)
    "Construction workers must wear hard hats and safety boots on site. OSHA regulations require proper fall protection equipment.",
    "Concrete mixing requires the right ratio of cement, sand, and water. Curing time depends on temperature and humidity.",
    "Building permits are required before starting major construction projects. Inspections ensure compliance with local codes.",
    "Excavation work requires careful planning to avoid underground utilities. Gas and electric lines must be marked before digging.",
    "Roofing materials include asphalt shingles, metal panels, and clay tiles. Proper installation prevents water leaks.",
    
    # Technology (20-24)
    "Machine learning algorithms learn patterns from training data. Neural networks are effective for image recognition tasks.",
    "Cloud computing provides scalable storage and processing power. AWS and Azure are leading cloud service providers.",
    "Cybersecurity protects systems from digital attacks and data breaches. Firewalls and encryption secure sensitive information.",
    "Software developers use version control systems like Git. Code reviews improve quality and catch bugs early.",
    "5G networks offer faster speeds and lower latency than 4G. Mobile connectivity continues to improve globally.",
    
    # Food/Cooking (25-29)
    "Baking bread requires yeast, flour, water, and salt. Kneading develops gluten for proper texture.",
    "Grilling vegetables brings out natural sweetness and adds smoky flavor. Brush with olive oil to prevent sticking.",
    "Food safety guidelines recommend cooking chicken to 165 degrees Fahrenheit. Proper temperature kills harmful bacteria.",
    "Mediterranean diet emphasizes fruits, vegetables, and olive oil. Studies show benefits for heart health.",
    "Meal prep saves time during busy weekdays. Cook large batches and portion into containers.",
    
    # Automotive (30-34)
    "Regular oil changes extend engine life and improve performance. Most cars need oil changed every 5000 miles.",
    "Tire pressure should be checked monthly for safety and fuel efficiency. Underinflated tires wear unevenly.",
    "Electric vehicles use battery power instead of gasoline. Charging infrastructure is expanding rapidly.",
    "Brake pads wear down over time and need replacement. Squealing sounds indicate worn brake pads.",
    "Car insurance rates depend on driving record and vehicle type. Comprehensive coverage protects against theft and damage."
    ]


    checkpoint = 'RobinAkan1/colbert-v1-tripclick'
    root = "./experiments" # Default folder created if not passed
    with Run().context(RunConfig(nranks=1, 
                                 root=root,
                                 experiment='notebook' # Experiment Folder inside "root"
                                 )):
        
        ## NOTE : colbert-v1-tripclick was trained with doc_maxlen=400, query_maxlen=32 
        # Because token length were centered around it. And I wanted to save memory during training.
        # If anyone can should raise discussion on how colbert would generalize outside it's training length, 
        # I would appreciate it :)
        config = ColBERTConfig(doc_maxlen=512,
                               query_maxlen=32,
                               # Index Compression params
                               nbits=nbits, 
                               )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=passages, overwrite=True)

if __name__ == '__main__':
    main()
```

### 2. Fetch top-n matching document to user query 
```
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from colbert.data import Queries

if __name__=='__main__':
    nbits = 1
    dataset_name = "test"
    index_name = f'{dataset_name}.{nbits}bits'
    root = "./experiments"
    k = 3 # fetch top 3

    with Run().context(RunConfig(nranks=1, experiment="notebook")):
        config = ColBERTConfig(
                    root=root,
                    doc_maxlen=512,
                    query_maxlen=32,
                    # Index Compression params
                    nbits=nbits, 
                    )
        searcher = Searcher(index=index_name, config=config)
        query = "how to treat high blood pressure"

        import pdb; pdb.set_trace()
        ranking = searcher.search(query, k=k)
        ## returns : ([0, 1, 3], [1, 2, 3], [15.703125, 15.1484375, 14.875]
        ## tuple of tuples of length k containing ((passage_id, passage_rank, passage_score), ...)

```


## ## How to Use as a Re-ranker 
Refer this : https://github.com/liuqi6777/eval_reranker/blob/main/eval_reranker.py

## How to Train this from Scratch (Optional)
( You need to request access to TripClick Dataset first | Author's seemed quite responsive )
- Clone the official [stanford-futuredata ColBERT Repo](https://github.com/stanford-futuredata/ColBERT/tree/main/colbert) and do an editable installation following [these instructions](https://github.com/stanford-futuredata/ColBERT/issues/356#issuecomment-3263277117). It also some addresses some other issues.
- Then, [apply these minimal changes](https://github.com/robinnarsinghranabhat/ColBERT/pull/1) in [`colbertv1`](https://github.com/stanford-futuredata/ColBERT/tree/colbertv1) branch. The PR also contains the training and and evaluation commands used.

## Out of Domain Performance
### 1. Reranking on a Subset of MIRAGE Benchmark
[MIRAGE](https://github.com/Teddy-XiongGZ/MIRAGE) is a collection of 5 medical-question-answering dataset aimed at evaluating Retrieval-Augmented Generation (RAG) systems. [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG/tree/main) implements a baseline RAG system consisting of :
1. **Retrieval System** to fetch top-k documents form a `Corpus`
2. LLM part to ingest these top-k docs to arrive at an answer.

Author's from above repo do a really good job comparing performance of different single-vector embedding models (SVEM) for end-to-end exact retrieval. I extended their work with using colberts and prev. works models in a re-ranker setting. 

( WHY > Because latency matters in real-world. In my few experiments, Exact search of SVEM from pg vector db took 180 secs. Even with Faiss, after loading 70 gigs of vector loaded in RAM, it still took 30 seconds. We are talking about 24 million vectors)

**Extended their Experiment in following setup**
- Benchmark : PubMedQA  (500 questions, 3 Options)
- Corpus : PubMed  (23.9M Docs, Avg. 296 word Length, BioMed Domain )
- Retrieval System fetches 32 documents as context at following  
    - SVEM allowed for end-to-end exact search for retrieval from PubMed Corpus. 
    - ColBERTs as re-rankers on top-2000 BM25 retrieved docs
    - SVEM were also tested in above re-ranking setup
- LLM  : [hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4) (this llm was run across 4 machines each with 1 32 Gig V100 using `vllm`. Details [here](https://gist.github.com/robinnarsinghranabhat/043049fe351aa9d7f7eabfd3475e629a) )


**Summary**

Not surprisig I suppose, tripclick-trained ColBERT performed much better in this setup.

|Retrieval Systems (fetching 32 documents)|Option A (out of 276)|Option B (out of 169)|Option C (out of 55)|Total Correct|Acc.|
| --- | :---: | :---: | :---: | :---: | :---: |
|No-RAG (chain of thought)|247|54|0|301|0.602|
|**colbert-v1-tripclick (ours)**|**258**|**144**|**1**|**403**|**0.806**|
|Contriever|253|132|1|386|0.772|
|MEDCPT as retriever|253|121|3|377|0.754|
|ColBERTv2|250|126|1|377|0.754|
|MEDCPT as reranker|256|129|1|386|0.772|
|Contriever (rerank)|243|126|0|369|0.738|
|answer-ai-colbert-small-v1 (rerank)|250|124|1|375|0.750|

**Code to Replicate**

TODO ...

### 2. Re-ranking on 13 datasets of BEIR Benchmark
I have extended the evalutions from [jina's colbert-v1](https://huggingface.co/jinaai/jina-colbert-v1-en) on couple of colberts out there apart from our own model. `Top-100` BM25 retrieved documents are re-ranked with **NDCG@10** as the metric. Their table is slighly incorrect for **ColBERTv2** and **BM25**. And original **ColBERTv2** is even outperforming  **jina-colbert-v1-en**.

**Code to Replicate**
Following [repo](https://github.com/liuqi6777/eval_reranker/) mentioned [by jina](https://huggingface.co/jinaai/jina-colbert-v1-en#reranking-performance) was used to generate the benchmark outputs. 

**Summary**
Interestingly, 33 million param `answerai-colbert-small-v1` is winning. Also, couldn't benchmark `jinaai/jina-colbert-v2` as it needs flash-attn module, not supported on my V100 machines (TODO :  Try  using Xformers or CPU inference ). To be fair, our `colbert-v1-tripclick` was only trained on 23 million <Query, +Doc, -Doc> pairs in style of ColBERTv1. ColBERTv2 was trained on 400 such triplets with knowledge distillation.

|Dataset|BM25|answerai-colbert-small-v1|ColBERTv2.0|colbert-v1-tripclick (ours)|jina-colbert-v1-en|
| --- | :---: | :---: | :---: | :---: | :---: |
|**BEIR Average**|**41.82**|**52.11**|**50.28**|**24.39**|**50.31**|
|Arguana|29.99|31.91|33.40|20.97|33.97|
|Climate-Fever|16.51|26.83|20.66|9.26|21.88|
|TREC-COVID|59.47|80.61|75.00|47.72|76.94|
|DBPedia|31.80|43.26|42.17|18.61|41.43|
|DL19|50.58|71.46|71.81|30.63|70.38|
|DL20|47.96|66.00|68.09|26.07|67.92|
|FEVER|65.13|86.99|81.06|32.30|83.51|
|FiQA|23.61|38.91|35.61|10.29|36.69|
|HotpotQA|63.30|74.60|68.84|33.87|68.62|
|MSMARCO-dev|22.84|39.02|40.85|10.84|40.56|
|News|39.52|47.61|46.18|21.96|45.29|
|NFCorpus|33.75|37.47|36.69|21.13|36.35|
|NQ|30.55|52.44|51.26|11.85|51.01|
|Quora|78.86|87.28|85.18|67.67|82.76|
|Robust04|40.70|50.79|47.46|21.22|47.70|
|SCIDOCS|14.90|17.92|15.40|8.49|16.67|
|SciFact|67.89|74.66|70.23|34.10|71.0|
|Signal|33.05|32.52|33.23|19.99|30.89|
|Touche2020|44.22|29.74|32.27|16.43|32.42|

### 3. End-to-End Retrieval Performance on BEIR (TODO)

...

### More about ColBERTs
----
* [**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**](https://arxiv.org/abs/2004.12832) (SIGIR'20).
* [**Relevance-guided Supervision for OpenQA with ColBERT**](https://arxiv.org/abs/2007.00814) (TACL'21).
* [**Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval**](https://arxiv.org/abs/2101.00436) (NeurIPS'21).
* [**ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction**](https://arxiv.org/abs/2112.01488) (NAACL'22).
* [**PLAID: An Efficient Engine for Late Interaction Retrieval**](https://arxiv.org/abs/2205.09707) (CIKM'22).

## Future Work 
- Push a clean codebase to replicate re-ranker benchmarks on MIRAGE (current one is a disaster).
- Train on : https://huggingface.co/datasets/sebastian-hofstaetter/tripclick-training
- Train with Knowledge-Distillation with in ColBERTv2 style
- Add relavant Jina Papers, Answer AI papers in refernces
