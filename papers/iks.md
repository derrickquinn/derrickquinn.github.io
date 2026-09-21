# Accelerating Retrieval-Augmented Generation

Derrick Quinn, Mohammad Nouri, Neel Patel, John Salihu, Alireza Salemi, Sukhan Lee, Hamed Zamani, Mohammad Alian

ASPLOS 2025

[Published paper](https://doi.org/10.1145/3669940.3707264) · [PDF](https://arxiv.org/pdf/2412.15246)

Full-text Markdown converted from the author-provided LaTeX source. Figures are linked images; equations use LaTeX math notation.

## Abstract

An evolving solution to address hallucination and enhance accuracy in large language models (LLMs) is Retrieval-Augmented Generation (RAG), which involves augmenting LLMs with information retrieved from an external knowledge source, such as the web. This paper profiles several RAG execution pipelines and demystifies the complex interplay between their retrieval and generation phases. We demonstrate that while exact retrieval schemes are expensive, they can reduce inference time compared to approximate retrieval variants because an exact retrieval model can send a smaller but more accurate list of documents to the generative model while maintaining the same end-to-end accuracy. This observation motivates the acceleration of the exact nearest neighbor search for RAG.

In this work, we design Intelligent Knowledge Store (IKS), a type-2 CXL device that implements a scale-out near-memory acceleration architecture with a novel cache-coherent interface between the host CPU and near-memory accelerators. IKS offers 13.4–27.9$\times$ faster exact nearest neighbor search over a 512GB vector database compared with executing the search on Intel Sapphire Rapids CPUs. This higher search performance translates to 1.7–26.3$\times$ lower end-to-end inference time for representative RAG applications. IKS is inherently a memory expander; its internal DRAM can be disaggregated and used for other applications running on the server to prevent DRAM – which is the most expensive component in today’s servers – from being stranded.

<a id="sec:intro"></a>

## 1 Introduction

State-of-the-art natural language processing systems heavily rely on large language models (LLMs)–deep Transformer networks \[[1](#ref-transformer)\] with hundreds of millions of parameters. There is much evidence that information presented in the LLM training corpora is “memorized” in the LLM parameters, forming a parametric knowledge base that the model depends on for generating responses. A major challenge with parametric knowledge is its static nature; it cannot be updated unless the model undergoes retraining or fine-tuning, which is an extremely costly process. This creates a critical issue, especially when it comes to non-stationary domains where fresh content is constantly being produced \[[2](#ref-reml)\]. Besides, previous studies have indicated that LLMs exhibit limited memorization for less frequent entities \[[3](#ref-kandpal2022deduplicating)\], are susceptible to hallucinations \[[4](#ref-shuster-etal-2021-retrieval-augmentation)\], and may experience temporal degradation \[[5](#ref-kasai2022realtime)\].

<a id="fig:rag-overview"></a>

![Figure 1. Overview of the Retrieval-Augmented Generation (RAG) pipeline.](assets/iks/figures-introduction-rag-overview.png)

**Figure 1.** Overview of the Retrieval-Augmented Generation (RAG) pipeline.

To overcome the challenges presented by LLMs, a potential solution is to enhance them with non-parametric knowledge, where the LLM is augmented with information retrieved from a knowledge source (e.g., text documents). These approaches have recently gained considerable attention in the machine learning communities \[[4](#ref-shuster-etal-2021-retrieval-augmentation), [6](#ref-10.5555/3495724.3496517), [7](#ref-fid-light), [8](#ref-liu2021retrievalaugmented), [9](#ref-parvez-etal-2021-retrieval-augmented), [10](#ref-salemi2023lamp), [11](#ref-izacard-grave-2021-leveraging)\], and have played key roles in some recent breakthrough applications in the tech industry, such as Google Gemini \[[12](#ref-geminiteam2024gemini)\], Microsoft Copilot \[[13](#ref-azure-rag)\], and OpenAI ChatGPT with Retrieval Plugins \[[14](#ref-gpt-plugins)\]. Retrieval-Augmented Generation (RAG) is the term that is used to refer to systems that adopt this approach in the context of LLMs.

A RAG application includes two key components: a retrieval model and an LLM for text generation, called the generative model. When a query is received, the retrieval model searches for relevant items (e.g., documents) and the top retrieved items, together with the input, are sent to the generative model. Current state-of-the-art retrieval approaches use bi-encoder neural networks (called dense retrieval) \[[15](#ref-dpr)\] for learning optimal embedding for queries and documents. Each item is then encoded into a high-dimension vector (called embedding vectors) and stored in a vector database. Such approaches use K-nearest neighbor algorithms for retrieving the top “K" items from the vector database. Figure [1](#fig:rag-overview) provides an overview of a RAG application. The accuracy of generated output in RAG hinges on the quality of the retrieved item list. Conducting an Exact Nearest Neighbors Search (ENNS) to retrieve the precise top K relevant items involves scanning all the embedding vectors in the vector database, which is costly in today’s memory bandwidth-limited systems. For example, in a RAG application with a 50GB (using a 16-bit floating point representation) vector database running on an Intel Xeon 4416+ with 8$\times$DDR5-4000 memory channels, and a generative model running on an NVIDIA H100 GPU, ENNS takes up to 97% of the end-to-end inference time (§[3](#sec:profiling)).

One strategy to mitigate the retrieval cost is to employ Approximate Nearest Neighbor Search (ANNS), and opt for a faster, but lower-quality search configuration. While lower-quality retrieval can improve search time, our extensive experiments on Question Answering applications demonstrate that a lower-quality search scheme should provide significantly more items to the language model in order to match the end-to-end RAG accuracy of ENNS or a higher-quality, but slower ANNS configuration. This virtually negates any benefits gained during the retrieval phase and even increases the end-to-end inference time.

In this paper, we extensively profile the execution pipeline of RAG, demystifying the complex interplay between various hardware and software configurations in RAG applications \[[16](#ref-Zaharia-2024-Shift)\]. Motivated by the need for high-performance, low-cost, high-quality search and the limitations of current commodity systems, we contribute the Intelligent Knowledge Store (IKS), a cost-optimized, purpose-built CXL memory expander that functions as a high-performance, high-capacity vector database accelerator. IKS offloads memory-intensive dot-product operations in ENNS to a distributed array of low-profile accelerators placed near LPDDR5X DRAM packages.

IKS implements a novel interface atop the CXL.cache protocol to seamlessly offload exact vector database search operations to near-memory accelerators. IKS is exposed as a memory expander that disaggregates its internal DRAM capacity and shares it with vector database applications and other co-running applications through CXL.mem and CXL.cache protocols. Instead of building a full-fledged vector database accelerator, IKS co-designs the hardware and software to implement a minimalist scale-out near-memory accelerator architecture. This design relies on software to map data into the internal IKS DRAM and scratchpads while performing the final top-K aggregation.

In summary, we make the following contributions:

- We demystify RAG by profiling its execution pipeline. We explore various hardware, system, and application-level configurations to assess the performance and accuracy of RAG.

- We demonstrate that RAG requires high-quality retrieval to perform effectively; nonetheless, current RAG applications are bottlenecked by a high-quality retrieval phase.

- We introduce Intelligent Knowledge Store (IKS), which is a specialized CXL-based memory expander equipped with low-profile accelerators for vector database search. IKS leverages CXL.cache to implement a seamless and efficient interface between the CPU and near-memory accelerators.

- We implemented an end-to-end accelerated RAG application using IKS. IKS accelerates ENNS for a 512GB knowledge store by 13.4–27.9$\times$, leading to a 1.7–26.3$\times$ end-to-end inference speedup for representative RAG applications.

<a id="sec:background"></a>

## 2 Background

<a id="sec:background:rag"></a>

### 2.1 Information Retrieval in RAG

Recent advancements in RAG indicate superior outcomes when employing dense retrieval over other methods, for uni-modal \[[6](#ref-10.5555/3495724.3496517), [11](#ref-izacard-grave-2021-leveraging), [17](#ref-izacard2021distilling)\] and multi-modal \[[18](#ref-gui-etal-2022-kat), [19](#ref-10.1145/3539618.3591629), [20](#ref-10.1145/3578337.3605137)\] scenarios. Consequently, our emphasis in this study centers on dense retrieval models, exploring their efficiency-related aspects.

In the context of dense retrieval, a query encoder, denoted as $E_q$, and a document encoder, denoted as $E_d$, are trained to encode queries and documents, respectively, and map them into a high-dimensional vector space. The similarity score between a document[^1] $d$ and a query $q$ is calculated as $s_d = E_q(q) \cdot E_d(d)$, where $E_q(q) \in \mathbb{R}^{h}$, $E_d(d) \in \mathbb{R}^{h}$ and $h$ is the hidden dimension of query and document encoders. Then documents are sorted based on their similarity scores and top documents are retrieved \[[15](#ref-dpr)\]. In a real RAG implementation, in an offline phase, all the documents are encoded into embedding vectors. The embedding vectors are stored in a vector database for dense retrieval. In the paper, we refer to the encoded documents as *embedding vectors* and the vectors generated by the retriever model as *query vectors*.

For dense retrieval, two distinct search algorithms are prevalent: Exact Nearest Neighbor Search (ENNS) and Approximate Nearest Neighbor Search (ANNS). ENNS exhaustively computes the complete pairwise distance matrix between embedding and query vectors. In ANNS, however, strategies such as Product Quantization (PQ) \[[21](#ref-5432202)\], Inverted File with Product Quantization (IVFPQ) \[[22](#ref-6248038)\], and Hierarchical Navigable Small World (HNSW) \[[23](#ref-hnsw)\], are employed to reduce the search space, seeking to trade off a small amount of search accuracy for higher search efficiency.

<a id="applications-of-rag"></a>

### 2.2 Applications of RAG

RAG has proven beneficial for various tasks in natural language processing \[[2](#ref-reml), [24](#ref-Li2022ASO), [25](#ref-Kim-2024-Retrieval)\], including dialogue response generation \[[4](#ref-shuster-etal-2021-retrieval-augmentation), [26](#ref-weston-etal-2018-retrieve), [27](#ref-Wu_Wei_Huang_Wang_Li_Zhou_2019), [28](#ref-cai-etal-2019-skeleton), [29](#ref-9982598), [30](#ref-thulke2021efficient), [31](#ref-Bonetta_2021), [32](#ref-tian-etal-2019-learning), [33](#ref-10.1007/978-3-031-44693-1_2)\], machine translation \[[34](#ref-Gu2017SearchEG), [35](#ref-xu-etal-2020-boosting), [36](#ref-he-etal-2021-fast), [37](#ref-zhang-etal-2018-guiding)\], grounded question answering \[[6](#ref-10.5555/3495724.3496517), [11](#ref-izacard-grave-2021-leveraging), [17](#ref-izacard2021distilling), [38](#ref-erag), [39](#ref-qu-etal-2021-rocketqa), [40](#ref-10.1162/tacl_a_00530), [41](#ref-kilt), [42](#ref-ium), [43](#ref-urag), [44](#ref-10.1145/3626772.3657923)\], abstractive summarization \[[9](#ref-parvez-etal-2021-retrieval-augmented), [45](#ref-peng-etal-2019-text)\], code generation \[[8](#ref-liu2021retrievalaugmented), [46](#ref-10.5555/3327546.3327670)\], paraphrase generation \[[47](#ref-kazemnejad-etal-2020-paraphrase), [48](#ref-su-etal-2021-keep)\], and personalization \[[10](#ref-salemi2023lamp), [49](#ref-rspg), [50](#ref-rag-vs-lora-personalization), [51](#ref-Kumar-2024-LongLaMP)\]. Additionally, RAG’s application extends to multi-modal data tasks like caption generation from images, image generation, and visual question answering \[[18](#ref-gui-etal-2022-kat), [19](#ref-10.1145/3539618.3591629), [52](#ref-10.1145/3549555.3549585), [53](#ref-Fei_2021), [54](#ref-ramos-etal-2023-retrieval), [55](#ref-Chen2022ReImagenRT), [56](#ref-chen-etal-2022-murag)\].

It is noteworthy that commercial LLM systems employing RAG are typically proprietary, and as such, their implementations are not openly accessible. Nevertheless, insights into the implementation of these systems can be gleaned from open-source releases by research labs within commercial entities. We adhere to a methodology akin to the approach outlined by \[[11](#ref-izacard-grave-2021-leveraging)\] and \[[6](#ref-10.5555/3495724.3496517)\], both of which are contributions from Meta AI. Our implementations closely align with the depicted pipeline in Figure [1](#fig:rag-overview). Specifically, we employ a dense document retrieval model as the retriever and leverage a language model for answer generation, consistent with the aforementioned work. Additionally, for efficient vector search capabilities, we utilize the Faiss \[[57](#ref-Jegou-2017-Faiss)\] library, similar to the aforementioned works.

<a id="sec:profiling"></a>

## 3 Demystifying RAG

In this section, we profile the end-to-end execution of three representative long-form question-answering RAG applications and quantify both the execution time breakdown and the generation accuracy of RAG with different hardware and software configurations: `FiDT5`, where we use the T5-based Fusion-in-Decoder \[[11](#ref-izacard-grave-2021-leveraging), [58](#ref-t5)\] as the generative model, as well as `Llama-8B`, and `Llama-70B`, where we use 4-bit-Quantized Llama-3-8B-Instruct and Llama-3-70B-Instruct \[[59](#ref-meta-2024-llama3)\] as the generative models, respectively. The knowledge source for all workloads is Wikipedia, and a trained BERT base (uncased) model is used to generate embedding vectors for documents. We assume 16-bit number format and test with various vector database sizes (corpus size) that store the embedding vectors. The documents themselves are stored in the CPU memory.

In `FiDT5`, the documents are presented via the Fusion-in-Decoder approach, where documents are encoded by the encoder stage of a T5 model, and these encoded representations are combined for the decoder stage. In `Llama-8B` and `Llama-70B`, retrieved documents are presented as plaintext in the prompt. For more information about the methodology, see Section [6](#sec:expr:method).

In the following subsections, we discuss both the accuracy of an end-to-end RAG system and the retrieval model on its own. *Retrieval accuracy* is discussed in terms of recall, where ENNS is considered to be perfect, and the recall score of an ANNS algorithm is the proportion of relevant documents retrieved by both ENNS and ANNS algorithm compared to the total number of relevant documents retrieved by ENNS. *Generation accuracy* refers to how well an end-to-end RAG system answers questions. For details on the evaluation of generation accuracy, see Section [6.2](#sec:expr:software).

<a id="sec:profiling:softwaretuning"></a>

### 3.1 Tuning RAG Software Parameters

Both the retrieval phase and generation phase of the RAG systems that we use offer support for batching of queries in order to improve data reuse and to amortize data movement overheads over several queries. However, batching is not always an option in practice, particularly in the case of latency-critical applications. We consider batch sizes of 1 for latency-critical uses across all applications and 16 for throughput-optimized applications. Batch size does not impact generation accuracy and only affects execution time. An important parameter in RAG is “K” or the *number of documents* retrieved and fed to the language model for text generation. Increasing the document count significantly impacts the generation time. The computation required for transformer inference scales at least linearly with the input size \[[1](#ref-transformer)\], and if we concatenate the retrieved documents, we face significant computation and memory overhead \[[60](#ref-pmlr-v201-duman-keles23a), [61](#ref-zhu-2024-accelerating)\]. In particular, the memory required to store a key-value cache entry for a single token can be computed as follows: $n_{\text{layers}}\times n_{\text{KV-heads}} \times d_{\text{head}} \times n_{\text{bytes}} \times 2$, where $n_{\text{bytes}}$ refers to the size of the number format \[[62](#ref-lienhart)\]. For `Llama-8B` with a 16-bit number format, this is $32\times 8 \times 128\times 2 \times 2 = 131$ kB per token. While exact token counts depend on the tokenization process, each document (for all applications) is 100 words long; for `Llama-8B` and `Llama-70B`, this averaged 127 tokens per document across our evaluation dataset.

<a id="sec:profiling:anns"></a>

### 3.2 Examining Approximate Search for RAG

An important algorithmic consideration that can impact the inference time and generation accuracy of RAG is the choice of retrieval algorithm from the vector database, where we can use exact nearest neighbor search (ENNS) or approximate nearest neighbor search (ANNS). The particular algorithm used for retrieval is implemented by a data structure called an *index*, which stores the embedding vectors computed offline, as described in Section [2.1](#sec:background:rag). For ENNS, an index is a wrapper around an array of embedding vectors sequentially iterated over during the search, but for ANNS, the index can be more complex. For example, HNSW stores embedding vectors in a graph-based data structure \[[23](#ref-hnsw)\].

To evaluate ANNS, we use the state-of-the-art HNSW \[[23](#ref-hnsw)\] ANNS algorithm, and fine-tune the *M* and *efConstruction* parameters to maximize retrieval accuracy while maintaining a reasonable graph, yielding an index with *M* of 32 and *efConstruction* of 128. From this, we evaluate two configurations, ANNS-1 and ANNS-2, which use different *efSearch* parameters: 2048 and 10000. In the context of an end-to-end RAG system, the trade-off of generation accuracy and runtime was evaluated for this index for various choices of *efSearch*. A lower *efSearch* provides higher search throughput, but lower generation accuracy, and a higher *efSearch* provides lower search throughput, but higher generation accuracy. Other HNSW and IVFPQ indexes were tested but provided lower generation accuracy, or similar runtime to ENNS(or even worse, in some cases), negating the benefits of approximation.

<a id="fig:acc-tpt-demystifying"></a>

![Figure 2. Generation accuracy vs. throughput (Queries/sec) of representative RAG applications for various retrieval algorithms and document counts (K). The corpus size is set to 50 GB and batch size to 16.](assets/iks/figures-profiling-acc-tpt-all.png)

**Figure 2.** Generation accuracy vs. throughput (Queries/sec) of representative RAG applications for various retrieval algorithms and document counts (K). The corpus size is set to 50 GB and batch size to 16.

**Generation Accuracy with ANNS vs. ENNS:** Figure [2](#fig:acc-tpt-demystifying) compares the generation accuracy and throughput of ANNS- and ENNS-based RAG applications for `FiDT5`, `Llama-8B`, and `Llama-70B`. The figure illustrates that retrieval quality strongly influences the end-to-end generation accuracy. As shown in Figure [2](#fig:acc-tpt-demystifying), with document count of one, compared to ENNS, the generation accuracy of ANNS-1 and ANNS-2 drops by 22.6 and 34.0% for `FiDT5`, 52.8 and 53.4% for `Llama-8B`, and 51.0 and 51.5% for `Llama-70B`, respectively. With a document count of 16, a similar trend in generation accuracy is observed, with ANNS-1 and ANNS-2 leading to an accuracy reduction of 13.6 and 22% for `FiDT5`, 38.4 and 42.2% for `Llama-8B`, and 38.4 and 45.2% for `Llama-70B`, respectively. Interestingly, the impact of retrieval quality on generation accuracy appears to be even larger when using large models that have not been fine-tuned for this task.

Several prior works \[[63](#ref-ann-benchmarks), [64](#ref-vdm_in_rd)\] demonstrate that hyper-parameter tuning can enhance the retrieval accuracy of ANNS, potentially matching that of ENNS across various workloads. While we optimized our HNSW indexes for accuracy and throughput, these indexes could not match ENNS in end-to-end generation accuracy while achieving significantly (more than $2\times$) faster search. By using a small *efSearch* value, retrieval speed improves significantly, allowing for the use of a larger value of K to compensate for the reduced retrieval quality. However, trading retrieval quality for retrieval speed in this way resulted in lower generation accuracy and end-to-end throughput compared to a larger *efSearch*, where a higher-quality, slower search scheme permits greater accuracy at lower K values (thus reducing generation times). For example, ANNS-2 with 16 documents have 3% higher accuracy and 128% higher throughput compared to ANNS-1 with 128 documents for `FiDT5`. Further improving retrieval quality via exact search gives ENNS-based RAG Pareto-superiority above sufficiently high accuracy thresholds ($\sim43\%$, $\sim27\%$, and $\sim14\%$ for `FiDT5`, `Llama-8B`, and `Llama-70B`, respectively) as demonstrated in Figure [2](#fig:acc-tpt-demystifying). In general, our findings highlight the potential for reducing generation time by leveraging high-quality retrieval methods when high accuracy is required.

**Scaling of ANNS and ENNS:** Previous works \[[23](#ref-hnsw), [63](#ref-ann-benchmarks)\] identified the trade-off between retrieval quality and runtime, and challenges with high-quality ANNS have motivated accelerators such as ANNA \[[65](#ref-anna)\] and NDSearch \[[66](#ref-ndsearch)\]. While lower-quality ANNS algorithms could possibly provide orders of magnitude faster nearest neighbor search compared with ENNS, high-quality ANNS algorithms are shown to provide only a modest speedup \[[67](#ref-ANN_HDD), [68](#ref-comprehensive_ANN)\]. For example, ANNS-2, which is the best performing ANNS configuration in Figure [2](#fig:acc-tpt-demystifying), offers only a 2.5$\times$ speedup compared with ENNS. In fact, all the Pareto frontier configurations that provide high generation accuracy in Figure [2](#fig:acc-tpt-demystifying) are ENNS. Therefore, in the rest of this section, we focus on understanding how to optimize and accelerate RAG applications with ENNS.

<a id="fig:latency"></a>

<a id="fig:latency:corpus"></a>

![Figure 3(a). Sensitivity to corpus size. All configurations use K=16.](assets/iks/figures-motivation-latency-ratio.png)

**Figure 3(a).** Sensitivity to corpus size. All configurations use K=16.
<a id="fig:latency:k"></a>

![Figure 3(b). Sensitivity to K. All configurations use a 50 GB corpus.](assets/iks/figures-motivation-ksensitivity.png)

**Figure 3(b).** Sensitivity to K. All configurations use a 50 GB corpus.

**Figure 3.** Latency breakdown of `FiDT5`, `Llama-8B`, `Llama-70B` for various values of K, corpus sizes. All configurations use batch size 1. Retrieval is ENNS and runs on CPU, generation runs on a single NVIDIA H100 (SXM) for all generative models. The value in each bar shows the absolute retrieval time.
<a id="sec:profiling:accel-gen"></a>

### 3.3 End-to-End RAG Performance with ENNS

In this subsection, we profile time-to-interactive (also known as time to first token) \[[69](#ref-greener_llm)\] for the `FiDT5`, `Llama-8B`, and `Llama-70B` RAG applications and report latency ratios for the retrieval and generation phases. For all experiments, retrieval uses ENNS and runs on the CPU, while generation runs on a single NVIDIA H100 GPU. We select CPU as the baseline for ENNS retrieval, rather than GPU. This decision is made based on the high cost of using GPU memory

As we discussed in Section [3.2](#sec:profiling:anns), the generation accuracy of RAG applications directly depends on the retrieval accuracy. However, as shown in Figure [3(a)](#fig:latency:corpus), utilizing ENNS for retrieval can quickly become an end-to-end bottleneck in RAG applications, even for large models. Although it is possible to compensate for the retrieval accuracy by increasing K (in case of using ANNS), as shown in Figure [3(b)](#fig:latency:k), increasing K would increase the generation time and is costly in terms of time to first token.

The two phases in a RAG pipeline have different characteristics: ENNS is extremely memory bandwidth-bound, and generation is relatively compute-bound. Nevertheless, the current state-of-the-art focus in building AI systems is only on accelerating the generation phase \[[70](#ref-piggyback), [71](#ref-shen2023efficient), [72](#ref-keller202395), [73](#ref-xiao2023efficient), [74](#ref-alizadeh2023llm), [75](#ref-lin2023awq), [76](#ref-xiao2023smoothquant), [77](#ref-yang2023inference), [78](#ref-miao2023specinfer), [79](#ref-gim2023prompt), [80](#ref-patel2024splitwise)\]. Next, we discuss the feasibility of accelerating high-quality nearest neighbor search for future RAG applications.

<a id="tab:retrieval_platforms"></a>

**Table 1.** Speedup of Intel AMX and GPU for ENNS, relative to a CPU baseline. AMX speedup is flat for very small batch sizes, due to the memory-bound nature of similarity search. For 50GB and 512GB corpus size, 1 and 8 H100 GPUs are used, respectively.

| Batch Size  |   1   |   1    |  16   |   16   |
|:-----------:|:-----:|:------:|:-----:|:------:|
| Corpus Size | 50 GB | 512 GB | 50 GB | 512 GB |
|     CPU     |   1   |   1    |   1   |   1    |
|     AMX     | 1.05  |  1.02  | 1.10  |  1.09  |
|     GPU     |  5.2  |  36.9  |  6.0  |  43.7  |

<a id="sec:motivation:enns"></a>

### 3.4 High-Quality Search Acceleration

Given the sensitivity of RAG generation accuracy, latency, and throughput to the retrieval quality, it is imperative to focus exclusively on accelerating the retrieval phase of future RAG applications. In this subsection, we discuss the feasibility of accelerating high-quality ANNS and ENNS.

**Acceleration of High-Quality ANNS:** High-quality ANNS can be as slow as ENNS \[[67](#ref-ANN_HDD)\]. There are prior works aimed at building hardware accelerators for high quality ANNS \[[65](#ref-anna), [66](#ref-ndsearch)\] because GPUs are not effective at accelerating key ANNS algorithms such as IVFPQ and HNSW \[[57](#ref-Jegou-2017-Faiss)\]. Unfortunately, the complex algorithms and memory access patterns used for ANNS algorithms also make ANNS accelerators highly task-specific; for example, ANNA \[[65](#ref-anna)\] and NDSearch \[[66](#ref-ndsearch)\] can only accelerate PQ-based and graph-based ANNS algorithms, respectively. However, our experimental results, which are in line with prior findings \[[68](#ref-comprehensive_ANN)\], show that different corpora are amenable to different ANNS algorithms.

**Acceleration of ENNS:** ENNS can be accelerated using conventional SIMD processors such as GPUs and Intel AMX because the algorithm is simple and data-parallel. Table [1](#tab:retrieval_platforms) compares the speedup of AMX and GPU against a CPU baseline. Although GPUs can significantly speed up ENNS, as the corpus size increases, the cost of offloading ENNS to GPUs increases significantly. For example, to fit the 50 GB and 512 GB corpus sizes tested in Table [1](#tab:retrieval_platforms), we need to use 1 and 8 H100 GPUs, respectively. One of the key contributors to the cost of GPUs is the high-bandwidth memory (HBM) used to implement GPU main memory, which is several times more expensive than DDR or LPDDR-based memories \[[81](#ref-hbm_price)\]. Lastly, GPUs provision huge amounts of compute relative to memory bandwidth[^2], meaning that a large GPU die is poorly utilized by the primarily memory-bound workload of ENNS \[[82](#ref-ibrahim2021analyzing)\].

<a id="sec:profiling:summary"></a>

### 3.5 Summary

The analysis presented in this section, using various software and system configurations for RAG applications, led to the following takeaways:

- Generation accuracy, time to interactive, and throughput of RAG applications can be improved by using a slower but higher-quality retrieval scheme.

- When high-quality retrieval is used, the retrieval phase accounts for a significant portion of end-to-end runtime, regardless of whether the search is performed via ENNS or high-quality ANNS.

- Using GPUs to accelerate ENNS is expensive, and GPUs are not able to accelerate high-quality ANNS effectively or affordably.

- New accelerators for ANNS are highly complex and task-specific due to the unique requirements of ANNS algorithms, while ENNS relies on a very simple scheme, making ENNS simpler to accelerate than ANNS.

<a id="sec:idea"></a>

## 4 Case for Near-Memory ENNS Acceleration

<a id="fig:roofline"></a>

![Figure 4. Roofline model for ENNS using Batch Size 1 and 16. See Section 6 for the experimental setup.](assets/iks/figures-motivation-roofline.png)

**Figure 4.** Roofline model for ENNS using Batch Size 1 and 16. See Section [6](#sec:expr:method) for the experimental setup.

ENNS is characterized by the following features:

- ENNS operations exhibit no data reuse for pair-wise similarity score calculations between corpus vectors and a query vector.

- ENNS operations consist of simple vector-vector dot-products coupled with top-K logic.

- ENNS has a regular and predictable memory access pattern.

- ENNS is highly parallelizable, allowing the corpus to be distributed across different processors with a simple aggregation of top-K similarities at the end.

These features make ENNS a prime candidate for near-memory acceleration due to the following reasons: (1) Deep cache hierarchies are not beneficial for ENNS and can even cause slowdown due to the complex cache maintenance and coherency operations managed by the hardware. This is evident from the roofline model in Figure [4](#fig:roofline) as ENNS running on the CPU cannot saturate the available DRAM bandwidth. (2) The limited data reuse with huge data set size enables low overhead software-managed cache coherency implementation between the host CPU and near-memory accelerators. (3) The regular memory access pattern of ENNS enables coarse-grain virtual to physical address translation on near-memory accelerators. (4) ENNS operations can be efficiently offloaded to a distributed array of near-memory accelerators that each operate in parallel on a shard of corpus data with a low-overhead top-K aggregation phase at the end.

Leveraging these unique features, we design, implement, and evaluate Intelligent Knowledge Store (IKS), a memory expander with a scale-out near-memory acceleration architecture, uniquely designed to accelerate vector database search in future scalable RAG systems. IKS is designed with three requirements in mind: (1) The memory capacity of IKS should be cost-effective and scalable because the size of vector databases for RAG applications is several tens or hundreds of gigabytes and is likely to increase. (2) The near-memory accelerators should be managed in userspace as the cost of context switches and kernel overhead would reduce the benefits of offloads. (3) The near-memory accelerators and host CPU should implement a shared address space; otherwise, explicit data movements between the CPU and near-memory accelerator address spaces will negate the benefits of near-memory offloads; another issue that GPU acceleration of ENNS suffers from. Moreover, a partitioned address space requires rewriting the entire vector database application, as ENNS is just one operation we want to accelerate near the memory, while other data manipulation operations, such as updates, should be managed by the host CPU.

We designed IKS, a type-2 CXL memory expander/accelerator, to meet all these requirements. Our rationale for choosing CXL over DDR-based (or DIMM-based) \[[83](#ref-alian-2018-mcn), [84](#ref-axdimm-aquabolt), [85](#ref-zhou_dimm-link_2023), [86](#ref-patel-xfm-2023)\] near-memory processing architecture is that DIMM-based near-memory processing (1) requires sophisticated mechanisms to share the address space between near-memory accelerators and the host \[[87](#ref-smartdimm)\], (2) limits per-rank memory capacity and compromises the memory capacity of the host CPU when used as an accelerator, and (3) has limited compute and thermal capacity. Instead, IKS relies on asynchronous CXL.mem and CXL.cache protocols to safely share the address space and independently scale the local and far memory capacity of the host CPU, implement a low-overhead interface for offloading from the userspace, and eliminate the limitations on the compute or thermal capacity of the PCIe-attached IKS card. In Section [5](#sec:arch), we explain the architecture of IKS and its interface to the host CPU, and how we used it to accelerate end-to-end RAG applications.

<a id="sec:arch"></a>

## 5 Intelligent Knowledge Store

<a id="sec:arch:overview"></a>

### 5.1 Overview

Figure [5](#fig:iks) provides an overview of the Intelligent Knowledge Store (IKS) architecture. IKS incorporates a scale-out near-memory processing architecture with low-profile accelerators positioned near the memory controllers of LPDDR5X packages. While IKS can function as a regular memory expander, it is specifically designed to accelerate ENNS over the embedding vectors stored in its LPDDR5X packages.

<a id="fig:iks"></a>

<a id="fig:iks:addresspace"></a>

![Figure 5(a). System Address Space with IKS](assets/iks/figures-5-architecture-architecture2.png)

**Figure 5(a).** System Address Space with IKS
<a id="fig:iks:overview"></a>

![Figure 5(b). IKS system integration and architecture overview](assets/iks/figures-5-architecture-architecture1.png)

**Figure 5(b).** IKS system integration and architecture overview
<a id="fig:iks:nma"></a>

![Figure 5(c). NMA internal architecture](assets/iks/figures-5-architecture-architecture3.png)

**Figure 5(c).** NMA internal architecture
<a id="fig:iks:dotproduct"></a>

![Figure 5(d). Dot-product unit and top-K units](assets/iks/figures-5-architecture-architecture4.png)

**Figure 5(d).** Dot-product unit and top-K units

**Figure 5.** (a) IKS internal DRAM, scratchpad spaces, and configuration registers are mapped to the host address space. The scratchpad and configuration register address ranges are labeled as Context Buffers (CB). (b) IKS is a compute-enabled CXL memory expander that includes eight LPDDR5X packages with one near-memory accelerator (NMA) chip near each package. (c) Each NMA includes 64 processing engines. (d) Dot-product units reuse the query vector (QV) dimension across 68 MAC units.

As shown in Figure [5(b)](#fig:iks:overview), IKS utilizes eight LPDDR5X packages, each directly connected to a Near-Memory Accelerator (NMA) that implements both LPDDR5X memory controllers and accelerator logic. Each package contains 512Gb LPDDR5X DRAM with eight 16-bit channels, similar to CXL-PNM \[[88](#ref-park-2024-cxlpnm)\] and MTIA \[[89](#ref-mtia-meta)\]. One of the key differences between IKS and these architectures is the *scale-out* near-memory acceleration architecture. IKS distributes the NMA logic over multiple chips, each providing high-bandwidth and low-energy access to its local LPDDR5X package.

<a id="sec:arch:why_scale_out"></a>

**Why Scale-Out NMA Architecture?** The rationale for such a scale-out NMA architecture is to keep the area of the NMA chip in check. Because memory PHYs are only implemented at the shoreline of a chip \[[89](#ref-mtia-meta), [90](#ref-loh-2015-interposer), [91](#ref-Patel-2024-CXL)\], to implement 64 LPDDR5X memory channels, we need a chip with an approximate perimeter of 160 $mm$. This is because each LPDDR5X channel PHY approximately occupies a shoreline of 2.5 $mm$, based on the die shots of Apple M2 \[[92](#ref-m2-die)\] in 5nm technology. A square-shaped chip with a 160 $mm$ perimeter has an area of 1600 $mm^{2}$, which is larger than the state-of-the-art lithography reticle limit \[[93](#ref-WikiChip-2024-Mask)\]. Although we can technically manufacture such a large accelerator using chiplets, the area of this huge multi-chip module would be wasted, as it is much larger than what is needed to implement the NMA logic, memory controllers, and PCIe/CXL controllers. For context, the area of an H100 GPU is 814 $mm^{2}$.

Splitting the NMAs into smaller chips increases the aggregate chip shoreline and improves yield. Using one NMA per LPDDR5X package requires only eight LPDDR5X memory channels per NMA, necessitating a minimum chip perimeter of 20 $mm$. IKS implements $\times$<!-- -->2 PCIe 5.0 to provide a 8 GBps uplink connecting each NMA to the CXL controller. With this design, the uplinks to the CXL controller are oversubscribed. Nevertheless, this oversubscription is neither a bottleneck for IKS operating in acceleration mode nor for IKS operating in memory expander mode. In acceleration mode, the bandwidth of local LPDDR5X channels is utilized for dot product calculations, and in memory expander mode, the data is interleaved over multiple LPDDR5X packages and read in parallel over the multiple $\times$<!-- -->2 PCIe uplinks.

**IKS is a type 2 CXL device.** IKS’s internal memory is exposed as host-managed device memory where both the CPU and IKS can cache addresses within this unified address space (Figure [5(a)](#fig:iks:addresspace)). IKS leverages the low-latency accesses of CXL.mem and CXL.cache protocols to implement a novel interface between the near-memory accelerators and the CPU that: (1) eliminates the need for DMA setup and buffer management, and (2) eliminates the overhead of interrupt and polling for implementing notifications between the CPU and near-memory accelerators (§[5.3](#sec:arch:cc-interface)).

**IKS supports spatial and coarse-grain temporal multi-tenancy.** In spatial multi-tenancy, the IKS driver partitions embedding vectors that belong to different vector databases across different packages, allowing each NMA to execute ENNS independently per vector database. For temporal multi-tenancy, the IKS driver time-multiplexes similarity search in NMAs among different vector databases that store their embedding vectors in the same LPDDRX5 package. Time multiplexing takes place at the boundary of a complete similarity search.

**Why LPDDR?** For IKS, a customized type-2 CXL device that should support cost-effective high capacity, neither HBM (expensive) nor DDR (general-purpose) are good options. LPDDR DRAM packages are integrated as part of system-on-chip designs, resulting in shorter interconnections, faster clocking, and less power wastage during data transmission. The most recent release of LPDDR, LPDDR5X, offers a bandwidth of 8533 Mbps per pin, exceeding that of DDR5, which provides a bandwidth of 7200 MTps. However, one challenge with using LPDDR in a datacenter setting is reliability, as LPDDR was originally designed for mobile systems. Although we could provision an in-line ECC processing block for error detection and correction, ENNS similarity search is resilient to bit flips, and rare bit flips in ENNS have negligible impact on the end-to-end RAG accuracy.

<a id="sec:arch:offloadmodel"></a>

### 5.2 Offload Model

The IKS address space is shared with the host CPU. The host CPU stores embedding vectors with a specific data layout (that we discuss in Section [5.5](#sec:arch:datalayout)) in contiguous physical addresses in IKS, while the actual documents are stored in the host memory (either in DDR memory or CXL memory). The CPU runs the vector database application, which offloads the similarity calculations (i.e., dot-products between the query vectors and embedding vectors) using `iks_search(query)`, a blocking API that does not require a system call or context switch. After each update operation, the vector database application will flush CPU caches to ensure that when `iks_search(query)` is called, IKS does not contain any stale values.

`iks_search(query)` hides the complexity of interacting with IKS hardware from the programmer by writing an *offload context* to IKS and initiates an offload by writing into a doorbell register. The offload context and doorbells are communicated through memory-mapped regions called *context buffers* to the IKS as shown in Figure [5(a)](#fig:iks:addresspace). An *offload context* includes query vectors, vector dimensions, and the base address of the first embedding vector stored in each LPDDR5X package. The host process then uses `umwait()` to block on the doorbell register (shared between IKS and the host and kept coherent via the CXL.cache protocol) to implement efficient notification between the paused CPU process and near-memory accelerators \[[94](#ref-yuan-rambda)\].

As IKS uses a scale-out near-memory processing architecture (§[5.1](#sec:arch:overview)), the embedding vectors are distributed across different near-memory accelerators’ local DRAM. Therefore, after all the near-memory accelerators complete the offload, the CPU process waiting on `umwait()` will be notified and execute an aggregation routine to construct a single top-K list. This top-K list is then used to retrieve the actual top-K documents from the host memory. The CPU will locate documents based on the physical addresses of the top-K embedding vectors, as the addresses of the embedding vectors stored in IKS are known a priori.

<a id="sec:arch:cc-interface"></a>

### 5.3 Cache Coherent Interface

<a id="fig:iks-trans"></a>

![Figure 6. CPU-IKS interface through cache coherent CXL interconnect.](assets/iks/figures-iks-transactions.png)

**Figure 6.** CPU-IKS interface through cache coherent CXL interconnect.

IKS leverages the cache-coherent interconnect in CXL.cache to implement an efficient interface between near-memory accelerators and host processes through shared memory. Figure [6](#fig:iks-trans) illustrates the transactions through the CXL.cache interface between the host and IKS to initiate and conclude an offload. The host process writes the *offload context* to the predefined *context buffer* address range shared between NMAs and the host CPU (step 1). Note that the context buffer is cacheable, and the CPU uses temporal writes to populate the buffers. Next, the host process writes into a doorbell register, which is mapped to a cache line shared by NMAs. NMAs poll on the doorbell register, and as soon as there is a change, the offload starts (step 4). Once the host updates the doorbell register, it calls `umwait()` to monitor the register for changes from the IKS side.

Before computation in the NMA can start, the NMA reads the offload context from the IKS cache (step 5) and the context written by the host is moved to NMA’s scratchpad. Once the NMA computation is complete, the NMA updates the context buffers with the partial list of similarity scores and physical addresses of the corresponding embedding vectors. Lastly, the NMA writes into the doorbell register, and the host gets notified of the completion of the offload through the `umwait()` mechanism (step 11).

Our experimental results on a two-socket Sapphire Rapids CPU show that communicating the offload context through cache-coherent shared memory provides 1.6$\times$ higher throughput compared with using non-temporal writes that mimic the PCIe MMIO datapath (i.e., CXL.io). Using a cache-coherent interconnect to implement the notification mechanism through the producer/consumer-style doorbell register eliminates the need for expensive interrupt or polling mechanisms.

<a id="sec:arch:nma-arch"></a>

### 5.4 NMA Architecture

As shown in Figure [5(c)](#fig:iks:nma), each NMA implements 64 processing engines to accommodate similarity score calculations for up to 64 query vectors in parallel. Each processing engine includes a query scratchpad, dot-product unit, Top-K unit, and output scratchpad. There is a central control unit in each NMA that generates memory accesses, controls data movement within the NMA, and activates processing engines based on the number of query vectors provided by the host CPU. The network-on-chip implements a fixed broadcast network from DRAM to all the processing engines to reuse data when multiple processing engines are active and evaluate similarity scores against different query vectors.

As shown in Figure [5(d)](#fig:iks:dotproduct), the dot-product unit includes 68 MAC units, each operating at a 1 GHz frequency and providing 68 GFLOPS (16-bit floating point multiply-accumulate operations) compute throughput; therefore saturating the 136 GBps memory bandwidth of the LPDDR5X channels. Each MAC unit evaluates the similarity score between the query (stored in the query scratchpad) and an embedding vector that is read from DRAM in *VD* (Vector Dimension) cycles. All the processing engines operate on the same data that is read from the DRAM; in other words, each processing engine evaluates the similarity score between different query vectors and the same set of embedding vectors. Therefore, for a batch size of one, only one processing engine is utilized, and for a batch size of 64, all the processing engines are utilized. This way, we reuse the embedding vectors that are read from DRAM across different batch sizes.

As illustrated in Figure [5(d)](#fig:iks:dotproduct), within an active dot-product unit, 68 MAC operations are performed in each clock cycle. The first input of the MAC units is dimension $j$ of the query vector in processing engine $PE$ (QV\[PE\]\[j\]), and the second input is dimension $j$ of the embedding vectors $i$ to $i+67$ read from DRAM. As mentioned earlier, it takes *VD* (Vector Dimension) cycles for a dot-product unit to evaluate the similarity score for a block of 68 embedding vectors. Once the similarity score is evaluated, it is loaded into a *score register* (shown in Figure [5(d)](#fig:iks:dotproduct)) in the next clock cycle, and the MAC unit gets busy evaluating a new similarity score for the next 68 embedding vector block. The score registers (68 per processing engine) are then streamed out to the Top-K unit in the next 68 clock cycles.

The Top-K unit maintains an ordered list of the scores by comparing the incoming similarity scores with the head of the ordered list. Figure [5(d)](#fig:iks:dotproduct) illustrates the Top-K unit. If the value of the incoming score is larger, it is ignored; otherwise, it is inserted into the ordered list. Because the vector dimensions are much larger than 68, the serialized insertion into the ordered list is overlapped with the similarity score evaluations and is not on the critical path of the NMA offload.

After all the embedding vectors stored in the DRAM are evaluated, the control unit signals the end of the offload by loading the ordered Top-K list into the output scratchpad and writing to the doorbell register. The host CPU is then notified and can read the content of the output scratchpads through the CXL.cache protocol. Note that both the query scratchpads and the output scratchpads are mapped to the host memory address space. In the current incarnation of the NMA, the size of the query scratchpad (per processing engine) is 2KB, and we keep an ordered list of 32 scores (i.e., we set K to 32 in the hardware).

<a id="fig:datalayout-dram"></a>

![Figure 7. Data layout inside each LPDDR5X package. The host CPU communicates the base address "B", vector dimension "VD", and the number of vectors "N" to the NMAs for each offload. Four embedding vectors (EVs) are highlighted in this layout.](assets/iks/figures-5-architecture-dram-data-layout.png)

**Figure 7.** Data layout inside each LPDDR5X package. The host CPU communicates the base address "*B*", vector dimension "*VD*", and the number of vectors "*N*" to the NMAs for each offload. Four embedding vectors (EVs) are highlighted in this layout.
<a id="fig:datalayout-sp"></a>

![Figure 8. Data layout inside the query scratchpads mapped to host memory address at query scratchpad base address "QS_B". As we increase the batch size, more query scratchpads are populated with distinct query vectors.](assets/iks/figures-5-architecture-qs-data-layout.png)

**Figure 8.** Data layout inside the query scratchpads mapped to host memory address at query scratchpad base address "QS_B". As we increase the batch size, more query scratchpads are populated with distinct query vectors.
<a id="sec:arch:datalayout"></a>

### 5.5 Data Layout Inside DRAM and Query Scratchpad

The host CPU is required to store the embedding vectors in blocks of 68 vectors, laid out in the DRAM as shown in Figure [7](#fig:datalayout-dram). Because each embedding vector dimension is 2 bytes (16-bit floating point), each block is stored in $136 \times VD$ bytes within DRAM, where *VD* is the vector dimension. Within a block, the embedding vectors are stored in column-major order. This layout allows for efficient batching of corpus vectors, as each may be read and processed dimension-by-dimension. Consequently, each NMA will access up to 136 bytes per cycle from the memory controller read queue, comprising one element from 68 distinct embedding vectors.

As discussed in Section [5.3](#sec:arch:cc-interface), the host CPU will fill the query scratchpads with query vectors before an offload starts. The query vectors are stored in sequential addresses within the query scratchpads, as illustrated in Figure [8](#fig:datalayout-sp).

This data layout inside DRAM and query scratchpads simplifies the address generation as well as the network-on-chip architecture of the NMAs. We modified the memory allocation scheme in the vector database application to implement the block data mapping of embedding vectors inside IKS DRAM as shown in Figure [7](#fig:datalayout-dram).

<a id="sec:expr:method"></a>

## 6 Experimental Methodology

<a id="tab:compute"></a>

**Table 2.** Processing Element Options. Memory configuration for Intel AMX is the same as for CPU.

|    Platform    | Parameter |                  Description                  |
|:--------------:|:---------:|:---------------------------------------------:|
|      CPU       | CPU model |     Intel Xeon 4416+ 16 cores @ 2.00 GHz      |
|      CPU       | L1 Cache  |           48 kB dcache, 32kB icache           |
|      CPU       | L2 Cache  |                      2MB                      |
|      CPU       | L3 Cache  |                37.5 MB shared                 |
|      CPU       |    AVX    |    2x AVX-512 FMA units (164 GFlop/s/core)    |
|      CPU       |    OS     |                Ubuntu 22.04.3                 |
|      CPU       |  Kernel   |            Linux 5.15.0-88-generic            |
|      CPU       |  Memory   | 512 GB DDR5-4000 across 8 channels (256 GB/s) |
|      AMX       |     –     |    Intel AMX (BFloat16, 500 GFlop/s/core)     |
| IKS (emulated) |     –     |            1.1 TB/s, 69.9 TFlop/s             |
|      GPU       | GPU Model |   NVIDIA H100 SXM: 3.35 TB/s, 1979 TFlop/s    |

<a id="experimental-setup"></a>

### 6.1 Experimental Setup

To evaluate the performance of the IKS, we developed a simulator (see appendix  [A](#sec:appendix)) and fed ENNS traces into it to obtain the retrieval time of IKS. The simulator is a cycle-approximate performance model that utilizes timing parameters from the RTL synthesis, LPDDR5X access timing, PCIe/CXL timing \[[95](#ref-li-2023-pond), [96](#ref-schuh-2024-cc-nic)\], along with calculations of real software stack overhead (top-K aggregation and `umwait()` overhead). It emulates an IKS as a CXL device running on a remote CPU socket. We implemented the end-to-end RAG application described in Section [3](#sec:profiling) (i.e., `FiDT5`, `Llama-8B`, and `Llama-70B`), including the APIs for distributing queries to the NMA query scratchpad and reducing partial top-32 lists on the CPU. We ran the experiments on two servers equipped with Intel Xeon 4$^{th}$ generation CPUs and one NVIDIA H100 GPU NVIDIA GPUs. The system configuration is shown in Table [2](#tab:compute).

We implemented the RTL design of the Near-Memory Accelerator (NMA) used in IKS and synthesized it using Synopsys Design Compiler targeting TSMC’s 16nm technology node. This process involved collecting key metrics such as area, power, and timing to ensure the design meets the optimal criteria for operation at 1 GHz. For other components, we estimated the area of the memory controllers and PHYs based on die shots from the Apple M2 chip, which utilizes LPDDR5 in a 5nm process \[[97](#ref-Locuza-2022-Die)\]. Since the area scaling of mixed-signal components is negligible \[[98](#ref-su-2017-amd), [99](#ref-horowitz-2014)\], we assumed the same area for the LPDDR5X PHYs and memory controllers when scaling to 16nm technology.

We developed a power model by evaluating the energy consumption of processing operations at the RTL level and incorporating the energy required for data access to scratchpads and LPDDR memory. For example, accessing data in SRAM consumes 39 fJ per bit, while LPDDR memory access requires 4 pJ per bit \[[100](#ref-domain-specific-hw-acc)\]. Since these energy values depend on the underlying technology node, we scaled them to correspond to a 16nm technology node for consistency \[[101](#ref-deepscale)\].

<a id="sec:expr:software"></a>

### 6.2 Software configuration

Google’s Natural Questions (NQ) dataset \[[102](#ref-natq), [103](#ref-latent-retrieval)\] is used for the evaluation of models. Meta’s KILT benchmark \[[104](#ref-Petroni-2020-KILT)\] divides these into training (*nq-train*) and validation (*nq-dev*) datasets. For the retrieval phase, we use a BERT base (uncased) model trained to perform similarity searches between questions and their supporting documents in *nq-train*. The document corpus is constructed as described in \[[15](#ref-dpr)\], and an index is created using Faiss \[[57](#ref-Jegou-2017-Faiss)\] to perform the similarity search[^3]. Across ENNS and ANNS, Faiss is used for index management. The only change made in our evaluation is the use of Intel’s OneMKL BLAS backend for ENNS for all batch sizes, as this provided better performance than the default Faiss search scheme, which uses only BLAS for batch sizes 20 and above.

**`FiDT5` Application:** For testing the accuracy of `FiDT5`, as described in \[[11](#ref-izacard-grave-2021-leveraging)\], the generator is initialized as a pretrained T5-base model (220 million parameters), then fine-tuned to predict answers from question-evidence pairs in the *nq-train* dataset.

To evaluate `FiDT5` on the *nq-dev* dataset, we use the exact match metric \[[105](#ref-rajpurkar2016squad)\], which normalizes answers and compares them against a list of acceptable answers. For `FiDT5`, *generation accuracy* scores refer to the percentage of *nq-dev* questions for which the RAG application generates a correct answer based on this exact match criterion.

**`Llama-8B` and `Llama-70B` Applications:** To evaluate `Llama-8B` and `Llama-70B` on the *nq-dev* dataset, we guide the model via prompting and evaluate *generation accuracy* using a Rouge-L “recall” metric \[[106](#ref-rouge)\], which scores answer predictions based on the proportion of the correct answer that is continuously present in the predicted answer. The model is instructed to give a short answer and to answer only if it is “completely sure.” The prompting approach is used over fine-tuning to reflect an implementation that preserves the generality of the models. However, the downside of this approach is that evaluation is limited by prompt adherence, which is why the “recall” metric is used over precision or F1-Score. When evaluating end-to-end RAG systems, the applications process a batch of queries by first performing retrieval, then generation, before processing the next batch.

<a id="sec:expr"></a>

## 7 Experimental Results

<a id="sec:exp:effectiveness"></a>

### 7.1 Effectiveness and Scalability of IKS Retrieval

<a id="fig:scalability"></a>

![Figure 9. Comparison of ENNS retrieval time for CPU, AMX, GPU (1, 2, 4, and 8 devices), and IKS (1, and 4 devices) for various corpus sizes. The absence of bars in specific GPU and IKS configurations indicates that the corpus exceeds the capacity of the accelerator memory. The Y-axis is in log-scale.](assets/iks/figures-experimental-results-scalability.png)

**Figure 9.** Comparison of ENNS retrieval time for CPU, AMX, GPU (1, 2, 4, and 8 devices), and IKS (1, and 4 devices) for various corpus sizes. The absence of bars in specific GPU and IKS configurations indicates that the corpus exceeds the capacity of the accelerator memory. The Y-axis is in log-scale.
<a id="fig:end-to-end"></a>

<a id="fig:end-to-end:fid"></a>

![Figure 10(a). FiDT5](assets/iks/figures-experimental-results-iks-fid.png)

**Figure 10(a).** `FiDT5`
<a id="fig:end-to-end:mistral"></a>

![Figure 10(b). Llama-8B](assets/iks/figures-experimental-results-iks-8b.png)

**Figure 10(b).** `Llama-8B`
<a id="fig:end-to-end:lamma"></a>

![Figure 10(c). Llama-70B](assets/iks/figures-experimental-results-iks-70b.png)

**Figure 10(c).** `Llama-70B`

**Figure 10.** Inference time breakdown of CPU vs. IKS retrieval for `FiDT5`, `Llama-8B`, and `Llama-70B`. Generative model runs on GPU.

Figure [9](#fig:scalability) compares the performance of IKS with CPU, AMX (idealized, based on speedup for matrix multiplication), and GPU ENNS retrieval. IKS provisions compute and memory bandwidth to balance the pipeline at the maximum batch size of 64; as such, performance is almost flat for batch sizes less than 64. As shown, the purposefully built NMA logic for ENNS enables 1 IKS unit to outperform 1 GPU for a 50 GB corpus for batch sizes 1 and 16 by 2.6$\times$ and 4.6$\times$, respectively. This counterintuitive speedup of IKS over GPUs, which theoretically have both higher FLOPS and memory bandwidth than IKS, is due to two reasons: (1) top-K tracking and aggregation on GPUs is not efficient, while IKS includes specialized Top-K units; and (2) low utilization of the GPU chip translates to limited memory bandwidth usage, as saturating the entire HBM memory bandwidth requires many streaming multiprocessors and tensor cores to issue memory accesses to DRAM in parallel.

To demonstrate the scalability of IKS, we include the retrieval time of multi-GPU and multi-IKS setups. Because each H100 GPU can fit 80 GB of embedding vectors, 8 GPUs can accommodate maximum corpus size of 640 GB. However, with only four IKS devices, we can fit up to a 2 TB corpus size. As shown in Figure [9](#fig:scalability), with additional GPUs and IKS units, the retrieval time for the same corpus size decreases, demonstrating the high data-level parallelism of ENNS and the strong scaling of both GPU and IKS. For example, GPU retrieval time for a 50 GB corpus size reduces by 1.9$\times$, 3.6$\times$, and 6.9$\times$ with 2, 4, and 8 GPU devices, respectively, and IKS retrieval time for a 50 GB corpus size reduces by 1$\times$ and 3.9$\times$ with 1 and 4 IKS units, respectively. Due to the low-overhead IKS-CPU interface, the dominance of similarity search latency in end-to-end ENNS retrieval, and the highly parallelizable nature of ENNS, IKS also provides near-perfect weak scaling. For instance, the retrieval time for a 2 TB corpus on 4 IKS units is only 100$\mu$s longer than for a 512 GB corpus on 1 IKS unit. However, we do not evaluate configurations with more than four IKS units, and the overhead of host-side final top-K aggregation scales as additional units are added. Additionally, we do not evaluate deployments of IKS spanning multiple nodes.

<a id="tab:IKS breakdown"></a>

**Table 3.** Breakdown of ENNS latency on IKS.

|     Corpus Size     |  50 GB   |  50 GB   |  512 GB  |  512 GB  |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|
|     Batch Size      |    1     |    64    |    1     |    64    |
| Write Query Vector  |  0.3 us  |   1 us   |  0.3 us  |   1 us   |
|     Dot-Product     | 45.96 ms | 45.96 ms | 470.6ms  | 470.6 ms |
| Partial Top-32 Read |  0.7 us  | 22.4 us  |  0.7 us  | 22.4 us  |
|  Top-K Aggregation  |  19 us   |  540 us  |  23 us   |  390 us  |
|        Total        | 46.0 ms  | 46.5 ms  | 470.6 ms | 471.0 ms |

Table [3](#tab:IKS breakdown) reports the absolute time breakdown of ENNS retrieval on IKS. We break down the retrieval time of IKS into four components: transfer time of query vectors over the CXL interconnect to the NMAs, time for performing dot-products (both computation and DRAM accesses), updating the top-k score lists in parallel on all NMAs, and time for reducing the partial top-32 lists into a single one on the CPU. The retrieval time of IKS does not change with the value of K (with a maximum K value of 32). This is because IKS always returns 32 top similarity scores, and it is up to the retriever model to pass between 1 to 32 of them to the generative model. As shown in the table, the majority of time is spent on computations and DRAM accesses, and the overhead of initiating offload over the cache-coherent interconnect and aggregating top-K documents on the CPU is negligible.

<a id="end-to-end-performance"></a>

### 7.2 End-to-End Performance

Figure [10](#fig:end-to-end) compares the end-to-end inference time of `FiDT5`, `Llama-8B`, and `Llama-70B` when CPU and IKS are used for ENNS retrieval for various batch sizes, document counts, and corpus sizes. As shown, for large corpus sizes or large batch sizes, the inference time of the RAG applications with CPU retrieval exceeds several seconds, which is not acceptable for user-facing question-answering applications. IKS significantly reduces the ENNS retrieval time for the applications. The end-to-end inference time speedup provided by IKS ranges between 5.6 and 25.6$\times$ for `FiDT5`, between 5.0 and 24.6$\times$ for `Llama-8B`, and between 1.7 and 16.8$\times$ for `Llama-70B` for various batch sizes, corpus sizes, and document counts.

<a id="fig:acc-tpt"></a>

![Figure 11. Comparison of accuracy and throughput of FiDT5, Llama-8B, and Llama-70B for various configurations. ANNS-2 is an HNSW index with M, efConstruction, and efSearch of 32, 128, and 2048, respectively.](assets/iks/figures-experimental-results-acc-tpt.png)

**Figure 11.** Comparison of accuracy and throughput of `FiDT5`, `Llama-8B`, and `Llama-70B` for various configurations. ANNS-2 is an HNSW index with *M*, *efConstruction*, and *efSearch* of 32, 128, and 2048, respectively.

To gain a comprehensive understanding of how the performance and accuracy of RAG applications with IKS acceleration compare across various configurations, Figure [11](#fig:acc-tpt) depicts the queries per second and accuracy of `FiDT5`, `Llama-8B`, and `Llama-70B` implemented using four different configurations: RAG with ENNS running on CPU, RAG with ANNS(two configurations) running on CPU, and RAG with ENNS running on IKS. The generative model runs on the GPU in all these configurations. As illustrated in Figure [11](#fig:acc-tpt), although ANNS-2 configurations exhibit higher throughput compared to ENNS(running on the CPU), their accuracy is lower. For RAG applications that use IKS, retrieval is not a bottleneck, and throughput is significantly improved, even compared to ANNS, as the same generation accuracy can be achieved with smaller values of K (i.e., smaller but more accurate context sent to the generative model).

<a id="power-and-area-analysis"></a>

### 7.3 Power and Area Analysis

The area of each NMA, which contains 64 processing engines, each comprising a dot-product unit, a 2 KB SRAM query scratchpad, a top-K unit, and an output scratchpad, is approximately 3.4 mm$^2$ in the 16nm TSMC technology node. Additionally, 14 mm$^2$ is required for the PHYs and memory controllers. However, the area of the NMA chip is determined by the shoreline because the 21 mm of shoreline required per NMA (20 mm for the LPDDR5X PHYs and 1 mm for PCIe PHYs §[5.1](#sec:arch:overview)) necessitates that the NMA occupy at least 27.56 mm$^2$ in the 16nm technology node. The NMA can be manufactured using older technology nodes to reduce costs and prevent area wastage, as the PHY area (which is mixed-signal) does not scale at the same rate as SRAM and logic \[[98](#ref-su-2017-amd), [99](#ref-horowitz-2014)\].

For a batch size of 1 and vector dimensions of 1024, the processing engines, along with the corresponding query scratchpad accesses, consume approximately 59 $mW$, while accessing embedding vectors from LPDDR memory requires 4.35 $W$. As a result, the total power consumption of IKS for a batch size of 1 is 35.2 $W$. With larger batch sizes, data reuse ensures that the power required for LPDDR access remains constant, but the power consumption of the processing engines increases linearly as more engines are activated to handle the additional workload. For instance, at full utilization with a batch size of 64, the total power consumption increases to 65 $W$.

<a id="sec:expr:gpucost"></a>

### 7.4 Cost and Power Comparison with GPU

IKS utilizes LPDDR5X memory to store embedding vectors. While figures for the cost of LPDDR5X are not yet available, we assume that HBM is more than $3\times$ more expensive than LPDDR \[[81](#ref-hbm_price)\]. Since a single IKS unit includes $6.4\times$ as much onboard memory as a single NVIDIA H100 GPU, the memory cost of IKS is expected to be approximately $2.5\times$ greater than that of a GPU.

For the comparison of compute unit cost, the GPU has a die area of 826 $mm^2$, while the IKS NMAs total a die area of 220 $mm^2$. Because the production cost of a chip increases superlinearly with die area \[[107](#ref-naffziger-2020-amdchiplet)\], an IKS unit (with $5\times$ larger memory capacity) is expected to cost a fraction of a GPU.

<a id="discussion"></a>

## 8 Discussion

IKS provides a cost-effective solution for accelerating ENNS, where the quality of the search is not dataset-dependent. However, if the dataset is amenable to clustering, then the accuracy gap between ENNS and ANNS would reduce, making ANNS more attractive for retrieval. Moreover, IKS is best-suited to RAG applications requiring very high recall, and for datasets difficult to search with existing ANNS schemes with relatively large batch sizes. For example, modern ANNS schemes cannot eliminate more than 99% of the search space for the GloVe dataset \[[68](#ref-comprehensive_ANN)\], so at least 64% of the corpus must be read by an ANNS that does not offer data re-use across queries; in which case the overheads of common ANNS schemes reduces performance to below that of ENNS. However, for datasets that are easier to filter, there is an opportunity for improvement by incorporating approximation techniques into IKS; however, this introduces significant challenges as IKS owes much of its performance to the sequential memory access pattern of ENNS.

One key inefficiency of IKS is that it performs an exhaustive search over the entire corpus, which consumes energy and saturates memory bandwidth. The high internal memory bandwidth utilization of ENNS can cause slowdowns for external accesses by other applications that use IKS as a memory expander, rather than a vector database accelerator. Exploring early termination of similarity search \[[108](#ref-busolin-2024-early), [109](#ref-Li-2020-improving)\] could be a natural solution for reducing the memory bandwidth utilization of ENNS without compromising search accuracy.

Another inefficiency in the current version of IKS is the low NMA chip utilization for batch sizes less than 64. The rationale for overprovisioning NMA compute is that we effectively have free area on the NMA chip. Note that each NMA chip requires eight LPDDR5X memory channels, which demand 20 mm of chip shoreline. Therefore, the minimum NMA chip area is 25 mm$^{2}$ (§[5.1](#sec:arch:why_scale_out)). Thus, the area on NMA is effectively free up to a cap of 25 mm$^{2}$. We chose to utilize this “free” area to overprovision compute so that IKS remains memory-bandwidth bound for all batch sizes below 64. There are opportunities for circuit-level techniques, such as clock and power gating, to power off extra processing engines when the batch size is below 64. Moreover, dynamic voltage and frequency scaling can be used to reduce the frequency and voltage of the NMA chip for batch sizes less than 64, allowing multiple processing engines to perform similarity searches for each query vector.

<a id="related-work"></a>

## 9 Related Work

\[[110](#ref-cxl_accel_mem)\] implement a computational CXL memory solution for near-memory processing and showcased ENNS acceleration inside the CXL memory. However, this work implements CXL memory using DDR DRAM, which does not meet the power and bandwidth requirements for ENNS on large corpus sizes used in RAG. Additionally, our work implements a novel interface between host and near-memory accelerators through CXL.cache. \[[65](#ref-anna)\] and \[[66](#ref-ndsearch)\] present near-data accelerators for PQ- and Graph-based ANNS, respectively. However, we accelerate ENNS because different corpora are amenable to different ANNS algorithms, and the complex algorithms and memory access patterns of such ANNS schemes also make ANNS accelerators highly task-specific. \[[111](#ref-ke-2022-axdimm)\] propose near-memory acceleration of DLRM on Samsung AxDIMM. AxDIMM is based on a DIMM form factor that limits per-rank memory capacity and compromises the memory capacity of the host CPU when used as an accelerator (§[4](#sec:idea)). In contrast, IKS does not strand the internal DRAM space and does not have capacity or compute throughput limitations.

Concurrent with our work, others have also observed that low-quality retrieval can lead to both low-quality and slow generation. Corrective RAG filters out irrelevant documents from the retrieved list before sending them to the LLM \[[112](#ref-yan-2024-corrective)\], while Sparse RAG enables LLMs to use only highly relevant retrieved information \[[61](#ref-zhu-2024-accelerating)\]. In this work, we used ENNS to eliminate the risk of low-quality retrieval and reduce the context size.

<a id="sec:conclusion"></a>

## 10 Conclusion

In this work, we profiled representative RAG applications and showed that the retrieval phase can be an accuracy, latency, and throughput bottleneck, highlighting the importance of an exact, yet high-performance and scalable retrieval scheme for future RAG applications. We designed, implemented, and evaluated the Intelligent Knowledge Store (IKS), a CXL-type-2 device for near-memory acceleration of exact K nearest neighbor search. The key novelty of IKS is the hardware/software co-design that enables a scale-out near-memory processing architecture by leveraging cache-coherent shared memory between the CPU and near-memory accelerators. IKS offers 18-52$\times$ faster exact nearest neighbor search over a 512 GB vector database compared to executing the search on Intel Sapphire Rapids accelerators, leading to 2.0-49$\times$ lower end-to-end RAG inference time.

<a id="acknowledgments"></a>

## Acknowledgments

This work was supported in part by NSF grant numbers 2239020, 1565570, and 2402873, in part by ACE, one of the seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program sponsored by DARPA, in part by the Office of Naval Research contract number N000142412612, and in part by the Center for Intelligent Information Retrieval. Any opinions, findings, conclusions, and recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsors. We thank Jae-sun Seo and Yuan Liao from Cornell University for their help in synthesizing the near-memory accelerators on 16nm TSMC technology.

<a id="markdown-appendix"></a>

<a id="sec:appendix"></a>

## A Artifact Appendix

<a id="abstract"></a>

### A.1 Abstract

This appendix describes two artifacts: 1–The cycle-approximate simulator for IKS, which models IKS using timing data gathered from RTL synthesis. 2–FAISS modified for fast ENNS on Intel CPUs. All artifacts are available via Github.

<a id="artifact-check-list"></a>

### A.2 Artifact check-list

- **Simulator:** <https://github.com/architecture-research-group/iks_simulator>

- **Optimized Faiss:** <https://github.com/architecture-research-group/ae-asplo25-iks-faiss/tree/main>

- **Compilation:** Please refer to each program’s repository.

- **OS requirement:** Modern Linux kernel.

- **Hardware requirement:** Intel® 4th Gen Xeon Scalable Processors or newer, with AMX equipped and enabled.

- **Software requirement:** Intel® MKL Installed.

- **Publicly available?:** Yes.

<a id="bibliography"></a>

## References

<a id="refs"></a>

<a id="ref-transformer"></a>

\[1\] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. (2017). Attention is all you need. *Advances in neural information processing systems*. 30. <https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>.

<a id="ref-reml"></a>

\[2\] H. Zamani, F. Diaz, M. Dehghani, D. Metzler, and M. Bendersky. (2022). Retrieval-enhanced machine learning. *Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval*. pp. 2875–2886. <https://doi.org/10.1145/3477495.3531722>.

<a id="ref-kandpal2022deduplicating"></a>

\[3\] N. Kandpal, E. Wallace, and C. Raffel. (2022). Deduplicating training data mitigates privacy risks in language models. *International conference on machine learning*. pp. 10697–10707. <https://proceedings.mlr.press/v162/kandpal22a.html>.

<a id="ref-shuster-etal-2021-retrieval-augmentation"></a>

\[4\] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston. (2021). Retrieval augmentation reduces hallucination in conversation. *Findings of the association for computational linguistics: EMNLP 2021*. pp. 3784–3803. <https://doi.org/10.18653/v1/2021.findings-emnlp.320>.

<a id="ref-kasai2022realtime"></a>

\[5\] J. Kasai, K. Sakaguchi, Y. Takahashi, R. Le Bras, A. Asai, X. V. Yu, D. Radev, N. A. Smith, Y. Choi, and K. Inui. (2024). REALTIME QA: What’s the answer right now? *Proceedings of the 37th international conference on neural information processing systems*. <https://dl.acm.org/doi/10.5555/3666122.3668252>.

<a id="ref-10.5555/3495724.3496517"></a>

\[6\] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Proceedings of the 34th international conference on neural information processing systems*. <https://doi.org/10.5555/3495724.3496517>.

<a id="ref-fid-light"></a>

\[7\] S. Hofstätter, J. Chen, K. Raman, and H. Zamani. (2023). FiD-light: Efficient and effective retrieval-augmented text generation. *Proceedings of the 46th international ACM SIGIR conference on research and development in information retrieval*. pp. 1437–1447. <https://doi.org/10.1145/3539618.3591687>.

<a id="ref-liu2021retrievalaugmented"></a>

\[8\] S. Liu, Y. Chen, X. Xie, J. K. Siow, and Y. Liu. (2021). Retrieval-augmented generation for code summarization via hybrid GNN. *International conference on learning representations*. <https://openreview.net/forum?id=zv-typ1gPxA>.

<a id="ref-parvez-etal-2021-retrieval-augmented"></a>

\[9\] M. R. Parvez, W. Ahmad, S. Chakraborty, B. Ray, and K.-W. Chang. (2021). Retrieval augmented code generation and summarization. *Findings of the association for computational linguistics: EMNLP 2021*. pp. 2719–2734. <https://doi.org/10.18653/v1/2021.findings-emnlp.232>.

<a id="ref-salemi2023lamp"></a>

\[10\] A. Salemi, S. Mysore, M. Bendersky, and H. Zamani. (2024). LaMP: When large language models meet personalization. *Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers)*. pp. 7370–7392. <https://doi.org/10.18653/v1/2024.acl-long.399>.

<a id="ref-izacard-grave-2021-leveraging"></a>

\[11\] G. Izacard and E. Grave. (2021). Leveraging passage retrieval with generative models for open domain question answering. *Proceedings of the 16th conference of the european chapter of the association for computational linguistics: Main volume*. pp. 874–880. <https://doi.org/10.18653/v1/2021.eacl-main.74>.

<a id="ref-geminiteam2024gemini"></a>

\[12\] G. Team. (2024). Gemini: A family of highly capable multimodal models. *arXiv*. <https://doi.org/10.48550/arXiv.2312.11805>.

<a id="ref-azure-rag"></a>

\[13\] H. Steen and D. Wahlin. (2023). Retrieval augumented generation overview. *Microsoft Learn*. <https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview>.

<a id="ref-gpt-plugins"></a>

\[14\] OpenAI. (2023). ChatGPT plugins. *ChinaFlashMarket*. <https://openai.com/blog/chatgpt-plugins>.

<a id="ref-dpr"></a>

\[15\] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP)*. pp. 6769–6781. <https://doi.org/10.18653/v1/2020.emnlp-main.550>.

<a id="ref-Zaharia-2024-Shift"></a>

\[16\] M. Zaharia, O. Khattab, L. Chen, J. Q. Davis, H. Miller, C. Potts, J. Zou, M. Carbin, J. Frankle, N. Rao, and A. Ghodsi. (2024). The shift from models to compound AI systems. <https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/>.

<a id="ref-izacard2021distilling"></a>

\[17\] G. Izacard and E. Grave. (2021). Distilling knowledge from reader to retriever for question answering. *International conference on learning representations*. <https://openreview.net/forum?id=NTEz-6wysdb>.

<a id="ref-gui-etal-2022-kat"></a>

\[18\] L. Gui, B. Wang, Q. Huang, A. Hauptmann, Y. Bisk, and J. Gao. (2022). KAT: A knowledge augmented transformer for vision-and-language. *Proceedings of the 2022 conference of the north american chapter of the association for computational linguistics: Human language technologies*. pp. 956–968. <https://doi.org/10.18653/v1/2022.naacl-main.70>.

<a id="ref-10.1145/3539618.3591629"></a>

\[19\] A. Salemi, J. Altmayer Pizzorno, and H. Zamani. (2023). A symmetric dual encoding dense retrieval framework for knowledge-intensive visual question answering. *Proceedings of the 46th international ACM SIGIR conference on research and development in information retrieval*. pp. 110–120. <https://doi.org/10.1145/3539618.3591629>.

<a id="ref-10.1145/3578337.3605137"></a>

\[20\] A. Salemi, M. Rafiee, and H. Zamani. (2023). Pre-training multi-modal dense retrievers for outside-knowledge visual question answering. *Proceedings of the 2023 ACM SIGIR international conference on theory of information retrieval*. pp. 169–176. <https://doi.org/10.1145/3578337.3605137>.

<a id="ref-5432202"></a>

\[21\] H. Jégou, M. Douze, and C. Schmid. (2011). Product quantization for nearest neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 33, no. 1, pp. 117–128. <https://doi.org/10.1109/TPAMI.2010.57>.

<a id="ref-6248038"></a>

\[22\] A. Babenko and V. Lempitsky. (2012). The inverted multi-index. *2012 IEEE conference on computer vision and pattern recognition*. pp. 3069–3076. <https://doi.org/10.1109/CVPR.2012.6248038>.

<a id="ref-hnsw"></a>

\[23\] Y. A. Malkov and D. A. Yashunin. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 42, no. 4, pp. 824–836. <https://doi.org/10.1109/TPAMI.2018.2889473>.

<a id="ref-Li2022ASO"></a>

\[24\] H. Li, Y. Su, D. Cai, Y. Wang, and L. Liu. (2022). A survey on retrieval-augmented text generation. *ArXiv*. abs/2202.01110. <https://api.semanticscholar.org/CorpusID:246472929>.

<a id="ref-Kim-2024-Retrieval"></a>

\[25\] T. E. Kim, A. Salemi, A. Drozdov, F. Diaz, and H. Zamani. (2024). Retrieval-enhanced machine learning: Synthesis and opportunities. *arXiv*. <https://doi.org/10.48550/arXiv.2407.12982>.

<a id="ref-weston-etal-2018-retrieve"></a>

\[26\] J. Weston, E. Dinan, and A. Miller. (2018). Retrieve and refine: Improved sequence generation models for dialogue. *Proceedings of the 2018 EMNLP workshop SCAI: The 2nd international workshop on search-oriented conversational AI*. pp. 87–92. <https://doi.org/10.18653/v1/W18-5713>.

<a id="ref-Wu_Wei_Huang_Wang_Li_Zhou_2019"></a>

\[27\] Y. Wu, F. Wei, S. Huang, Y. Wang, Z. Li, and M. Zhou. (2019). Response generation by context-aware prototype editing. *Proceedings of the AAAI Conference on Artificial Intelligence*. 33, no. 01, pp. 7281–7288. <https://doi.org/10.1609/aaai.v33i01.33017281>.

<a id="ref-cai-etal-2019-skeleton"></a>

\[28\] D. Cai, Y. Wang, W. Bi, Z. Tu, X. Liu, W. Lam, and S. Shi. (2019). Skeleton-to-response: Dialogue generation guided by retrieval memory. *Proceedings of the 2019 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers)*. pp. 1219–1228. <https://doi.org/10.18653/v1/N19-1124>.

<a id="ref-9982598"></a>

\[29\] Y. Ahn, S.-G. Lee, J. Shim, and J. Park. (2022). Retrieval-augmented response generation for knowledge-grounded conversation in the wild. *IEEE Access*. 10, pp. 131374–131385. <https://doi.org/10.1109/ACCESS.2022.3228964>.

<a id="ref-thulke2021efficient"></a>

\[30\] D. Thulke, N. Daheim, C. Dugast, and H. Ney. (2021). Efficient retrieval augmented generation from unstructured knowledge for task-oriented dialog. *arXiv preprint arXiv:2102.04643*. <https://doi.org/10.48550/arXiv.2102.04643>.

<a id="ref-Bonetta_2021"></a>

\[31\] G. Bonetta, R. Cancelliere, D. Liu, and P. Vozila. (2021). Retrieval-augmented transformer-XL for close-domain dialog generation. *The International FLAIRS Conference Proceedings*. 34, no. 1. <https://doi.org/10.32473/flairs.v34i1.128369>.

<a id="ref-tian-etal-2019-learning"></a>

\[32\] Z. Tian, W. Bi, X. Li, and N. L. Zhang. (2019). Learning to abstract for memory-augmented conversational response generation. *Proceedings of the 57th annual meeting of the association for computational linguistics*. pp. 3816–3825. <https://doi.org/10.18653/v1/P19-1371>.

<a id="ref-10.1007/978-3-031-44693-1_2"></a>

\[33\] Z. Wang, P. Gong, Y. Zhang, J. Gu, and X. Yang. (2023). Retrieval-augmented knowledge-intensive dialogue. *Natural language processing and chinese computing*. pp. 16–28. <https://doi.org/10.48550/arXiv.2005.11401>.

<a id="ref-Gu2017SearchEG"></a>

\[34\] J. Gu, Y. Wang, K. Cho, and V. O. Li. (2018). Search engine guided neural machine translation. *Proceedings of the AAAI conference on artificial intelligence*. 32. <https://doi.org/10.1609/aaai.v32i1.12013>.

<a id="ref-xu-etal-2020-boosting"></a>

\[35\] J. Xu, J. Crego, and J. Senellart. (2020). Boosting neural machine translation with similar translations. *Proceedings of the 58th annual meeting of the association for computational linguistics*. pp. 1580–1590. <https://doi.org/10.18653/v1/2020.acl-main.144>.

<a id="ref-he-etal-2021-fast"></a>

\[36\] Q. He, G. Huang, Q. Cui, L. Li, and L. Liu. (2021). Fast and accurate neural machine translation with translation memory. *Proceedings of the 59th annual meeting of the association for computational linguistics and the 11th international joint conference on natural language processing (volume 1: Long papers)*. pp. 3170–3180. <https://doi.org/10.18653/v1/2021.acl-long.246>.

<a id="ref-zhang-etal-2018-guiding"></a>

\[37\] J. Zhang, M. Utiyama, E. Sumita, G. Neubig, and S. Nakamura. (2018). Guiding neural machine translation with retrieved translation pieces. *Proceedings of the 2018 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long papers)*. pp. 1325–1335. <https://doi.org/10.18653/v1/N18-1120>.

<a id="ref-erag"></a>

\[38\] A. Salemi and H. Zamani. (2024). Evaluating retrieval quality in retrieval-augmented generation. *Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval*. pp. 2395–2400. <https://doi.org/10.1145/3626772.3657957>.

<a id="ref-qu-etal-2021-rocketqa"></a>

\[39\] Y. Qu, Y. Ding, J. Liu, K. Liu, R. Ren, W. X. Zhao, D. Dong, H. Wu, and H. Wang. (2021). RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. *Proceedings of the 2021 conference of the north american chapter of the association for computational linguistics: Human language technologies*. pp. 5835–5847. <https://doi.org/10.18653/v1/2021.naacl-main.466>.

<a id="ref-10.1162/tacl_a_00530"></a>

\[40\] S. Siriwardhana, R. Weerasekera, E. Wen, T. Kaluarachchi, R. Rana, and S. Nanayakkara. (2023). Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering. *Transactions of the Association for Computational Linguistics*. 11, pp. 1–17. <https://doi.org/10.1162/tacl_a_00530>.

<a id="ref-kilt"></a>

\[41\] F. Petroni, A. Piktus, A. Fan, P. Lewis, M. Yazdani, N. De Cao, J. Thorne, Y. Jernite, V. Karpukhin, J. Maillard, V. Plachouras, T. Rocktäschel, and S. Riedel. (2021). KILT: A benchmark for knowledge intensive language tasks. *Proceedings of the 2021 conference of the north american chapter of the association for computational linguistics: Human language technologies*. pp. 2523–2544. <https://doi.org/10.18653/v1/2021.naacl-main.200>.

<a id="ref-ium"></a>

\[42\] A. Salemi and H. Zamani. (2024). Learning to rank for multiple retrieval-augmented models through iterative utility maximization. <https://arxiv.org/abs/2410.09942>.

<a id="ref-urag"></a>

\[43\] A. Salemi and H. Zamani. (2024). Towards a search engine for machines: Unified ranking for multiple retrieval-augmented large language models. *Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval*. pp. 741–751. <https://doi.org/10.1145/3626772.3657733>.

<a id="ref-10.1145/3626772.3657923"></a>

\[44\] H. Zamani and M. Bendersky. (2024). Stochastic RAG: End-to-end retrieval-augmented generation through expected utility maximization. *Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval*. pp. 2641–2646. <https://doi.org/10.1145/3626772.3657923>.

<a id="ref-peng-etal-2019-text"></a>

\[45\] H. Peng, A. Parikh, M. Faruqui, B. Dhingra, and D. Das. (2019). Text generation with exemplar-based adaptive decoding. *Proceedings of the 2019 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers)*. pp. 2555–2565. <https://doi.org/10.18653/v1/N19-1263>.

<a id="ref-10.5555/3327546.3327670"></a>

\[46\] T. B. Hashimoto, K. Guu, Y. Oren, and P. Liang. (2018). A retrieve-and-edit framework for predicting structured outputs. *Proceedings of the 32nd international conference on neural information processing systems*. pp. 10073–10083. <https://dl.acm.org/doi/10.5555/3327546.3327670>.

<a id="ref-kazemnejad-etal-2020-paraphrase"></a>

\[47\] A. Kazemnejad, M. Salehi, and M. Soleymani Baghshah. (2020). Paraphrase generation by learning how to edit from samples. *Proceedings of the 58th annual meeting of the association for computational linguistics*. pp. 6010–6021. <https://doi.org/10.18653/v1/2020.acl-main.535>.

<a id="ref-su-etal-2021-keep"></a>

\[48\] Y. Su, D. Vandyke, S. Baker, Y. Wang, and N. Collier. (2021). Keep the primary, rewrite the secondary: A two-stage approach for paraphrase generation. *Findings of the association for computational linguistics: ACL-IJCNLP 2021*. pp. 560–569. <https://doi.org/10.18653/v1/2021.findings-acl.50>.

<a id="ref-rspg"></a>

\[49\] A. Salemi, S. Kallumadi, and H. Zamani. (2024). Optimization methods for personalizing large language models through retrieval augmentation. *Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval*. pp. 752–762. <https://doi.org/10.1145/3626772.3657783>.

<a id="ref-rag-vs-lora-personalization"></a>

\[50\] A. Salemi and H. Zamani. (2024). Comparing retrieval-augmentation and parameter-efficient fine-tuning for privacy-preserving personalization of large language models. <https://arxiv.org/abs/2409.09510>.

<a id="ref-Kumar-2024-LongLaMP"></a>

\[51\] I. Kumar, S. Viswanathan, S. Yerra, A. Salemi, R. A. Rossi, F. Dernoncourt, H. Deilamsalehy, X. Chen, R. Zhang, S. Agarwal, N. Lipka, C. V. Nguyen, T. H. Nguyen, and H. Zamani. (2024). LongLaMP: A benchmark for personalized long-form text generation. *arXiv*. <https://doi.org/10.48550/arXiv.2407.11016>.

<a id="ref-10.1145/3549555.3549585"></a>

\[52\] S. Sarto, M. Cornia, L. Baraldi, and R. Cucchiara. (2022). Retrieval-augmented transformer for image captioning. *Proceedings of the 19th international conference on content-based multimedia indexing*. pp. 1–7. <https://doi.org/10.1145/3549555.3549585>.

<a id="ref-Fei_2021"></a>

\[53\] Z. Fei. (2021). Memory-augmented image captioning. *Proceedings of the AAAI Conference on Artificial Intelligence*. 35, no. 2, pp. 1317–1324. <https://doi.org/10.1609/aaai.v35i2.16220>.

<a id="ref-ramos-etal-2023-retrieval"></a>

\[54\] R. Ramos, D. Elliott, and B. Martins. (2023). Retrieval-augmented image captioning. *Proceedings of the 17th conference of the european chapter of the association for computational linguistics*. pp. 3666–3681. <https://doi.org/10.18653/v1/2023.eacl-main.266>.

<a id="ref-Chen2022ReImagenRT"></a>

\[55\] W. Chen, H. Hu, C. Saharia, and W. W. Cohen. (2022). Re-imagen: Retrieval-augmented text-to-image generator. *ArXiv*. abs/2209.14491. <https://api.semanticscholar.org/CorpusID:252596087>.

<a id="ref-chen-etal-2022-murag"></a>

\[56\] W. Chen, H. Hu, X. Chen, P. Verga, and W. Cohen. (2022). MuRAG: Multimodal retrieval-augmented generator for open question answering over images and text. *Proceedings of the 2022 conference on empirical methods in natural language processing*. pp. 5558–5570. <https://doi.org/10.18653/v1/2022.emnlp-main.375>.

<a id="ref-Jegou-2017-Faiss"></a>

\[57\] H. Jégou, M. Douze, and J. Johnson. (2017). Faiss: A library for efficient similarity search. <https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/>.

<a id="ref-t5"></a>

\[58\] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*. 21, no. 1. <https://doi.org/10.5555/3455716.3455856>.

<a id="ref-meta-2024-llama3"></a>

\[59\] M. AI. (2024). Llama 3. <https://llama.meta.com/llama3/>.

<a id="ref-pmlr-v201-duman-keles23a"></a>

\[60\] F. Duman Keles, P. M. Wijewardena, and C. Hegde. (2023). On the computational complexity of self-attention. *Proceedings of the 34th international conference on algorithmic learning theory*. 201, pp. 597–619. <https://proceedings.mlr.press/v201/duman-keles23a.html>.

<a id="ref-zhu-2024-accelerating"></a>

\[61\] Y. Zhu, J.-C. Gu, C. Sikora, H. Ko, Y. Liu, C.-C. Lin, L. Shu, L. Luo, L. Meng, B. Liu, and J. Chen. (2024). Accelerating inference of retrieval-augmented generation via sparse context selection. *arXiv*. <https://doi.org/10.48550/arXiv.2405.16178>.

<a id="ref-lienhart"></a>

\[62\] P. Lienhart. (2024). LLM inference series: 4. KV caching, a deeper look. *Pierre Leinhart (Medium)*. <https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8>.

<a id="ref-ann-benchmarks"></a>

\[63\] M. Aumüller, E. Bernhardsson, and A. Faithfull. (2020). ANN-benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. *Information Systems*. 87, pp. 101374. <https://doi.org/10.48550/arXiv.1807.05614>.

<a id="ref-vdm_in_rd"></a>

\[64\] Y. Zhang, S. Liu, and J. Wang. Are there fundamental limitations in supporting vector data management in relational databases? A case study of PostgreSQL. *Preprint*. <https://www.cs.purdue.edu/homes/csjgwang/pubs/ICDE24_VecDB.pdf>.

<a id="ref-anna"></a>

\[65\] Y. Lee, H. Choi, S. Min, H. Lee, S. Beak, D. Jeong, J. W. Lee, and T. J. Ham. (2022). ANNA: Specialized architecture for approximate nearest neighbor search. *2022 IEEE international symposium on high-performance computer architecture (HPCA)*. pp. 169–183. <https://doi.org/10.1109/HPCA53966.2022.00021>.

<a id="ref-ndsearch"></a>

\[66\] Y. Wang, S. Li, Q. Zheng, L. Song, Z. Li, A. Chang, H. "Helen" Li, and Y. Chen. (2024). NDSEARCH: Accelerating graph-traversal-based approximate nearest neighbor search through near data processing. *Proceedings of the 39th annual international symposium on computer architecture*. <https://doi.org/10.48550/arXiv.2312.03141>.

<a id="ref-ANN_HDD"></a>

\[67\] W. Li, Y. Zhang, Y. Sun, W. Wang, M. Li, W. Zhang, and X. Lin. (2020). Approximate nearest neighbor search on high dimensional data — experiments, analyses, and improvement. *IEEE Transactions on Knowledge and Data Engineering*. 32, no. 8, pp. 1475–1488. <https://doi.org/10.1109/TKDE.2019.2909204>.

<a id="ref-comprehensive_ANN"></a>

\[68\] M. Wang, X. Xu, Q. Yue, and Y. Wang. (2021). A comprehensive survey and experimental comparison of graph-based approximate nearest neighbor search. *Proceedings of the VLDB Endowment*. 14, no. 11, pp. 1964–1978. <https://doi.org/10.14778/3476249.3476255>.

<a id="ref-greener_llm"></a>

\[69\] J. Stojkovic, E. Choukse, C. Zhang, I. Goiri, and J. Torrellas. (2024). Towards greener LLMs: Bringing energy-efficiency to the forefront of LLM inference. *arXiv preprint arXiv:2403.20306*. <https://doi.org/10.48550/arXiv.2403.20306>.

<a id="ref-piggyback"></a>

\[70\] A. Agrawal, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, and R. Ramjee. (2023). SARATHI: Efficient LLM inference by piggybacking decodes with chunked prefills. *Microsoft Research Blog*. <https://doi.org/10.48550/arXiv.2308.16369>.

<a id="ref-shen2023efficient"></a>

\[71\] H. Shen, H. Chang, B. Dong, Y. Luo, and H. Meng. (2023). Efficient llm inference on cpus. *arXiv preprint*. <https://doi.org/10.48550/arXiv.2311.00502>.

<a id="ref-keller202395"></a>

\[72\] B. Keller, R. Venkatesan, S. Dai, S. G. Tell, B. Zimmer, C. Sakr, W. J. Dally, C. T. Gray, and B. Khailany. (2023). A 95.6-TOPS/w deep learning inference accelerator with per-vector scaled 4-bit quantization in 5 nm. *IEEE Journal of Solid-State Circuits*. 58, no. 4, pp. 1129–1141. <https://doi.org/https://doi.org/10.1109/VLSITechnologyandCir46769.2022.9830277>.

<a id="ref-xiao2023efficient"></a>

\[73\] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis. (2024). Efficient streaming language models with attention sinks. *The twelfth international conference on learning representations*. <https://openreview.net/forum?id=NG7sS51zVF>.

<a id="ref-alizadeh2023llm"></a>

\[74\] K. Alizadeh, I. Mirzadeh, D. Belenko, K. Khatamifard, M. Cho, C. C. Del Mundo, M. Rastegari, and M. Farajtabar. (2023). Llm in a flash: Efficient large language model inference with limited memory. *arXiv preprint arXiv:2312.11514*. <https://doi.org/10.48550/arXiv.2312.11514>.

<a id="ref-lin2023awq"></a>

\[75\] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, and S. Han. (2023). Awq: Activation-aware weight quantization for llm compression and acceleration. *arXiv preprint arXiv:2306.00978*. <https://doi.org/10.48550/arXiv.2306.00978>.

<a id="ref-xiao2023smoothquant"></a>

\[76\] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han. (2023). Smoothquant: Accurate and efficient post-training quantization for large language models. *International conference on machine learning*. pp. 38087–38099. <https://doi.org/10.5555/3618408.3619993>.

<a id="ref-yang2023inference"></a>

\[77\] N. Yang, T. Ge, L. Wang, B. Jiao, D. Jiang, L. Yang, R. Majumder, and F. Wei. (2023). Inference with reference: Lossless acceleration of large language models. *arXiv preprint arXiv:2304.04487*. <https://doi.org/10.48550/arXiv.2304.04487>.

<a id="ref-miao2023specinfer"></a>

\[78\] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, C. Shi, Z. Chen, D. Arfeen, R. Abhyankar, and Z. Jia. (2024). SpecInfer: Accelerating large language model serving with tree-based speculative inference and verification. *Proceedings of the 29th ACM international conference on architectural support for programming languages and operating systems, volume 3*. pp. 932–949. <https://doi.org/10.1145/3620666.3651335>.

<a id="ref-gim2023prompt"></a>

\[79\] I. Gim, G. Chen, S. Lee, N. Sarda, A. Khandelwal, and L. Zhong. (2023). Prompt cache: Modular attention reuse for low-latency inference. *arXiv preprint arXiv:2311.04934*. <https://doi.org/10.48550/arXiv.2311.04934>.

<a id="ref-patel2024splitwise"></a>

\[80\] P. Patel, E. Choukse, C. Zhang, A. Shah, I. Goiri, S. Maleki, and R. Bianchini. (2024). Splitwise: Efficient Generative LLM Inference Using Phase Splitting. *2024 ACM/IEEE 51st annual international symposium on computer architecture (ISCA)*. pp. 118–132. <https://doi.org/10.1109/ISCA59077.2024.00019>.

<a id="ref-hbm_price"></a>

\[81\] T. P. Morgan. (2024). He who can pay top dollar for HBM memory controls AI training. *The Next Platform*. <https://www.nextplatform.com/2024/02/27/he-who-can-pay-top-dollar-for-hbm-memory-controls-ai-training/>.

<a id="ref-ibrahim2021analyzing"></a>

\[82\] M. A. Ibrahim, O. Kayiran, Y. Eckert, G. H. Loh, and A. Jog. (2021). Analyzing and leveraging decoupled L1 caches in GPUs. *2021 IEEE international symposium on high-performance computer architecture (HPCA)*. pp. 467–478. <https://doi.org/10.1109/HPCA51647.2021.00047>.

<a id="ref-alian-2018-mcn"></a>

\[83\] M. Alian, S. W. Min, H. Asgharimoghaddam, A. Dhar, D. K. Wang, T. Roewer, A. McPadden, O. O’Halloran, D. Chen, J. Xiong, and others. (2018). Application-transparent near-memory processing architecture with memory channel network. *2018 51st annual IEEE/ACM international symposium on microarchitecture (MICRO)*. pp. 802–814. <https://doi.org/10.1109/MICRO.2018.00070>.

<a id="ref-axdimm-aquabolt"></a>

\[84\] J. H. Kim, S.-H. Kang, S. Lee, H. Kim, Y. Ro, S. Lee, D. Wang, J. Choi, J. So, Y. Cho, J. Song, J. Cho, K. Sohn, and N. S. Kim. (2022). Aquabolt-XL HBM2-PIM, LPDDR5-PIM with in-memory processing, and AXDIMM with acceleration buffer. *IEEE Micro*. 42, no. 3, pp. 20–30. <https://doi.org/10.1109/MM.2022.3164651>.

<a id="ref-zhou_dimm-link_2023"></a>

\[85\] Z. Zhou, C. Li, F. Yang, and G. Sun. (2023). DIMM-Link: Enabling Efficient Inter-DIMM Communication for Near-Memory Processing. *2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*. pp. 302–316. <https://doi.org/10.1109/HPCA56546.2023.10071005>.

<a id="ref-patel-xfm-2023"></a>

\[86\] N. Patel, A. Mamandipoor, D. Quinn, and M. Alian. (2023). XFM: Accelerated software-defined far memory. *Proceedings of the 56th annual IEEE/ACM international symposium on microarchitecture*. pp. 769–783. <https://doi.org/10.1145/3613424.3623776>.

<a id="ref-smartdimm"></a>

\[87\] N. Patel, A. Mamandipoor, M. Nouri, and M. Alian. (2024). SmartDIMM: In-memory acceleration of upper layer protocols. *2024 IEEE international symposium on high-performance computer architecture (HPCA)*. pp. 312–329. <https://doi.org/10.1109/HPCA57654.2024.00032>.

<a id="ref-park-2024-cxlpnm"></a>

\[88\] S.-S. Park, K. Kim, J. So, J. Jung, J. Lee, K. Woo, N. Kim, Y. Lee, H. Kim, Y. Kwon, J. Kim, J. Lee, Y. Cho, Y. Tai, J. Cho, H. Song, J. H. Ahn, and N. S. Kim. (2024). An LPDDR-based CXL-PNM platform for TCO-efficient inference of transformer-based large language models. *2024 IEEE international symposium on high-performance computer architecture (HPCA)*. pp. 970–982. <https://doi.org/10.1109/HPCA57654.2024.00078>.

<a id="ref-mtia-meta"></a>

\[89\] A. Firoozshahian, J. Coburn, R. Levenstein, R. Nattoji, A. Kamath, O. Wu, G. Grewal, H. Aepala, B. Jakka, B. Dreyer, A. Hutchin, U. Diril, K. Nair, E. K. Aredestani, M. Schatz, Y. Hao, R. Komuravelli, K. Ho, S. Abu Asal, J. Shajrawi, K. Quinn, N. Sreedhara, P. Kansal, W. Wei, D. Jayaraman, L. Cheng, P. Chopda, E. Wang, A. Bikumandla, A. Karthik Sengottuvel, K. Thottempudi, A. Narasimha, B. Dodds, C. Gao, J. Zhang, M. Al-Sanabani, A. Zehtabioskuie, J. Fix, H. Yu, R. Li, K. Gondkar, J. Montgomery, M. Tsai, S. Dwarakapuram, S. Desai, N. Avidan, P. Ramani, K. Narayanan, A. Mathews, S. Gopal, M. Naumov, V. Rao, K. Noru, H. Reddy, P. Venkatapuram, and A. Bjorlin. (2023). MTIA: First generation silicon targeting meta’s recommendation systems. *Proceedings of the 50th annual international symposium on computer architecture*. <https://doi.org/10.1145/3579371.3589348>.

<a id="ref-loh-2015-interposer"></a>

\[90\] G. H. Loh, N. E. Jerger, A. Kannan, and Y. Eckert. (2015). Interconnect-memory challenges for multi-chip, silicon interposer systems. *Proceedings of the 2015 international symposium on memory systems*. pp. 3–10. <https://doi.org/10.1145/2818950.2818951>.

<a id="ref-Patel-2024-CXL"></a>

\[91\] D. Patel and J. E. Ontiveros. (2024). CXL is dead in the AI era. <https://www.semianalysis.com/p/cxl-is-dead-in-the-ai-era>.

<a id="ref-m2-die"></a>

\[92\] D. Patel. (2022). Apple M2 Die Shot and Architecture Analysis – Big Cost Increase And A15 Based IP. *SemiAnalysis*. <https://www.semianalysis.com/p/apple-m2-die-shot-and-architecture>.

<a id="ref-WikiChip-2024-Mask"></a>

\[93\] WikiChip. (2024). Mask / reticle. <https://en.wikichip.org/wiki/mask>.

<a id="ref-yuan-rambda"></a>

\[94\] Y. Yuan, J. Huang, Y. Sun, T. Wang, J. Nelson, D. R. K. Ports, Y. Wang, R. Wang, C. Tai, and N. S. Kim. (2023). Rambda: RDMA-driven acceleration framework for memory-intensive µs-scale datacenter applications. *2023 IEEE international symposium on high-performance computer architecture (HPCA)*. pp. 499–515. <https://doi.org/10.1109/HPCA56546.2023.10071127>.

<a id="ref-li-2023-pond"></a>

\[95\] H. Li, D. S. Berger, L. Hsu, D. Ernst, P. Zardoshti, S. Novakovic, M. Shah, S. Rajadnya, S. Lee, I. Agarwal, M. D. Hill, M. Fontoura, and R. Bianchini. (2023). Pond: CXL-based memory pooling systems for cloud platforms. *Proceedings of the 28th ACM international conference on architectural support for programming languages and operating systems, volume 2*. pp. 574–587. <https://doi.org/10.1145/3575693.3578835>.

<a id="ref-schuh-2024-cc-nic"></a>

\[96\] H. N. Schuh, A. Krishnamurthy, D. Culler, H. M. Levy, L. Rizzo, S. Khan, and B. E. Stephens. (2024). CC-NIC: A cache-coherent interface to the NIC. *Proceedings of the 29th ACM international conference on architectural support for programming languages and operating systems, volume 1*. pp. 52–68. <https://doi.org/10.1145/3617232.3624868>.

<a id="ref-Locuza-2022-Die"></a>

\[97\] Locuza. (2022). Die analysis: Samsung exynos 2200 with RDNA2 graphics. <https://locuza.substack.com/p/die-analysis-samsung-exynos-2200>.

<a id="ref-su-2017-amd"></a>

\[98\] L. T. Su, S. Naffziger, and M. Papermaster. (2017). Multi-chip technologies to unleash computing performance gains over the next decade. *2017 IEEE international electron devices meeting (IEDM)*. pp. 1.1.1–1.1.8. <https://doi.org/10.1109/IEDM.2017.8268306>.

<a id="ref-horowitz-2014"></a>

\[99\] M. Horowitz. (2014). 1.1 computing’s energy problem (and what we can do about it). *2014 IEEE international solid-state circuits conference digest of technical papers (ISSCC)*. pp. 10–14. <https://doi.org/10.1109/ISSCC.2014.6757323>.

<a id="ref-domain-specific-hw-acc"></a>

\[100\] W. J. Dally, Y. Turakhia, and S. Han. (2020). Domain-specific hardware accelerators. *Communications of the ACM*. 63, no. 7, pp. 48–57. <https://doi.org/10.1145/3361682>.

<a id="ref-deepscale"></a>

\[101\] S. Sarangi and B. Baas. (2021). DeepScaleTool: A tool for the accurate estimation of technology scaling in the deep-submicron era. *2021 IEEE international symposium on circuits and systems (ISCAS)*. pp. 1–5. <https://doi.org/10.1109/ISCAS51556.2021.9401196>.

<a id="ref-natq"></a>

\[102\] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, M. Kelcey, J. Devlin, K. Lee, K. N. Toutanova, L. Jones, M.-W. Chang, A. Dai, J. Uszkoreit, Q. Le, and S. Petrov. (2019). Natural questions: A benchmark for question answering research. *Transactions of the Association of Computational Linguistics*. <https://doi.org/10.1162/tacl_a_00276>.

<a id="ref-latent-retrieval"></a>

\[103\] K. Lee, M.-W. Chang, and K. Toutanova. (2019). Latent retrieval for weakly supervised open domain question answering. *arXiv preprint arXiv:1906.00300*. <https://doi.org/10.48550/arXiv.1906.00300>.

<a id="ref-Petroni-2020-KILT"></a>

\[104\] F. Petroni, A. Piktus, and A. Fan. (2020). Introducing KILT, a new unified benchmark for knowledge-intensive NLP tasks. <https://ai.meta.com/blog/introducing-kilt-a-new-unified-benchmark-for-knowledge-intensive-nlp-tasks/>.

<a id="ref-rajpurkar2016squad"></a>

\[105\] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *Proceedings of the 2016 conference on empirical methods in natural language processing*. pp. 2383–2392. <https://doi.org/10.18653/v1/D16-1264>.

<a id="ref-rouge"></a>

\[106\] C.-Y. Lin. (2004). ROUGE: A package for automatic evaluation of summaries. *Text summarization branches out*. pp. 74–81. <https://aclanthology.org/W04-1013>.

<a id="ref-naffziger-2020-amdchiplet"></a>

\[107\] S. Naffziger, K. Lepak, M. Paraschou, and M. Subramony. (2020). 2.2 AMD chiplet architecture for high-performance server and desktop products. *2020 IEEE international solid- state circuits conference - (ISSCC)*. pp. 44–45. <https://doi.org/10.1109/ISSCC19947.2020.9063103>.

<a id="ref-busolin-2024-early"></a>

\[108\] F. Busolin, C. Lucchese, F. M. Nardini, S. Orlando, R. Perego, and S. Trani. (2024). Early exit strategies for approximate k-NN search in dense retrieval. *Proceedings of the 33rd ACM international conference on information and knowledge management*. pp. 3647–3652. <https://doi.org/10.1145/3627673.3679903>.

<a id="ref-Li-2020-improving"></a>

\[109\] C. Li, M. Zhang, D. G. Andersen, and Y. He. (2020). Improving approximate nearest neighbor search through learned adaptive early termination. *Proceedings of the 2020 ACM SIGMOD international conference on management of data*. pp. 2539–2554. <https://doi.org/10.1145/3318464.3380600>.

<a id="ref-cxl_accel_mem"></a>

\[110\] J. Sim, S. Ahn, T. Ahn, S. Lee, M. Rhee, J. Kim, K. Shin, D. Moon, E. Kim, and K. Park. (2023). Computational CXL-memory solution for accelerating memory-intensive applications. *IEEE Computer Architecture Letters*. 22, no. 1, pp. 5–8. <https://doi.org/10.1109/LCA.2022.3226482>.

<a id="ref-ke-2022-axdimm"></a>

\[111\] L. Ke, X. Zhang, J. So, J.-G. Lee, S.-H. Kang, S. Lee, S. Han, Y. Cho, J. H. Kim, Y. Kwon, K. Kim, J. Jung, I. Yun, S. J. Park, H. Park, J. Song, J. Cho, K. Sohn, N. S. Kim, and H.-H. S. Lee. (2022). Near-memory processing in action: Accelerating personalized recommendation with AxDIMM. *IEEE Micro*. 42, no. 1, pp. 116–127. <https://doi.org/10.1109/MM.2021.3097700>.

<a id="ref-yan-2024-corrective"></a>

\[112\] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling. (2024). Corrective retrieval augmented generation. *arXiv*. <https://doi.org/10.48550/arXiv.2401.15884>.

[^1]: The term “document” refers to any retrievable item from the knowledge source.

[^2]: NVIDIA H100 80GB provisions  296 Flops/Byte and  592 Int8 Ops/Byte

[^3]: We adapt the Faiss implementation of ENNS by using Intel MKL as the BLAS backend, using BLAS for all batch sizes, and increasing the corpus block size from 1024 to 16384. See appendix [A](#sec:appendix).
