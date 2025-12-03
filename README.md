# From Clusters to Retrieval: Hybrid BERT-Based Taxonomy and Similarity Search for Medical Chatbot Questions

![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master of Liberal Arts (ALM), Data Science**

## **CSCI E-108 Data Mining, Discovery and Exploration**

## Professor: **Stephen Elston, PhD**, Princeton University, Principal Consultant, Quantia Analytics LLC

## Name: **Dai Phuong Ngo (Liam)**

Here is a full revision of those sections, rewritten around the seven clustering pipelines
v1.1.3, v1.2.3, v1.4.3, v1.2.1.3, v2.2.3, v2.3.3, v2.4.3 and the two similarity pipelines.
I use I and my sometimes, avoid you/your/we/us/our, and avoid bold and special characters.

---

## Abstract

My work compares seven clustering pipelines (v1.1.3, v1.2.3, v1.4.3, v1.2.1.3, v2.2.3, v2.3.3, v2.4.3) and two retrieval and reranking pipelines (Similarity v1 and v2) on an approximately 47.5k-row corpus of short medical chatbot questions with associated tags. The goals are:

1. to build an explainable, business-ready taxonomy for routing and analytics,
2. to understand how far unsupervised structure can go before supervised intent models become necessary, and
3. to design a similarity search and reranking layer that can reuse these representations for FAQ reuse and future RAG-style answering.

All clustering versions now work in semantic embedding spaces that combine BERT sentence embeddings with tag features. The v1.x family starts with BERT(all-MiniLM-L6-v2) plus SVD-reduced tag vectors and KMeans (v1.1.3, v1.2.3), then explores graph-based spectral clustering (v1.4.3) and a transitional PCA-reduced KMeans variant (v1.2.1.3). The v2.x family refines the BERT+tags embedding and shifts to density-based clustering with OPTICS for micro-topics (v2.2.3), followed by a Nyström spectral approximation and MiniBatchKMeans with a k-scan (v2.3.3, v2.4.3) for a scalable, fully covering taxonomy.

On top of the BERT+tags representation, I implement two retrieval pipelines. Similarity Search and Reranking v1 uses dense semantic similarity plus tag Jaccard overlap to rank neighbors. Similarity Search and Reranking v2 extends this with a hybrid architecture that combines semantic cosine, BM25 lexical scores, and tag overlap in a blended scoring function. Across versions, early KMeans runs (v1.1.3, v1.2.3) produce clean, board-friendly top-level categories but suffer from one or two very broad clusters. The density-based variant v2.2.3 uncovers extremely tight topic islands but labels a large portion of the corpus as noise. The Nyström plus MiniBatchKMeans pipelines (v2.3.3, v2.4.3), combined with the hybrid similarity search in v2, strike the best balance between semantic coherence, tag alignment, runtime, coverage, and retrieval quality. I close with an integrated picture of how these clusters and the similarity search stack can support labeling, routing, retrieval, and future RAG-style answering.

---

## Executive Summary

I run seven main clustering variants on the same underlying dataset and tag universe, and then layer two retrieval and reranking pipelines on top of the shared BERT+tags feature space used in the v2 line.

1. v1.1.3 and v1.2.3: BERT + tags with KMeans
   These baselines use POS-filtered text embedded with BERT(all-MiniLM-L6-v2) and SVD-reduced tags. v1.1.3 uses a straightforward KMeans in the fused space. v1.2.3 adds a more systematic k-scan and better metric logging. Both runs already produce a recognizable medical taxonomy (pregnancy and cycles, vaccines, sexual health, musculoskeletal pain, medication questions, mental health, insurance, etc.), but they also create one or two mega-clusters mixing diffuse symptom questions. Internal metrics such as silhouette and intra-cluster cosine are reasonable, but cluster size statistics show heavy imbalance. At this stage, retrieval is still implicit nearest-neighbor search in the same space, not yet a designed reranking pipeline.

2. v1.4.3: BERT + tags with spectral clustering
   In v1.4.3, I keep the BERT+tags representation but impose stricter POS-based noun and proper-noun filtering and move from pure KMeans to a graph-driven spectral clustering approach. A similarity graph in the fused space is approximated, eigenvectors are computed, and KMeans is applied in the spectral coordinates. This makes clusters less spherical and more aligned with the manifold structure. With a k-scan and dendrogram on cluster centroids, v1.4.3 gives a more nuanced top-level view, although it still works in the same underlying embedding and still lacks a dedicated retrieval stack.

3. v1.2.1.3: PCA-reduced BERT + tags with KMeans
   v1.2.1.3 is an intermediate step that reduces the BERT+tags representation via PCA before clustering. The goal is to stabilize KMeans, reduce noise, and improve numerical behavior. Compared to v1.2.3, clusters are somewhat more balanced, and silhouettes improve modestly. This version helps prepare the ground for the more explicitly manifold-aware v2.x line.

4. v2.2.3: BERT + tags with OPTICS
   In v2.2.3, I adopt a fused BERT+tags representation (typically 80-dimensional PCA BERT plus 20-dimensional tag SVD) and apply OPTICS with cosine distance. OPTICS discovers dense, clinically coherent micro-topics such as shingles, hepatitis, thyroid conditions, diabetes, and some medication regimes. However, realistic hyperparameters make OPTICS label only a small fraction of points as core cluster members and treat the remainder as noise. This variant is an upper bound on how sharp clusters can be if coverage is ignored.

5. v2.3.3 and v2.4.3: BERT + tags with Nyström spectral + MiniBatchKMeans
   v2.3.3 and v2.4.3 keep the mature BERT+tags feature design and replace density-based clustering with a scalable Nyström spectral embedding followed by MiniBatchKMeans and a k-scan. Nyström uses a subset of landmarks and a cosine kernel to approximate a spectral embedding in a compact manifold space (for example, 40 dimensions). MiniBatchKMeans then runs over a range of k values, with metrics recorded for each k. v2.3.3 is the first stable Nyström pipeline; v2.4.3 refines the k-scan, metrics, and interpretation steps. The chosen v2.4.3 solution balances global silhouette, tag purity, and cluster size distribution better than earlier versions.

6. Similarity Search and Reranking v1: dense semantic plus tags
   Similarity v1 sits on top of the v2-style BERT+tags representation. It uses dense semantic similarity from BERT embeddings (after PCA and L2 normalization) for candidate generation, then reranks candidates with a blended score combining semantic cosine and tag Jaccard overlap. Text-only queries rely mainly on semantics; tag-driven queries emphasize tag overlap. This design produces a clean, mathematically simple retrieval engine aligned with the clustering geometry.

7. Similarity Search and Reranking v2: hybrid semantic, BM25, and tags
   Similarity v2 extends v1 by introducing a BM25 lexical index over normalized questions. At query time, candidates come from the union of dense semantic neighbors and BM25 neighbors. A blended score then combines semantic cosine, tag Jaccard, and normalized BM25. Query-time POS filtering can be turned off for speed when needed. In practice, Similarity v2 preserves semantic coherence while recovering more exact-phrase and rare-term matches, which matters for short medical queries and potential RAG-style workflows.

From my perspective, v1.1.3, v1.2.3, and v1.4.3 provide increasingly refined KMeans and spectral baselines in a BERT+tags space; v1.2.1.3 stabilizes KMeans via PCA; v2.2.3 explores the limit of density-based micro-topics; and v2.4.3, combined with Similarity v2, offers the most balanced and deployment-ready combination.

---

## Project Overview

This project is clustering-first with an explicit retrieval layer sitting on top. The dataset consists of short, informal medical questions paired with a multi-label tag field. The ultimate aim is a taxonomy and retrieval stack that:

1. makes sense to clinicians and product owners,
2. can be reused for routing, reporting, intent definition, and semantic search, and
3. behaves predictably as new data arrives and distributions drift.

To reach that point, I iterate through seven clustering versions and two retrieval versions. Each iteration adjusts:

1. the NLP pipeline (normalization, POS filtering, BERT embedding, tag handling),
2. the clustering backend (KMeans, spectral clustering, OPTICS, Nyström plus MiniBatchKMeans), and
3. in the v2 retrieval line, the similarity search architecture (dense semantic plus tags in v1, hybrid dense plus BM25 plus tags in v2).

The guiding questions are not only “Does this clustering organize questions into clinically sensible groups?” but also “Given this representation, how should neighboring questions be ranked for retrieval and RAG?”

---

## Problem Statement

Given tens of thousands of short, noisy medical questions and a large, messy tag universe, the problem is to construct:

1. Clusters that
   a) align with real medical themes rather than simple word-frequency artifacts,
   b) remain explainable enough for domain experts to review and label,
   c) handle the long tail of rare conditions and paraphrases without exploding the number of clusters, and
   d) can be recomputed and audited over time as new questions arrive.

2. A similarity search and reranking stack that
   a) reuses the same semantic and tag representations used for clustering,
   b) supports both semantic paraphrase matching and exact lexical matches,
   c) uses tags as a weak supervision signal to enforce intent consistency, and
   d) can be integrated into RAG or FAQ reuse workflows.

There is an inherent tension between simple top-level clusters for dashboards and KPIs, fine-grained micro-topics for intent libraries, and a flexible retrieval layer that can work within or across these clusters.

In this project:

* The v1 line (v1.1.3, v1.2.3, v1.4.3, v1.2.1.3) leans toward top-level and mid-level clusters in a BERT+tags space using KMeans and spectral clustering.
* v2.2.3 pushes density-based clustering for micro-topics using OPTICS.
* v2.3.3 and v2.4.3 aim for a middle ground via Nyström spectral plus MiniBatchKMeans.
* Similarity v1 and v2 explore how to expose this structure to a chatbot or RAG system via semantic and hybrid retrieval.

---

## Data Exploration

After preprocessing and POS-based cleanup, the dataset contains about 47,491 rows. A TF-IDF vocabulary (when used) reaches on the order of tens of thousands of terms, and the multi-hot tag matrix covers roughly several thousand distinct tags.

The distributions are highly long-tailed:

* Common tags such as pregnancy, period, pain, anxiety, depression, flu, and insurance appear frequently.
* Many tags appear only a handful of times.

Cosine similarity histograms in both LSA-inspired spaces (for the early experiments) and in the BERT+tags spaces show two distinct structures:

1. Dense islands representing specific conditions or topics, such as shingles, hepatitis C, hypothyroidism, diabetes, certain drug regimens, or narrow anatomic complaints.
2. A broad low-similarity background of vague symptom questions, general worry, and “could this be X” prompts.

This explains the behavior of later clustering methods:

* In KMeans-based versions such as v1.1.3 and v1.2.3, one or two large generic clusters emerge alongside a set of more targeted clusters.
* In the density-based v2.2.3, the dense islands become clusters and the background turns into noise.
* In Nyström-based versions such as v2.3.3 and v2.4.3, these structures are preserved but partitioned more evenly.

Outliers that become zero vectors or get fully filtered by POS and tag cleaning tend to be extremely short prompts like “hi” or “help please,” which are naturally problematic for both clustering and retrieval. These are treated as edge cases and either dropped or relegated to catch-all groups.

---

## Modelling

Clustering and retrieval are treated as two sides of the same organizing problem: clusters give global structure, and similarity search lets a system zoom into local neighborhoods.

v1 line (v1.1.3, v1.2.3, v1.4.3, v1.2.1.3)
In the v1 line, text is normalized, POS-filtered, and embedded with BERT. Tags are parsed, normalized, and projected with SVD. These are combined into a fused semantic plus tag feature space. KMeans (v1.1.3, v1.2.3, v1.2.1.3) and spectral clustering (v1.4.3) are then used to produce clusters at different granularities. Retrieval is still informal: nearest neighbors in the same space can be computed, but there is no separate retrieval design.

Transition through v1.4.3 and v1.2.1.3
v1.4.3 uses spectral clustering to capture non-spherical manifold structures. v1.2.1.3 uses PCA to stabilize KMeans and reduce noise before clustering. These experiments show how sensitive structure can be to the geometry imposed on the same BERT+tags representation.

v2 line (v2.2.3, v2.3.3, v2.4.3) and retrieval
In the v2 line, the fused representation is formalized:

* POS-filtered questions receive BERT all-MiniLM-L6-v2 embeddings.
* BERT vectors are reduced via PCA to a semantic subspace.
* Tag vectors are compressed via SVD to a low-dimensional tag space.
* These are concatenated and normalized.

Within this representation:

* v2.2.3 uses OPTICS to reveal high-purity topic islands at the cost of labeling many points as noise.
* v2.3.3 and v2.4.3 use Nyström spectral approximation plus MiniBatchKMeans with a k-scan to produce fully covering, scalable clusterings.

Parallel to that, Similarity v1 and v2 are added:

* Similarity v1 uses dense semantic similarity and tag Jaccard to rerank neighbors.
* Similarity v2 adds BM25 lexical signals and a union-based candidate generation strategy.

I rely on internal metrics, tag coherence, and qualitative retrieval audits to judge which combinations work best for routing, long-tail discovery, and per-query retrieval quality.

---

## Algorithm and Evaluation Strategy

Each clustering version follows a similar macro pattern:

1. Build a feature space that combines content (question text) and weak supervision (tags).
2. Reduce dimensionality if necessary.
3. Run a clustering algorithm suited to that geometry.
4. Evaluate the resulting structure both numerically and qualitatively.

Similarity v1 and v2 follow a parallel pattern:

1. Reuse shared semantic and tag representations from clustering.
2. Add lexical features (BM25 in v2).
3. Define candidate generation (semantic, lexical, or both).
4. Design a blended scoring function for reranking.
5. Evaluate results qualitatively and prepare for future IR metrics.

v1 family

* v1.1.3 and v1.2.3: BERT+tags fused space, KMeans with different k scanning and hyperparameters.
* v1.4.3: spectral clustering on a similarity graph derived from the same fused space, followed by KMeans in eigenvector coordinates.
* v1.2.1.3: PCA on the fused space followed by KMeans.

v2 family

* v2.2.3: OPTICS with cosine distance in a normalized 100-dimensional fused space, tuned for dense island discovery.
* v2.3.3 and v2.4.3: Nyström spectral approximation to a compact manifold space, then MiniBatchKMeans with a k-scan and metric tracking.

Similarity v1

* BERT embeddings (PCA-reduced) form the semantic tower.
* Tag sets form a discrete tower.
* Candidates come from dense semantic nearest neighbors.
* Reranking uses a blended score combining semantic cosine and tag Jaccard.

Similarity v2

* Same semantic and tag components as v1.
* Additional BM25 index over normalized question text.
* Candidates come from the union of semantic and BM25 top lists.
* Reranking uses a three-term blended score combining semantic cosine, tag Jaccard, and normalized BM25.

---

## Data Preprocessing

All versions share a common preprocessing backbone, with later variants adding more sophisticated NLP components that are reused by both clustering and retrieval.

Normalization

* Lowercase text.
* Normalize curly quotes and dashes.
* Remove most non-alphanumeric symbols, except where apostrophes or question marks matter.
* Store as short_question_norm for any lexical features such as TF-IDF or BM25.

Tag parsing

* Parse raw tag strings into normalized lists.
* Lowercase and strip whitespace.
* Convert to a multi-label binary matrix using a multi-label binarizer.
* Keep a tag_sets view per row for Jaccard computations.

POS-based filtering

* Use a spaCy pipeline with NER and textcat disabled for speed.
* Keep nouns and proper nouns, plus short domain-specific abbreviations such as hpv, uti, rbc, wbc, and similar.
* Remove general and domain-specific stopwords, numeric tokens, and very short tokens outside the whitelist.
* Store the result as short_question_pos.
* Drop rows where both short_question_pos is empty and the tag list is empty.

Dimensionality reduction

* v1 line:

  * Earlier experiments used TF-IDF and LSA; current v1.x versions rely mainly on BERT embeddings plus tag SVD, with PCA added in v1.2.1.3.
* v2 line:

  * BERT embeddings on short_question_pos.
  * PCA to an 80-dimensional semantic subspace, preserving a substantial proportion of variance.
  * Tag SVD to a roughly 20-dimensional tag subspace.
  * Concatenate to form a 100-dimensional vector for clustering.

Scaling and normalization

* Apply a standard scaler where needed.
* Apply L2 normalization before algorithms such as OPTICS and Nyström so that cosine distance is meaningful.
* For similarity search, ensure that semantic vectors are L2 normalized so that dot products equal cosine similarity.

---

## Model Architectures

Phase 1 – v1 family

* Text representation:

  * Normalized and POS-filtered text embedded with BERT all-MiniLM-L6-v2.
  * Tag vectors reduced via SVD and concatenated with BERT embeddings.
  * In v1.2.1.3, this fused representation is further reduced via PCA.

* Clustering:

  * v1.1.3 and v1.2.3: KMeans or MiniBatchKMeans in the fused space.
  * v1.4.3: spectral clustering over a similarity graph of the fused space, using KMeans in the eigenvector space.
  * v1.2.1.3: PCA followed by KMeans to improve stability.

Phase 2 – v2 family and retrieval

* Text representation:

  * POS-filtered questions embedded with BERT all-MiniLM-L6-v2.
  * PCA to an 80-dimensional semantic space.

* Tag representation:

  * Multi-label tag matrix reduced by SVD to about 20 dimensions.

* Clustering:

  * Fused 100-dimensional vectors created by concatenation of semantic and tag representations.
  * v2.2.3: OPTICS with cosine distance on L2-normalized fused vectors.
  * v2.3.3 and v2.4.3: Nyström spectral approximation to a compact space, then MiniBatchKMeans with k-scan.

* Retrieval:

  * Semantic tower: L2-normalized 80-dimensional BERT PCA vectors.
  * Tag tower: tag_sets per question.
  * Similarity v1: semantic cosine plus tag Jaccard, candidates from semantic neighbors.
  * Similarity v2: same plus BM25 over short_question_norm, with candidates from both semantic and BM25 neighbors.

This separation allows clustering and retrieval to remain loosely coupled but conceptually aligned.

---

## Training Configuration and Reproducibility

Even though the clustering runs are unsupervised, I treat configuration and runtime behavior as if they were training processes that require careful control and documentation.

Key choices include:

* Fixed random seeds for NumPy and clustering algorithms, for example 42.
* Reasonable bounds on any TF-IDF-like vocabulary and minimum document frequency when those features are used.
* Batched BERT embedding on GPU when available.
* PCA and SVD component counts chosen to balance explained variance and dimensionality.
* KMeans and MiniBatchKMeans configured with multiple initializations and sensible batch sizes.
* OPTICS parameters tuned in v2.2.3 for discovery of dense islands without over-fragmentation.
* Nyström landmark sampling sizes chosen to balance approximation quality and speed.
* Detailed timing logs for CSV loading, NLP preprocessing, vectorization, spectral embedding, clustering, metric computation, UMAP, and dendrogram generation.

For similarity stacks, I also log preprocessing time, embedding and PCA time, SVD time, BM25 index-building time, and save offline timing summaries so that Similarity v1 and v2 can be compared.

---

## Evaluation Strategy

Because different versions operate in different geometries, evaluation is layered rather than reduced to a single scalar score.

For clustering versions:

* Internal metrics:

  * Silhouette (often with cosine, sometimes subsampled).
  * Calinski–Harabasz index.
  * Davies–Bouldin index.

* Size and coverage:

  * Cluster size distribution and imbalance.
  * For OPTICS in v2.2.3: the ratio of clustered points to noise.

* Cosine structure:

  * Average intra-cluster cosine similarity.
  * Mean inter-centroid cosine similarity.

* Tag coherence:

  * Dominant tag purity per cluster.
  * Tag entropy per cluster.
  * Mean intra-cluster Jaccard similarity between tag sets.

* Human inspection:

  * Top terms per cluster from a TF-IDF fit on the cluster.
  * Top tags per cluster.
  * Representative example questions.

For similarity versions:

* Qualitative neighbor inspection:

  * Evaluate neighbors for text-only, tag-driven, and mixed queries.
  * Compare neighbor lists from Similarity v1 and v2 on the same queries.

* Score behavior:

  * Inspect ranges and distributions of semantic cosine, tag Jaccard, BM25 scores, and blended scores.
  * Check that BM25 in Similarity v2 can promote exact or rare lexical matches without destroying semantic structure.

A natural next step for future work is to collect a small set of query–relevance pairs and compute Precision at k, Recall at k, and nDCG at k to evaluate retrieval versions more formally.

---

## Processing Pipeline

All experiments share the same backbone; clustering versions and similarity stacks plug into different stages.

1. Load and clean

   * Read CSV from storage.
   * Normalize short_question, parse and clean tags.
   * Apply POS filtering to create short_question_pos.
   * Drop rows with no informative text and no tags.

2. Feature construction

   * v1 line:

     * BERT embeddings on short_question or short_question_pos.
     * Tag vectors reduced via SVD.
     * Fusion by concatenation, with PCA added in v1.2.1.3.

   * v2 line:

     * BERT embeddings on short_question_pos.
     * PCA to 80-dimensional semantic space.
     * Tag SVD to about 20 dimensions.
     * Concatenate to 100-dimensional fused vectors.
     * Build BM25 index on short_question_norm for Similarity v2.

3. Clustering

   * v1.1.3 and v1.2.3: KMeans in fused space with different k strategies.
   * v1.4.3: spectral clustering plus KMeans in eigenvector space.
   * v1.2.1.3: PCA plus KMeans.
   * v2.2.3: OPTICS with cosine distance.
   * v2.3.3 and v2.4.3: Nyström spectral embedding plus MiniBatchKMeans, with k-scan and metric logging.

4. Similarity search

   * Similarity v1:

     * Candidates from dense semantic neighbors.
     * Reranking with semantic cosine plus tag Jaccard.

   * Similarity v2:

     * Candidates from semantic nearest neighbors and BM25 neighbors.
     * Reranking with a blended semantic plus tags plus BM25 score.

5. Diagnostics and outputs

   * Internal metrics, UMAP plots, dendrograms, and k-versus-metric plots for clustering.
   * Neighbor tables with decomposed scores for retrieval.
   * Cluster assignment CSVs, cluster summaries, model-selection tables, and timings JSON.

---

## Conclusion

Looking across v1.1.3, v1.2.3, v1.4.3, v1.2.1.3, v2.2.3, v2.3.3, v2.4.3 and the two similarity pipelines:

* v1.1.3 and v1.2.3 deliver reliable top-level and mid-level taxonomies in a BERT+tags space, using KMeans. They are strong for initial stakeholder conversations, but mega-clusters limit their use for fine-grained routing.
* v1.4.3 shows that spectral clustering on the same embedding sharpens themes and reveals hierarchical relations, but still relies on the same representation and does not yet add a formal retrieval layer.
* v1.2.1.3 demonstrates that PCA smoothing can stabilize KMeans and slightly improve metrics and cluster balance.
* v2.2.3 uses OPTICS to push density-based clustering as far as possible, yielding highly coherent islands at the cost of calling many points noise.
* v2.3.3 and especially v2.4.3 use Nyström spectral plus MiniBatchKMeans to build a scalable manifold-based clustering with full coverage, balanced cluster sizes, and a good trade-off between silhouette, tag purity, and interpretability.
* Similarity v1 translates the fused representation into a clean dense and tags retrieval engine that works well but can miss some exact lexical effects.
* Similarity v2 adds BM25 and a hybrid candidate strategy so that semantic paraphrase matching, rare-word anchoring, and tag alignment all contribute.

If a single clustering version and a single retrieval version must be chosen for deployment, v2.4.3 for clustering and Similarity Search and Reranking v2 for retrieval form a natural pair. Together, they provide a global taxonomy and a flexible similarity layer that balance coherence, tag alignment, runtime, coverage, and explainability in a way that feels suitable for real medical chatbot or RAG-style applications.

---

## Lessons Learned

Several lessons become clear once similarity search and reranking are added on top of clustering.

1. Representation quality dominates algorithm choice.
   The biggest jump in interpretability and retrieval behavior comes from moving to a carefully designed BERT+tags embedding. After that, different clustering and retrieval algorithms mostly determine how that space is sliced and scored.

2. KMeans remains a strong baseline.
   In both the original fused space and the Nyström spectral space, KMeans and MiniBatchKMeans produce stable and interpretable categories. With Nyström, the approach scales well and yields reusable cluster identifiers.

3. Density-based clustering is powerful but harsh.
   OPTICS in v2.2.3 produces the tightest and most clinically coherent topics but at the price of discarding a large fraction of the corpus as noise. Those islands are good seeds for prototypes, but they are not enough for a full taxonomy.

4. Hybrid retrieval works better than dense-only.
   Similarity v1, which uses semantic similarity and tags, is elegant but can miss edge cases, especially rare phrases or shorthand. Adding BM25 in Similarity v2 helps recover those without losing the benefits of dense embeddings and tags.

5. Single metrics can be misleading.
   High silhouette values on a subset of points in a density-based run do not reflect noise coverage. High semantic cosine alone does not guarantee intent alignment. For this application, a moderate global silhouette with full coverage and strong tag coherence, as in v2.4.3, combined with a hybrid retrieval score as in Similarity v2, is more useful than perfect metrics on restricted subsets.

---

## Limitations and Future Work

Several limitations remain for both clustering and retrieval.

* Metric comparability across spaces is imperfect. Silhouette in an LSA-inspired setting is not directly comparable to silhouette in a 40-dimensional spectral manifold. The same holds for cosine similarities and BM25 scores. A more rigorous comparison would evaluate cohesion and separation within a common reference space and with a labeled reference set.

* Human evaluation is limited in scale. Manual audits of clusters and retrieval results are still relatively small. A more systematic effort with multiple raters and agreement analysis would strengthen the evidence for coherence, naming, and retrieval utility.

* There is no formal IR evaluation yet. The similarity pipelines currently rely on qualitative inspection. A clear next step is to construct a query–relevance set and compute quantitative IR metrics such as Precision at k, Recall at k, and nDCG at k for both Similarity v1 and v2.

* Domain drift is not yet fully addressed. Medical queries change over time. A production system would benefit from scheduled retraining of embeddings, clusters, and BM25 statistics, as well as drift monitoring over embedding distributions, term statistics, and cluster size trends.

* From clusters and similarity to intents and RAG remains an open path. Clusters and neighbor sets are unsupervised proto-intents. To make them operational, one natural plan is:

  1. Fix v2.4.3 cluster assignments as a starting taxonomy.
  2. Use Similarity v2 neighborhoods to propose canonical exemplars.
  3. Work with clinicians or product owners to name and refine these groups.
  4. Train supervised intent models to predict these labels for new questions, with retrieval used as a RAG backbone and cluster IDs used for routing and reporting.

Overall, comparing v1.1.3 through v2.4.3 and Similarity v1 and v2 gives a clear view of the design space. v1.x illustrates strong, explainable BERT+tags taxonomies with KMeans and spectral clustering. v2.2.3 shows the extremes of density-based micro-topic discovery. v2.4.3 and Similarity v2 together form a pragmatic, production-leaning combination for taxonomy, retrieval, and future RAG integration.

-----------

# EDA

The dataset contains 47,603 medical chatbot questions with associated short answers, tags and labels. After applying my preprocessing steps, including normalizing text, extracting noun/proper-noun tokens through POS filtering and removing rows that contained neither meaningful POS tokens nor tags, I retained 47,491 usable questions. Only 112 rows were dropped, which confirms that the noun-focused pipeline preserves nearly all clinically relevant content despite being intentionally aggressive in filtering out functional or conversational language.

<img width="696" height="246" alt="CSCI E-108 Project - Viz - photo 6" src="https://github.com/user-attachments/assets/4ccec77b-ea23-4777-bf5c-529df4e8303b" />

# Visualization

<img width="989" height="490" alt="CSCI E-108 Project - Viz - photo 1" src="https://github.com/user-attachments/assets/83ed6320-01da-4e49-a6dd-771f89bdddb8" />

<img width="989" height="490" alt="CSCI E-108 Project - Viz - photo 2" src="https://github.com/user-attachments/assets/ad5226b6-4b60-4ac5-9ea2-0ed9445d5a65" />

Examining the question length distributions reveals a heavily skewed pattern: most normalized questions fall under forty tokens, with a long tail of narrative-style queries extending beyond 400 tokens. Once POS filtering is applied, these same questions compress to compact "keyword cores" with a mean of around seven content words. This compression is valuable for clustering because it reduces noise from filler phrasing and retains the medical entities, such as symptoms, conditions, body parts, procedures, that are central to semantic similarity. It also addresses a common issue in high-dimensional text clustering where extremely long documents can disproportionately influence cosine distances. Here, long questions become manageable abstractions instead of dominating the embedding space.

<img width="989" height="590" alt="CSCI E-108 Project - Viz - photo 3" src="https://github.com/user-attachments/assets/030152ab-36ec-48a6-8a73-6c39a6d5c1a6" />

Tag statistics further illustrate the structure of the dataset. Each question holds on average two tags, with a noticeable spike at one to two tags and a tapering tail up to twelve. The distribution is characteristically heavy-tailed: a few tags such as pregnancy, period, pain, exercise and sexual intercourse appear thousands of times, while many tags occur only a handful of times. This confirms that tags carry meaningful supervision but cannot be treated as strict class labels. They are better used as soft guidance, through multi-hot encodings and SVD projections, than as ground truth categories. Importantly, the dominant tags align closely with the most frequent terms extracted from the POS-filtered text. Words such as pain, period, sex, pregnancy, infection, skin, surgery, weight, pill and disease appear at the top of the TF-IDF list. This alignment shows that the tags are semantically coherent and reflect the same conceptual fields visible in the raw language, which is encouraging for downstream cluster evaluation via tag purity and entropy.

<img width="989" height="590" alt="CSCI E-108 Project - Viz - photo 4" src="https://github.com/user-attachments/assets/c9b23538-3f80-4d30-8e93-98e3aebb60d5" />

<img width="989" height="790" alt="CSCI E-108 Project - Viz - photo 5" src="https://github.com/user-attachments/assets/fecc4186-ea93-42fd-b8c0-8c5d32e55bc0" />

The UMAP visualization, constructed from a 5,000-sample TF-IDF subset, offers a high-level view of the semantic landscape. The embedding forms a broad, continuous cloud with several denser lobes, suggesting that the dataset does not naturally fracture into sharply separated clusters. When coloring points by the presence of the pregnancy tag, I can clearly see a concentrated region at the upper portion of the manifold, though not a perfectly isolated one. This semi-cohesive pattern matches what I observe later with OPTICS- and Spectral-based clustering: pregnancy questions tend to form a meaningful neighborhood, but they also share vocabulary with nearby themes such as menstruation, sexual health and abdominal pain. The UMAP therefore helps justify my multi-version approach. Simpler KMeans models (v1.1-v1.4) exploit the smooth global structure to produce business-readable macro-categories, while density-based and spectral approaches in the v2.x series can carve out more nuanced micro-topics within the same high-density areas.

Taken together, my EDA results provide a strong empirical basis for my modeling decisions across versions v1.1 through v2.4. The dataset is clean, semantically rich and dominated by medically meaningful vocabulary even after heavy filtering. Tags and terms reinforce each other, validating their roles as complementary signals. The manifold structure observed in UMAP makes it clear why no single clustering algorithm is sufficient: centroid-based models capture high-level structure, while density-based and spectral methods are better suited for identifying fine-grained subtopics. This interplay between dataset behavior and algorithmic design is what ultimately shaped the progressive improvements across all pipeline versions.


# Similarity Search and Reranking 

## v1

### 1. What my current pipeline is actually doing

### 1.1 Data preprocessing & NLP

**Data preprocessing**

* Load `train_data_chatbot.csv`.
* Fill missing `short_question` / `tags` with empty strings.
* Normalize text:

  * lowercasing, quote/dash normalization, strip non-alphanumeric except `'` and `?`.
  * store as `short_question_norm`.
* Parse tags:

  * regex `TAG_PATTERN = r"'([^']+)'"` → `tag_list` as a list of normalized strings.

**POS filtering (spaCy)**

* I run `build_pos_filtered_texts` over all questions:

  * `en_core_web_sm` (NER & textcat disabled).
  * Keep tokens where:

    * POS ∈ {NOUN, PROPN}, **or**
    * lemma in `MED_KEEP_SHORT` (HPV, UTI, HIV, etc.).
  * Drop:

    * standard + domain stopwords,
    * numeric-like tokens,
    * length ≤ 2 unless whitelisted.

→ This gives me `short_question_pos`, which focuses on *medical content nouns* (diseases, body parts, meds, etc.) and strips chatty filler.

**Effect / insight**

* I'm pushing all semantic weight onto medically meaningful nouns.
* For clustering & retrieval, this:

  * removes noise and makes *islands* like “shingles / hypothyroidism / hepatitis” more obvious;
  * but throws away tense, negation, and some modifiers (e.g. “not pregnant” vs “pregnant” may look similar).

Preprocessing runtime (from my log):

* `preprocess_s ≈ 123 s` → spaCy is the main one-time cost.

---

### 1.2 Modeling / representation

I then build a **joint dense representation**:

1. **Sentence BERT over `short_question_pos`:**

   * `all-MiniLM-L6-v2` → `X_sem` (dense vectors).
   * optional row-wise L2 normalization.
   * PCA → 80D (`N_BERT_DIM`), explained variance ≈ 0.72.
   * StandardScaler + L2 → `X_sem_norm` (shape `(47491, 80)`).

2. **Tags → MultiLabelBinarizer → sparse matrix → TruncatedSVD:**

   * SVD → 20D (`N_TAG_DIM`), explained variance ≈ 0.24.
   * StandardScaler + L2 → `X_tag_norm` (shape `(47491, 20)`).

I **don’t** concatenate here; for retrieval I keep them as two separate blocks and combine at scoring time (semantic cosine + Jaccard).

**Insights**

* BERT+PCA block encodes semantic similarity over noun-phrases.
* Tag SVD encodes “weak supervision” across tags (co-occurrence structure).
* I treat tags in **two ways**:

  * as dense latent vectors (`X_tag_norm`) used implicitly (even though I don’t yet use `X_tag_norm` in scoring), and
  * as discrete sets (`tag_sets`) for Jaccard similarity.

Right now, in `search_questions`, the **score actually used** is:

* `semantic_score = X_sem_norm @ q_sem_norm`
* `tag_jaccard` on the candidate set
* `blended_score = 0.7 * semantic_score + 0.3 * tag_jaccard`

`X_tag_norm` is built but not yet exploited in the scoring formula — we could use it later if I want a *dense* tag similarity.

---

### 1.3 Retrieval logic

For each query:

1. **Encode query** (same tower as docs):

   * normalize text → POS filter → Sentence BERT → PCA → StandardScaler → L2 → `q_sem_norm`.
   * tags (if any) → mlb → SVD → StandardScaler → L2 → `q_tag_norm` + `q_tag_set`.

2. **Candidate selection**:

   * semantic similarity only:

     ```python
     sem_scores = X_sem_norm @ q_sem_norm
     cand_idx = top candidate_k by sem_scores (e.g. 300)
     ```

3. **Reranking**:

   * compute **tag Jaccard** on `cand_idx` using `q_tag_set` and each doc’s `tag_sets[i]`.
   * blended score `alpha*semantic + beta*tag_jaccard`.

4. **Return top_k** with:

   * original question
   * tags, pos text
   * `semantic_score`, `tag_jaccard`, `blended_score`
   * overlap tags for inspection.

**Behaviour**

* Text-only queries: behave as pure dense search in BERT space.
* Tags-only (Example 2): semantic vector is all zeros → only tag Jaccard drives ranking.
* Mixed: BERT gets I semantically similar questions; tag Jaccard pulls up items with overlapping intents when tags provided.

---

### 1.4 Measurement & metrics

Right now I'm measuring “quality” by:

* The **scores themselves**:

  * `semantic_score` ~ cosine similarity (because of L2 norm) in `[-1, 1]`, typically `[0, 1]`.
  * `tag_jaccard` in `[0, 1]`.
  * `blended_score` in some interpolated range.

* **Qualitative auditing**:

  * I printed neighbours and judged them by eye (e.g., pregnancy/period examples look very coherent).
  * no explicit retrieval metrics like nDCG / recall@k because there isn’t a labeled pairing dataset.

For my use case (medical FAQ reuse + intent discovery), this is fine: the main evaluation is “are the neighbours medically meaningful and tag-aligned?”

---

## 2. What’s slow and what can be improved?

From my timings:

* `preprocess_s ≈ 123.6 s` → spaCy POS across 47.5k rows (one-off).
* `vectorize_s ≈ 92.7 s` → BERT embedding + PCA + SVD + scaling (one-off).
* Query-time: quite cheap (one BERT forward + one spaCy+POS on the query + matrix multiply).

**Key points:**

1. **Offline cost is fine** — I do it once, then save.

2. **Online cost:**

   * spaCy+POS on *every query* is a bit heavy; I could:

     * Option A: keep POS at **index time only**, and at query time just use a lighter normalization without POS.
     * Option B: keep POS for high-quality queries but add a flag to bypass spaCy if I need speed.

3. **Similarity search:**

   * I do full dot product `X_sem_norm @ q_sem_norm` over 47k docs → this is **fast** (< 0.01 s typically), no real need for FAISS yet.
   * If I scale to millions of rows later, I would add an ANN index (FAISS/hnswlib).

4. **Under-used pieces:**

   * `X_tag_norm` (dense tag SVD) is built but not used.
   * `BM25` / lexical scoring is missing — helpful where synonyms *aren’t* learned by BERT or the question is extremely short.

---

## 3. HDBSCAN vs BM25 for improving similarity & reranking

* **HDBSCAN** is **clustering**, not retrieval:

  * Great for discovering “islands” in embedding space (like my v2.3 OPTICS).
  * Not ideal for similarity search because:

    * many points may be labeled as noise,
    * I still need a distance measure within clusters to rank neighbours,
    * user queries aren’t pre-clustered.

  ➜ I would **not** add HDBSCAN into this retrieval pipeline. It’s something I’d use **offline** to define topic clusters, not as a scoring component.

* **BM25** is a **lexical ranking function**:

  * Scores docs based on overlapping tokens, with IDF and length normalization.
  * Great complement to dense BERT:

    * catches exact phrase / keyword matches,
    * helps when the query contains rare words that BERT might underweight.

  ➜ For **similarity search and reranking**, BM25 is the right tool to add, not HDBSCAN.

---

## 4. Concrete improvements I’d suggest

### 4.1 Architectural / logic improvements

1. **Hybrid scoring: semantic + BM25 + tag Jaccard**

   New blended score:

   [
   \text{score} = \alpha \cdot \text{semantic_cosine} + \beta \cdot \text{tag_jaccard} + \gamma \cdot \text{bm25_norm}
   ]

   For example:

   * `alpha_semantic = 0.6`
   * `beta_tag_jaccard = 0.2`
   * `gamma_bm25 = 0.2`

2. **Candidate selection via union of top-K semantic & top-K BM25**

   * Get top `candidate_k_sem` docs by BERT cosine.
   * Get top `candidate_k_bm25` docs by BM25.
   * Take the union → typically a few hundred docs.
   * Rerank this candidate set with the blended score.

3. **Optional: use dense tag similarity**

   * I could add a fourth term `delta * cosine(X_tag_norm[i], q_tag_norm)`.
   * For now, Jaccard is simpler and more interpretable.

4. **Optional: spaCy at query-time**

   * Add a flag `USE_SPACY_FOR_QUERY = False` if I want to speed up many user queries.
   * When `False`, just use `normalize_text_basic` and skip POS in the query tower.

---

## 5. Refactored full code with BM25 reranker

Below is a full script based on my current one, **plus**:

* BM25 index built on `short_question_norm`.
* Hybrid candidate selection: semantic + BM25.
* Blended score: semantic + tag Jaccard + BM25.

> I can paste this into a fresh Colab cell and run it end-to-end.
> It reuses my existing logic, just adds the BM25 bits and a few toggles.

## v2

### 1. Data preprocessing

**What I see in the log**

* Original dataset: ~47,603 rows
* After POS cleanup: **47,491 rows**

  > `[Preprocess] Rows after POS cleanup: 47491`

So the POS-based filtering only removed **112 questions** – that means the pipeline is aggressive on function words but conservative on dropping rows. Good: coverage is essentially intact.

**Key preprocessing steps implied:**

1. **POS cleanup on `short_question`**

   * Keep mostly **content words** (NOUN / PROPN, maybe some others), remove fillers, verbs, auxiliaries, etc.
   * This produces something like `short_question_norm`, which is later used for BM25 and semantic encoding.

2. **Tag normalization**

   * Tags appear as a **clean list**:

     ```python
     ['breast' 'cramps' 'period' 'pregnancy' 'sexual intercourse']
     ```
   * This suggests:

     * stripping brackets / quotes,
     * splitting / trimming,
     * lowercasing,
     * applying frequency filters (in earlier versions: freq>100, rare-tag pruning, etc.).

**Why this is good**

* I keep **almost all rows**, so the retrieval index is faithful to the original dataset.
* POS-cleaned questions reduce noise while preserving core medical entities (“period”, “pregnancy”, “breast”, “cramps”).
* Tags are normalized into a machine-friendly format, which later supports **tag-based retrieval** and **Jaccard similarity**.

---

## 2. NLP feature engineering

The log shows:

> `[Features] BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245)`
> `[Shapes] X_sem_norm: (47491, 80), X_tag_norm: (47491, 20)`

### 2.1 Text (semantic) side

* Model: **SentenceTransformers – `all-MiniLM-L6-v2`**
* Original embedding dim: 384
* PCA → **80 dimensions**, with **72.1% explained variance**.

Interpretation:

* I am compressing each question into an 80D vector that still retains most (~72%) of the semantic variance.
* This is a strong trade-off between:

  * Memory / compute efficiency, and
  * Retaining enough semantic structure for good **cosine similarity**.

The matrix:

* `X_sem_norm`: shape `(47491, 80)`
  → one 80D semantic vector per question, likely L2-normalized.

### 2.2 Tag side

* Tags are turned into some bag-of-tags representation and then:
* Truncated SVD to **20D**, with **24.5% explained variance**.

Interpretation:

* Tag space is much smaller and more discrete, so even though EV=0.245 looks “low”, tags are sparse and categorical; SVD mainly offers:

  * Dimensionality reduction,
  * Smoothed co-occurrence structure between tags,
  * A compact representation for clustering / later models.

The matrix:

* `X_tag_norm`: shape `(47491, 20)`
  → one 20D tag-based vector per question, probably normalized as well.

**Big picture**

* Each question is now backed by:

  * An **80D semantic vector** (from BERT+PCA),
  * A **20D tag vector** (from tag SVD),
  * Plus raw text for BM25.

This gives me three “views” of the data: lexical (BM25), semantic (BERT), and categorical (tags).

---

## 3. Retrieval model (hybrid semantic + lexical + tags)

The printed examples show the scoring logic clearly.

### 3.1 Text-only query

Example 1:

> Query: `"period late cramps"`
> Columns: `short_question`, `tags`, `semantic_score`, `bm25_score`, `blended_score`

Observations:

* `semantic_score` ≈ 0.78–0.85 (cosine similarity to BERT embedding of the query).
* `bm25_score` ≈ 12–20 (classical sparse lexical match).
* `blended_score` somewhere between 0.61–0.64.

This suggests:

* I compute:

  * A **semantic similarity** (cosine) between query BERT embedding and `X_sem_norm`.
  * A **BM25 similarity** over `short_question_norm`.
* Then I combine them into a **normalized blended score**, probably something like:
  [
  \text{blended} = \alpha \cdot \text{semantic_norm} + \beta \cdot \text{bm25_norm}
  ]
* The top results show **strong semantic and lexical overlap**:

  * “sore breast… bad cramps but no period… could I be pregnant”
  * “period was 8 days late… cramps… could I be pregnant”
* That’s exactly the behavior I want:
  BM25 handles exact terms (“period”, “cramps”), BERT picks paraphrases / variants.

### 3.2 Tags-only query

Example 2:

> Query tags: `['pregnancy', 'period']`
> Columns: `semantic_score`, `tag_jaccard`, `bm25_score`, `blended_score`

Observations:

* `semantic_score` = **0.0**, `bm25_score` = **0.0** for all rows.
  That means:

  * For a pure tag-based query, I intentionally **turn off** text/semantic scores.
* `tag_jaccard` ranges from **1.0** (exact tag match) to 0.5, 0.25, etc.
* `blended_score` ≈ 0.2, 0.1, etc.
  This implies a simple mapping from Jaccard to a scaled blended score.

Interpretation:

* When I query only with tags, I rely **exclusively** on **set overlap between query tags and question tags**:
  [
  J(A,B) = \frac{|A \cap B|}{|A \cup B|}
  ]
* This is nice because:

  * It’s **transparent** and easy to reason about.
  * It acts as a strong filter when a doctor or model already knows the relevant tags.

### 3.3 Mixed query (text + tags)

Example 3:

> Mixed query: text about "period late" + tags like `['pregnancy', 'period']` (inferred)
> Columns: `semantic_score`, `tag_jaccard`, `bm25_score`, `blended_score`

Observations:

* `semantic_score` ~ 0.69–0.89
* `tag_jaccard` ~ 0.75–1.0
* `bm25_score` ~ 16–21
* `blended_score` ~ 0.75–0.81

Interpretation:

* For mixed queries, I combine **three signals**:

  * Semantic (BERT),
  * Lexical (BM25),
  * Tag overlap (Jaccard).
* The top results are all very on-topic:

  * “period due… spotting… pregnancy test… am I pregnant”
  * “missed my period last month… very fertile day… test was negative… pregnant?”
* This is exactly the kind of medically coherent neighborhood I want for a retrieval system powering a chatbot.

---

## 4. Measurement and metrics

### 4.1 Retrieval quality (qualitative)

At this v2 stage, the **main measurement** is qualitative:

* For **text-only query** (“period late cramps”), the top hits mention:

  * sore breasts, bad cramps, late period, pregnancy concern.
* For **tags-only query** (`['pregnancy', 'period']`), the top hits:

  * all include both pregnancy and period tags, with high Jaccard.
* For **mixed query**, the system surfaces nuanced “am I pregnant?” scenarios tied to timing, tests, and symptoms.

So even without formal IR metrics yet, the **face validity** of top results is strong.

### 4.2 Timing / runtime metrics

Log:

> `[Timing] Wrote timings JSON to: .../retrieval_timings_bm25.json`
>
> `preprocess_s: 128.5489`
> `vectorize_s: 109.4065`
> `total_runtime_s: 246.7312`

Interpretation:

* Total end-to-end runtime for v2 on ~47k questions: **~247 seconds**.

  * ~129s for preprocessing (POS, normalization, tag cleanup).
  * ~109s for vectorization (BERT inference + PCA + SVD + BM25 index).
* These timings are saved to JSON, so I can:

  * Compare v1 vs v2 vs future versions.
  * Quantify trade-offs if I change BERT model, dimensionality, or BM25 parameters.

What’s missing (potential future work):

* Formal IR metrics:

  * **Precision@k**, **Recall@k**, **nDCG@k** using a small labeled set of query–relevant pairs.
* Latency per *single* query:

  * Right now I mainly log offline build time; per-query latency will matter for a live system.

---

## 5. Overall evaluation of v2

**Strengths**

* **Data preprocessing**:

  * POS cleanup preserves almost all rows while removing noise.
  * Tags are normalized and usable for both filtering and similarity.
* **NLP representations**:

  * BERT+PCA (80D) captures ~72% of semantic variance with a much smaller footprint.
  * Tag SVD (20D) provides a compact representation for later clustering / modeling.
* **Hybrid retrieval model**:

  * Combines **semantic**, **lexical**, and **tag** signals.
  * Behaves well across three query modes: text-only, tags-only, and mixed.
* **Measurement**:

  * Runtime is tracked in JSON, giving a baseline (~247s) for future optimizations.

**Limitations / next ideas**

* PCA to 80D still loses about 28% of semantic variance; for very fine-grained distinctions, I might push it to 100–128D and see if retrieval improves.
* Tag SVD keeps only 24.5% EV; I might:

  * increase to 32–50 dimensions, or
  * skip SVD for retrieval and use raw tag sets (as I already do via Jaccard).
* Evaluation is qualitative; adding a small **gold relevance set** would let me compute:

  * Precision@5 / @10, nDCG, and measure the direct contribution of BM25 vs BERT vs tags.
* I see duplicate rows in the output (same question printed twice). That suggests duplicates in the dataset or in the retrieval merging logic; I may want to deduplicate for cleaner UX.

# Comparison between v1 and v2

## 1. Big-picture comparison

| Aspect               | v1 (semantic + tags)                                        | v2 (semantic + tags + BM25)                                      |
| -------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------- |
| Text representation  | POS-filtered BERT (MiniLM) + PCA                            | Same POS-filtered BERT + PCA                                     |
| Tag representation   | Tag SVD + **Jaccard** used in scoring                       | Tag SVD (for completeness) + **Jaccard** in scoring              |
| Lexical signal       | None                                                        | **BM25 over `short_question_norm`**                              |
| Candidate generation | Top-K by semantic cosine only                               | Top-K semantic **union** Top-K BM25                              |
| Reranking signals    | Semantic cosine + tag Jaccard                               | Semantic cosine + tag Jaccard + normalized BM25                  |
| Query NLP            | POS-filtered text → BERT (always)                           | POS-filtered or raw (controlled by `USE_SPACY_FOR_QUERY`)        |
| Runtime (offline)    | ~218s (preprocess + vectorize)                              | ~247s (extra BM25 build + slightly heavier vectorization)        |
| Strength             | Clean semantic + tag intent blend                           | **Hybrid**: semantic, lexical, and tag signals                   |
| Weakness             | Can miss exact-phrase matches; pure text-only is dense only | Slightly slower; more hyperparameters (α, β, γ, candidate sizes) |

---

## 2. NLP & preprocessing logic

### Common backbone (v1 & v2)

Both versions share the same core NLP view of the corpus:

* Normalize raw text → `short_question_norm`.
* Parse tags into `tag_list`.
* Build POS-filtered text → `short_question_pos` using spaCy, keeping:

  * NOUN / PROPN,
  * plus medically important short tokens (`hpv`, `uti`, etc.).
* Drop rows with no content (no POS text and no tags) → 47,491 rows kept.

So at the **document level**, v1 and v2 are aligned: both are operating over a medically focused, noun-heavy representation. That keeps semantic space tight around conditions, body parts, drugs, etc.

### Query-side NLP

**v1:**

* Always:

  * Normalize query text.
  * POS-filter with spaCy.
  * BERT → PCA → scale → L2.
* Every query uses the **same heavy POS pipeline** as the corpus.

Pros:

* Query and corpus are treated strictly symmetrically.
* Helps when a raw query is chatty; the semantic vector is centered on content nouns.

Cons:

* Query-time cost: spaCy is expensive, especially for many queries.
* For short queries like “missed period”, POS filtering adds little but still costs a full spaCy pass.

**v2:**

* Normalizes text the same way, but:

  * POS-filtering at query-time is **configurable** (`USE_SPACY_FOR_QUERY`).
  * If `USE_SPACY_FOR_QUERY = True`: behaves like v1.
  * If False: skip spaCy, send normalized text straight into BERT.

This gives a **control knob** between fidelity and latency:

* When performance matters (e.g., a production API), POS can be turned off for queries.
* When a detailed semantic alignment is needed (e.g., offline analysis), POS can be kept.

**My takeaway:**
The underlying NLP representation of the corpus is nearly identical across v1 and v2. The difference is that v2 recognizes that query-time POS normalization is optional and expensive and gives a flag to turn it off. Conceptually, v1 is “always maximal NLP”, v2 is “maximal on the corpus, configurable on the query”.

---

## 3. Modeling logic – how retrieval actually works

### v1: dense semantic + tag-aware reranking

**Index:**

* Semantic tower: `short_question_pos` → BERT → PCA(80D) → scale → L2 → `X_sem_norm`.
* Tag tower: `tag_list` → MultiLabelBinarizer → SVD(20D) → scale → L2 → `X_tag_norm`.
* Tag sets: `tag_sets = [set(tags) for tags in tag_list]`.

**Query encoding:**

* Query text → POS → BERT → PCA → scale → L2 → `q_sem_norm`.
* Query tags (optional) → mlb → SVD → scale → L2 → `q_tag_norm`, `q_tag_set`.

**Candidate selection:**

* Semantic cosine: `sem_scores = X_sem_norm @ q_sem_norm`.
* Take top `candidate_k` by semantic similarity.

**Reranking:**

* For those candidates:

  * Tag Jaccard: `|intersection(q_tag_set, tag_sets[i])| / |union(...)|`.
* Blended score:

  ```python
  blended = alpha_semantic * sem_scores + beta_tag_jaccard * tag_jacc
  ```
* Sort and return top_k.

**Behavior:**

* Text-only queries: pure dense semantic retrieval.
* Tags-only queries: essentially tag Jaccard ranking (semantic vector ~ 0).
* Mixed: semantic neighbourhood restricted further by tag overlap.

v1 is a **two-signal reranker**: meaning (BERT) + supervision (tags).

---

### v2: hybrid semantic + tags + BM25

v2 keeps almost all of v1’s structure and adds a **third signal**: classical BM25 over the normalized text.

**Index additions:**

* Build BM25 index:

  * `short_question_norm` → token lists → `BM25Okapi`.
* Keep `bm25_corpus_tokens` for completeness.

**Query encoding additions:**

* Besides `q_sem_norm`, `q_tag_norm`, `q_tag_set`:

  * `bm25_query_tokens = norm.split()`.

**Candidate selection:**

1. Dense semantic:

   * `sem_scores = X_sem_norm @ q_sem_norm`.
   * `top_sem = argpartition(-sem_scores, candidate_k_sem)`.
2. Lexical BM25:

   * `bm25_scores = bm25.get_scores(bm25_query_tokens)`.
   * `top_bm25 = argpartition(-bm25_scores, candidate_k_bm25)`.
3. Combined candidate pool:

   * `cand_idx = unique(concatenate(top_sem, top_bm25))`.

**Reranking:**

On `cand_idx`:

* `sem_cand = sem_scores[cand_idx]`.
* `bm25_cand = bm25_scores[cand_idx]` → normalized to `[0,1]` → `bm25_norm`.
* Tag Jaccard: as in v1, but computed only for candidates.

Blended score:

```python
blended = (
    alpha_semantic    * sem_cand +
    beta_tag_jaccard  * tag_jacc +
    gamma_bm25        * bm25_norm
)
```

Defaults: `α=0.6, β=0.2, γ=0.2`.

Then sort and return top_k.

**Behavior change:**

* Text-only queries now benefit from **both**:

  * semantic neighbours (paraphrases / synonyms),
  * exact lexical matches and phrase overlaps via BM25.
* Tags-only queries still work (BM25 drops to zero, semantic vector close to zero), so the ranking is essentially tag Jaccard + small BM25 baseline.
* Mixed queries are the richest:

  * semantic pulls neighbors in embedding space,
  * BM25 ensures phrasing closeness (e.g., “missed period” vs “late period”),
  * tags enforce intent consistency.

**My mental model:**

* v1: “dense semantic retrieval with tag-aware reranking”.
* v2: “hybrid semantic + lexical + tag reranking”—much closer to a typical **RAG-style retrieval stack**.

---

## 4. Performance & metrics – what I learn from timings and scores

### Runtime

* v1:

  * preprocess ≈ 123.6s,
  * vectorize ≈ 92.7s,
  * total ≈ 217.9s.
* v2:

  * preprocess ≈ 128.5s,
  * vectorize ≈ 109.4s,
  * total ≈ 246.7s.

So v2 costs ~30 seconds more on the full pipeline due to:

1. Additional BM25 index build.
2. Slight extra overhead from tag + BM25 handling.

Given ~47k documents, this is still cheap as an offline job. Query-time latency is only marginally affected: the heavy part remains the BERT forward pass; BM25 scoring is vectorized over the corpus.

### Implicit “metrics” and qualitative performance

Both versions track:

* PCA explained variance (`≈0.721` for BERT 80D).
* Tag SVD explained variance (`≈0.245` for 20D).

This tells me:

* Dimensionality reduction is not aggressive enough to destroy semantic structure.
* Tag latent factors are compressed but still meaningful.

From the examples:

* v1’s blended score = semantic + tag Jaccard.
* v2’s blended score = semantic + tag Jaccard + scaled BM25.

Observations:

1. For a **text-only query**, v2 tends to:

   * Keep the same “type” of top results as v1.
   * Reorder them slightly based on exact token overlap and position (BM25).
   * Surface more diverse phrasings when semantic and BM25 disagree.

2. For a **tags-only query**, both v1 and v2 behave similarly:

   * `semantic_score ≈ 0` and `bm25_score ≈ 0`,
   * `blended_score ≈ beta_tag_jaccard * Jaccard`.
   * The ranking is essentially a measure of tag compatibility.

3. For a **mixed query**, v2’s ranking becomes more nuanced:

   * Items with strong semantic relevance but weaker lexical overlap remain high,
   * Items with both high semantic scores and high BM25 scores are pushed to the very top,
   * Tags break ties and encourage consistency across retrieved examples.

There is no annotated ground-truth to compute recall@k or nDCG, but qualitatively:

* v1 already finds clinically consistent neighbours.
* v2 preserves that and adds an additional control over exact-term priority.

---

## 5. Conceptual “insights” about v1 vs v2

### NLP design philosophy

* **v1**: “Let me make the semantic representation as clean and medical as possible and drive everything off that (plus tags).”

  * Strong commitment to POS-filtered text at both document and query side.
  * Retrieval is elegantly dense, with tags as a weak supervision signal.

* **v2**: “The semantic space is good, but real systems benefit from lexical matching too.”

  * Keeps the same careful NLP for the corpus.
  * Recognizes that queries might not always warrant full spaCy passes.
  * Adds an independent lexical model (BM25) over lightly normalized text, not over POS-filtered text.

In short, v1 is **semantic-first**, v2 is **hybrid-first**.

### Modeling / retrieval logic

* v1’s retrieval view:

  * The dense embedding space is the only place where similarity is defined.
  * Tags influence ranking only within that semantic neighbourhood.
  * Missing exact lexical matches is acceptable if semantic distance is low.

* v2’s retrieval view:

  * Similarity is a **triangular compromise**:

    * semantic alignment (BERT),
    * lexical exactness (BM25),
    * tag intent overlap (Jaccard).
  * Candidates come from either dense or lexical worlds, then get merged.
  * This is closer to how a production system might behave: allow either semantics or exact terms to “trigger” candidates and then blend with intent signals.

### Performance & operational trade-offs

* v1 is slightly lighter and conceptually simpler; ideal for:

  * pure embedding-based exploration,
  * tight semantic clustering and label propagation,
  * experiments where lexical matching is less important.

* v2 is more expressive but a bit heavier; ideal for:

  * retrieval scenarios where exact phrases matter (e.g., repeated complaints),
  * integration with RAG, where BM25 helps align with knowledge snippets,
  * robust performance across heterogeneous queries (some short, some longer).

---

## 6. How I would summarize the difference in one paragraph

In my head, v1 is “dense BERT + tags as a clean, mathematically elegant similarity engine”, whereas v2 is “a pragmatic hybrid search stack” that takes the good semantic foundation of v1 and anchors it with BM25. The NLP logic is almost identical at the document level—POS-filtered text plus tag parsing—but v2 acknowledges operational realities: query-time POS can be optional, and lexical BM25 should sit alongside dense embeddings to recover exact patterns. Performance-wise, v2 pays about 10–15% extra offline time to gain a third, orthogonal signal (lexical), which gives more control over how similar questions are surfaced and how tag signals are used to rerank.



# **Reference List for Clustering, NLP, EDA, Topic Modelling, and Retrieval**

### **Core NLP Preprocessing and POS Filtering**

1. Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. *spaCy 3: Industrial-Strength Natural Language Processing in Python* (2021).
   [https://spacy.io/](https://spacy.io/)
2. spaCy Documentation - Tokenization, Lemmatization, POS Tagging.
   [https://spacy.io/usage/linguistic-features](https://spacy.io/usage/linguistic-features)
3. Bird, S., Klein, E., & Loper, E. *Natural Language Processing with Python*. O’Reilly Media (2009).
   [https://www.nltk.org/book/](https://www.nltk.org/book/)

---

# **TF-IDF, Dimensionality Reduction (LSA/SVD/PCA), and Classical Clustering**

4. Manning, C., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*. Cambridge University Press (2008). (TF-IDF, cosine similarity, sparse vectors)
   [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)
5. Deerwester, S. et al. *Indexing by Latent Semantic Analysis*. JASIS (1990).
6. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR (2011). (KMeans, MiniBatchKMeans, PCA, TruncatedSVD)
   [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

---

# **UMAP & HDBSCAN (used heavily in v2.x and EDA visualization)**

7. McInnes, L., Healy, J., & Melville, J. *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426 (2018).
   [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
8. Campello, R. J. G. B., Moulavi, D., Sander, J. *Advances in Density-Based Clustering: HDBSCAN*. J. Intell. Inf. Syst. (2015).
   [https://hdbscan.readthedocs.io/](https://hdbscan.readthedocs.io/)
9. McInnes, L. *UMAP + HDBSCAN Examples*.
   [https://umap-learn.readthedocs.io/en/latest/clustering.html](https://umap-learn.readthedocs.io/en/latest/clustering.html)

---

# **OPTICS (Versions v2.1-v2.3)**

10. Ankerst, M. et al. *OPTICS: Ordering Points To Identify the Clustering Structure*. SIGMOD (1999).
11. Scikit-learn OPTICS Documentation.
    [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)

---

# **Spectral Clustering + Nyström Approximation (v2.4)**

12. Ng, A. Y., Jordan, M. I., Weiss, Y. *On Spectral Clustering: Analysis and an Algorithm*. NIPS (2001).
13. Drineas, P., Mahoney, M. *Nyström Methods and Their Use in Large-Scale Machine Learning* (2016).
14. Scikit-Learn SpectralEmbedding.
    [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html)

---

# **Sentence Embeddings & Transformer Models (v2.x feature stack)**

15. Reimers, N., & Gurevych, I. *Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks*. EMNLP (2019).
    [https://www.sbert.net/](https://www.sbert.net/) ([SentenceTransformers][1])
16. HuggingFace Transformers Documentation.
    [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

---

# **BERTopic (v5 in my proposal; also inspires UMAP/HDBSCAN setup)**

17. Grootendorst, M. *BERTopic: Neural Topic Modeling with Transformers*.
    [https://maartengr.github.io/BERTopic/](https://maartengr.github.io/BERTopic/)
18. BERTopic GitHub Repository.
    [https://github.com/MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)

---

# **Vector Search, Similarity Search, Retrieval, and Reranking (v1 & v2-style hybrid stack)**

19. Johnson, J., Douze, M., Jégou, H. *Billion-scale Similarity Search with GPUs - FAISS Library*. Facebook AI Research (2017).
    [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) ([DZone][2])

20. Karpukhin, V. et al. *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP (2020).
    (Dense bi-encoder retrieval that underpins many BERT-based semantic search systems.) ([Medium][3])

21. Khattab, O., & Zaharia, M. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. SIGIR (2020).
    (Late-interaction architecture for high-quality dense retrieval and reranking.) ([Emergent Mind][4])

22. SentenceTransformers Documentation - *Semantic Search and Cross-Encoder Reranking*.
    (Practical recipes for dual-encoder retrieval + cross-encoder reranking, similar in spirit to my semantic + BM25 + tag rerank pipeline.) ([SentenceTransformers][1])

23. Nogueira, R., Cho, K. *Passage Re-ranking with BERT*. arXiv:1901.04085 (2019).
    (Classic cross-encoder reranker, conceptually linked to my blended scoring/reranking stages.) ([ResearchGate][5])

24. Thakur, N. et al. *BEIR: A Heterogeneous Benchmark for Information Retrieval*. NeurIPS (2021).

25. Robertson, S. *Understanding BM25 and Beyond*. Foundations and Trends in Information Retrieval (2009).
    [https://plg-group.github.io/ir-book/Inverted-Indexes-BM25.html](https://plg-group.github.io/ir-book/Inverted-Indexes-BM25.html) ([Collaborate][6])

26. Elastic - *Semantic Reranking* Documentation.
    (Concrete example of hybrid BM25 + transformer-based reranking similar to my v2 design.) ([Elastic][7])

27. Milvus / SentenceTransformers - *How to Use Sentence Transformers for Semantic Search*.
    (Step-by-step description of embedding generation, indexing, and similarity matching for vector search.) ([Milvus][8])

28. dcarpintero. *Multilingual Semantic Search with Reranking* (GitHub).
    (End-to-end example combining dense retrieval with reranking on a large vectorized corpus.) ([GitHub][9])

---

# **Data Cleaning, Heavy-Tailed Distributions, and Tag Noise**

29. Sculley, D. *Web-Scale High-Dimensional Text Clustering* (2007).
30. Mikolov, T. et al. *Linguistic Regularities in Continuous Space Word Representations*.

---

# **General Machine Learning & Clustering Theory (writing support)**

31. Bishop, C. *Pattern Recognition and Machine Learning*. Springer (2006).
32. Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*. Springer (2009).
33. Manning, C., & Schütze, H. *Foundations of Statistical Natural Language Processing*. MIT Press (1999).

---

### **v1.1-v1.4 (LSA + KMeans family)**

References: 4, 5, 6, 12, 25, 31, 32

### **v2.1-v2.3 (BERT-Hybrid + OPTICS / HDBSCAN)**

References: 7, 8, 9, 10, 11, 15, 16

### **v2.4 (BERT-Hybrid + Nyström Spectral + KMeans)**

References: 12, 13, 14, 15, 16, 19, 20, 21

### **POS filtering + NLP normalization**

References: 1, 2, 3

### **TF-IDF, tags SVD, sparse matrices**

References: 4, 5, 6

### **UMAP & EDA visualization**

References: 7, 8, 9

### **Similarity search, hybrid retrieval, and reranking (v1 & v2)**

References: 15, 16, 19-28


