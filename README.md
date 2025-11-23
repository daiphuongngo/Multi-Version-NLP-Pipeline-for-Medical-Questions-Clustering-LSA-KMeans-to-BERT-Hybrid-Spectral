# From Clusters to Retrieval: Hybrid BERT-Based Taxonomy and Similarity Search for Medical Chatbot Questions

![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master of Liberal Arts (ALM), Data Science**

## **CSCI E-108 Data Mining, Discovery and Exploration**

## Professor: **Stephen Elston, PhD**, Princeton University, Principal Consultant, Quantia Analytics LLC

## Name: **Dai Phuong Ngo (Liam)**

## Abstract

My work compares seven clustering pipelines (v1.1.2, 1.2.2, 1.4.2, 2.1.2, 2.2.2, 2.3.2, 2.4.2) on an ≈47.5k-row corpus of short medical chatbot questions with associated tags. The goals are: (1) to build an explainable, business-ready taxonomy for routing and analytics, and (2) to understand how far unsupervised structure can go before supervised intent models become necessary.

I start with TF-IDF/LSA + KMeans baselines (v1.1.2–1.2.2), move to POS-filtered representations and spectral clustering in LSA space (v1.4.2), and then switch to BERT embeddings plus tag features for the v2.x family. On the clustering side, I experiment with KMeans, full spectral clustering, density-based OPTICS in a BERT+tags cosine space (v2.1.2–2.2.2), and finally a scalable Nyström spectral embedding followed by MiniBatchKMeans with a k-scan (v2.3.2–2.4.2).

Across versions, the early LSA + KMeans runs (v1.1.2–1.2.2) produce clean, board-friendly top-level categories but suffer from one or two mega-clusters. OPTICS (v2.1.2–2.2.2) uncovers extremely tight topic “islands” but labels most of the corpus as noise. The Nyström + KMeans pipelines (v2.3.2–2.4.2) strike the best balance between semantic coherence, tag alignment, runtime, and full coverage. I close with an integrated picture of how these clusters can support labeling, routing, and future retrieval or RAG-style answering.

---

## Executive Summary

I ran seven main clustering variants on the same underlying dataset and tags.

### v1.1.2–v1.2.2 (TF-IDF + LSA + KMeans)

Early baselines using bag-of-words and tags in a linear LSA space. After basic normalization and TF-IDF, I apply TruncatedSVD to obtain ~200-dimensional LSA vectors and run KMeans with a fixed k. These runs already produce a recognizable medical taxonomy (pregnancy, skin, mental health, general infection, etc.), but they also tend to create one or two **mega-clusters** that mix generic pain/symptom questions and “what is this” prompts. Internal metrics are reasonable, but silhouette and cluster size stats clearly show heavy imbalance.

### v1.4.2 (TF-IDF + LSA + Spectral Clustering)

In v1.4.2 I introduce stricter preprocessing (POS-based noun/proper-noun filtering) and move from vanilla KMeans to **spectral clustering** in LSA space. The POS filter removes a lot of filler words and keeps medically meaningful nouns (conditions, body parts, drugs). Spectral clustering makes clusters less spherical and more aligned with the manifold structure. With k-scan and a dendrogram on cluster centroids, v1.4.2 gives a good top-level view, but it still relies on TF-IDF/LSA features rather than a fully semantic embedding.

### v2.1.2–v2.2.2 (BERT + Tags + OPTICS)

In the v2.x line I switch to a joint **BERT + tags** representation. Questions are POS-filtered, embedded with a Sentence-Transformer model, compressed via PCA, and concatenated with SVD-reduced tag features. OPTICS with cosine distance is then applied in this 100D space.

* OPTICS discovers very **dense, clinically coherent micro-topics** (e.g., shingles, hepatitis, thyroid, certain medication side-effects).
* However, for realistic hyperparameters, it labels only a **small fraction** of points as cluster members and treats the remainder as noise.

The result is an interesting “upper bound” on how sharp clusters can be if coverage is ignored. These versions are excellent for discovering canonical islands but poor as full taxonomies.

### v2.3.2–v2.4.2 (BERT + Tags → Nyström Spectral + KMeans)

v2.3.2 and v2.4.2 keep the mature BERT+tags feature design but replace OPTICS with a scalable **Nyström spectral** embedding followed by **MiniBatchKMeans** and a k-scan:

* Nyström uses 2,000 landmarks and a cosine kernel to approximate the spectral embedding in 40 dimensions.
* KMeans runs in this compact spectral space for k between 4 and 15, with metrics recorded for each k (silhouette, CH, DB, max cluster diameter).

In the v2.4.2 run I report, the chosen solution has **k = 14** clusters over all 47,491 questions. Global silhouette is modest (≈0.15) for such a noisy domain, but:

* Many clusters have **very high mean silhouette** (0.7–0.9) and strong tag purity.
* Two large “basin” clusters absorb the generic symptom drift.
* Average intra-cluster cosine similarity ≈ 0.67, mean inter-centroid cosine ≈ 0.00 and average tag purity ≈ 0.62.

Clusters line up with intuitive themes such as:

* Musculoskeletal and arthritis complaints
* Sexual health, STIs, and surgery/blood-test questions
* Pregnancy and period timing
* Newborn/baby care
* General fever/infection and flu/vaccination
* Medication and treatment questions
* Anxiety/depression and panic disorders
* Insurance/Medicare/ACA and coverage issues
* Skin, acne, rash and dermatology topics

From my perspective, the story is:

* **v1.1.2–v1.2.2:** good top-level taxonomy, weaker semantics, mega-clusters.
* **v1.4.2:** same LSA space but better cleaned and sliced with spectral clustering.
* **v2.1.2–v2.2.2:** excellent micro-topics via OPTICS, but low coverage and slower runs.
* **v2.3.2–v2.4.2:** best trade-off—semantically meaningful, tag-aligned clusters that are scalable, fully covering, and easy to explain.

---

## Project Overview

I treat this as a clustering-first project with a strong operational flavor. The dataset is a large set of short, informal medical questions plus a multi-label tag field. The end goal is a taxonomy that:

* makes sense to clinicians and product owners,
* can be reused for routing, reporting, and later supervised models,
* and behaves predictably as new data arrives and distributions drift.

To get there, I iterate through seven versions. Each version changes:

* the **NLP pipeline** (tokenization, POS filtering, TF-IDF vs BERT, tag handling),
* the **clustering backend** (KMeans, spectral, OPTICS, Nyström + MiniBatchKMeans),
* or both.

The guiding question for each iteration is: *“Would this clustering actually help a real system organize and navigate these questions?”*

---

## Problem Statement

Given tens of thousands of short, noisy medical questions with a large and messy tag universe, the problem is to produce clusters that:

* align with real medical themes, not just word-frequency artifacts,
* stay explainable enough for domain experts to review and name,
* handle the long tail of rare conditions and paraphrases without exploding k,
* and can be recomputed and audited as new questions arrive.

There is a tension between:

* **simple, top-level clusters** (great for dashboards and KPIs), and
* **fine-grained micro-topics** (great for intent libraries and future retrieval/RAG).

In this project:

* The **v1.x** line leans toward top-level clusters in a TF-IDF/LSA space.
* **v2.1.2–v2.2.2** show what density-based clustering can do for micro-topics.
* **v2.3.2–v2.4.2** aim for a middle ground: a scalable spectral manifold + KMeans that covers all data and still respects micro-structure.

---

## Data Exploration

After preprocessing, I end up with:

* **47,491 rows** after POS-based cleanup (from 47,603; only ≈112 zero-information rows dropped),
* a TF-IDF vocabulary of up to ~40k terms (depending on `min_df`),
* and a sparse multi-hot tag matrix with ≈4k unique tags.

The distributions are very **heavy-tailed**:

* Tags like “pregnancy”, “period”, “pain” appear thousands of times.
* Many tags appear only a handful of times.

Cosine similarity histograms in both LSA and BERT+tags spaces show:

* dense **“islands”** corresponding to specific conditions or topics (e.g., shingles, hepatitis C, hypothyroidism, some medication regimes),
* and a broad low-similarity background corresponding to vague symptom, “could this be X”, and general worry questions.

This pattern explains later behavior:

* In v1.x KMeans, one or two **huge generic clusters** emerge alongside several more specialized ones.
* In v2.1.2–v2.2.2 OPTICS, the dense islands become clusters and the background becomes **noise**.

Outliers that turn into zero vectors or get filtered away tend to be extremely short prompts (“help pls”, “hi”) or lines fully stripped by POS filtering.

---

## Modelling

I treat clustering and retrieval as two sides of the same organizing problem.

* **v1.x line (1.1.2, 1.2.2, 1.4.2):**
  Start with classical TF-IDF + LSA + KMeans. The emphasis is on getting a stable, explainable taxonomy in a linear space and seeing how far simple KMeans can go in separating major medical themes.

* **Transition to v1.4.2:**
  Introduce POS-based cleaning and spectral clustering on the LSA representation. Clusters become less spherical, more manifold-aware, and easier to interpret via dendrograms.

* **v2.x line (2.1.2–2.4.2):**
  Switch to embedding-based pipelines that combine:

  * BERT sentence embeddings on POS-filtered text,
  * SVD-reduced tag vectors,
  * and consistent cosine geometry via scaling + L2 normalization.

  Within this family:

  * v2.1.2–2.2.2 use **OPTICS** to explore dense islands vs noise.
  * v2.3.2–2.4.2 use **Nyström spectral** + MiniBatchKMeans with k-scan.

Throughout, I rely on internal metrics and tag coherence measures to decide which versions are better for:

* **high-level routing** (balanced, fully covering clusters),
* vs **long-tail discovery** (islands with high purity but limited coverage).

---

## Algorithm and Evaluation Strategy

Algorithmically, each version follows the same macro pattern:

1. Build a feature space combining **content** (question text) and **weak supervision** (tags).
2. Reduce dimensionality where necessary.
3. Run a clustering algorithm suited to that geometry.
4. Evaluate cluster structure numerically and qualitatively.

### v1.x (LSA + KMeans / Spectral)

* Work in a 200D LSA space.
* v1.1.2–1.2.2 use KMeans with fixed k, experimenting with different k and initialization.
* v1.4.2 switches to spectral clustering on the same LSA features, adding k-scan and dendrograms for hierarchy.

### v2.x (BERT + Tags)

* Build a 100D fused feature space:

  * BERT(all-MiniLM-L6-v2) → PCA (≈80D),
  * tags → SVD (≈20D),
  * concatenate and L2-normalize.
* v2.1.2–2.2.2: OPTICS with cosine distance on normalized features, tuning `min_samples` and `min_cluster_size` to control noise vs cluster count.
* v2.3.2–2.4.2: Nyström spectral approximation:

  * sample landmarks,
  * compute K_mm and its eigenvectors,
  * approximate spectral coordinates for all points (40D),
  * run MiniBatchKMeans + k-scan in spectral space.

### Evaluation

For each version, I compute:

* **Global internal metrics:** silhouette, Calinski-Harabasz (CH), Davies-Bouldin (DB) in the space where clustering is done (LSA, fused 100D space, or 40D spectral space).
* **Cluster size stats:** min/max/mean/median/std of cluster sizes and noise points (for OPTICS).
* **Cosine-based structure metrics:**
  average intra-cluster cosine similarity, mean inter-centroid cosine similarity.
* **Tag-based coherence:**
  per-cluster dominant tag purity, tag entropy, mean intra-cluster Jaccard over tag sets.
* **Human-readable cluster summaries:**
  top TF-IDF terms, top tags, and example questions.

Within each family, I interpret metrics relatively:

* v1.x: good internal scores but visible mega-clusters in size stats.
* v2.1.2–2.2.2: excellent metrics on the labeled points, but most data marked as noise.
* v2.3.2–2.4.2: moderate global silhouette on a messy domain, high per-cluster silhouette for many clusters, strong tag purity, and full coverage.

---

## Data Preprocessing

All versions share a common preprocessing backbone, with later variants adding a more sophisticated NLP layer.

### Normalization

* Lowercase text.
* Normalize curly quotes and dashes (’→', “→", –→-).
* Remove non-alphanumeric symbols except `'` and `?` when helpful.
* Store this as `short_question_norm`.

### Tag Parsing

* Extract quoted tags from the raw `tags` column using a regex.
* Normalize whitespace, lowercase, and store as `tag_list`.
* Convert `tag_list` to a multi-label binary matrix with `MultiLabelBinarizer(sparse_output=True)`.

### POS-based filtering (mainly v1.4.2 and all v2.x)

* Run spaCy `en_core_web_sm` with NER/textcat disabled for speed.
* Keep tokens where:

  * POS ∈ {NOUN, PROPN}, or
  * lemma is in a whitelist of short medical tokens (`hpv`, `uti`, `flu`, `rbc`, `wbc`, etc.).
* Remove:

  * general + domain-specific stopwords (“today”, “year”, “time”, etc.),
  * numeric-like tokens,
  * very short tokens not in the whitelist.
* Store the result as `short_question_pos`.
* Drop rows where both `short_question_pos` is empty and `tag_list` is empty.

### Dimensionality Reduction

* **v1.x:**

  * TF-IDF (with n-grams) on normalized or POS-filtered text.
  * Optional tag features concatenated to TF-IDF.
  * TruncatedSVD to ≈200D (LSA space).

* **v2.x:**

  * BERT embeddings on `short_question_pos`.
  * PCA to ≈80D (explained variance ≈0.72).
  * Tag SVD to ≈20D (explained variance ≈0.24).
  * Concatenate to 100D.

### Scaling and Normalization

* StandardScaler (with_mean=True) where appropriate.
* Row-wise L2 normalization before OPTICS or spectral embedding so that cosine distance is meaningful.

---

## Model Architectures

The clustering “architectures” evolve in two phases.

### Phase 1 – LSA-based (v1.1.2, 1.2.2, 1.4.2)

* Text representation:

  * TF-IDF over normalized/POS-filtered text (unigrams + limited bigrams).
  * Optional tag one-hots or SVD-reduced tags.
  * TruncatedSVD → ~200D LSA space.

* Clustering:

  * v1.1.2–1.2.2: KMeans / MiniBatchKMeans with fixed k (e.g., 15 or 7), used as top-level taxonomies.
  * v1.4.2: Spectral clustering on the LSA representation; KMeans in eigenvector space, with k-scan and dendrograms.

These models are straightforward, relatively fast, and easy to explain, but they are limited to linear structures and still prone to one or two mega-clusters.

### Phase 2 – BERT + Tags (v2.1.2–2.4.2)

* Text representation:

  * Embed POS-filtered questions using `all-MiniLM-L6-v2`.
  * PCA → 80D with explained variance ≈0.72.
* Tag representation:

  * MultiLabelBinarizer + SVD → 20D with explained variance ≈0.24.
* Concatenate into a 100D dense vector per question.
* L2-normalize to keep cosine geometry consistent.

Clustering heads:

* **OPTICS (v2.1.2–v2.2.2)**

  * Run OPTICS with cosine distance on the 100D space.
  * Tune `min_samples` and `min_cluster_size` to trade off noise vs number of clusters.
  * Naturally identifies dense islands but yields huge noise sets when thresholds are conservative.

* **Nyström spectral + MiniBatchKMeans (v2.3.2–v2.4.2)**

  * Sample 2,000 landmarks.
  * Build landmark kernel K_mm with a cosine kernel + small ridge.
  * Compute eigenvectors/eigenvalues of K_mm.
  * Form a sparse S matrix of top-T similarities from each point to landmarks.
  * Compute spectral embedding Z ∈ ℝ^(N×40).
  * Run MiniBatchKMeans with k-scan (k = 4..15) on Z.
  * Choose k based on a tuple of metrics (sampled silhouette, CH, DB, max diameter).

---

## Training Configuration / Reproducibility

Even though all the runs are unsupervised, I treat configuration and runtime behavior as “training” that needs to be controlled and documented.

Key choices:

* Fixed random seeds (usually 42) for NumPy, KMeans, PCA/SVD, Nyström landmark sampling, and UMAP.
* For TF-IDF, bounded vocabulary size and `min_df` to avoid noise from ultra-rare grams.
* BERT embeddings computed in batches (e.g., 1,024 sentences per batch) and on GPU when available.
* PCA and SVD components chosen to keep a comfortable proportion of variance while keeping dimensions manageable for clustering.
* KMeans / MiniBatchKMeans with multiple `n_init` and reasonable batch sizes (4096) to stabilize centers without overspending runtime.
* OPTICS hyperparameters in v2.1.2–2.2.2 explicitly tuned for “dense island discovery” rather than full coverage.
* Detailed timing logs for each stage (CSV load, NLP preprocessing, vectorization, spectral embedding, clustering, metrics, UMAP, dendrogram).

For BERT-based pipelines (v2.1.2–2.4.2), the core configuration is:

* BERT: `sentence-transformers/all-MiniLM-L6-v2`,
* BERT PCA: 80 components (EV ≈ 0.721),
* Tag SVD: 20 components (EV ≈ 0.245),
* Nyström (v2.3.2–v2.4.2): `N_LANDMARKS=2000`, `TOP_T=64`, `SPECTRAL_EIGENVEC=40`,
* KMeans layer on Z: `k ∈ [4,15]`, `n_init=10`, batch_size=4096.

---

## Evaluation Strategy

Because the versions use different representation spaces (LSA vs fused BERT+tags vs spectral embedding), evaluation is layered rather than reduced to a single score.

Within each version, I compute:

* **Internal metrics** in the clustering space:

  * silhouette (often with cosine and sometimes sampled for scalability),
  * Calinski-Harabasz,
  * Davies-Bouldin.

* **Size and coverage metrics:**

  * cluster size distribution (min/max/mean/median/std),
  * for OPTICS: counts of clustered points vs noise.

* **Cosine structure metrics:**

  * average intra-cluster cosine similarity (cohesion),
  * mean inter-centroid cosine similarity (separation).

* **Tag coherence metrics:**

  * dominant tag purity per cluster,
  * entropy over tag counts,
  * mean intra-cluster Jaccard similarity between tag sets.

* **Human inspection:**

  * top TF-IDF terms per cluster,
  * top tags per cluster,
  * 3–5 example questions per cluster (especially for outliers and low-silhouette clusters).

Interpretation is relative:

* v1.x: internal metrics are decent but cluster size stats reveal mega-clusters.
* v2.1.2–2.2.2: cluster-internal metrics are very strong, but coverage is poor because most points are noise.
* v2.4.2: global silhouette is modest (≈0.15) but:

  * many clusters have silhouette ≈0.7–0.9,
  * tag purity averages ≈0.62,
  * intra-cluster cosine is high (≈0.67),
  * inter-centroid cosine is low (≈0.00),
  * and coverage is 100%.

For this application, that trade-off is much more valuable than a high silhouette on a tiny labeled subset.

---

## Processing Pipeline

All experiments share the same backbone:

1. **Load & clean**

   * Read CSV from Google Drive.
   * Normalize `short_question`, parse tags, fill missing values.
   * Apply POS-based cleaning to create `short_question_pos`.
   * Drop rows with no useful text and no tags.

2. **Feature construction**

   * v1.x:

     * TF-IDF on normalized or POS-filtered text.
     * Optional tag features.
     * TruncatedSVD → 200D LSA.

   * v2.x:

     * BERT embeddings on `short_question_pos`.
     * PCA → 80D.
     * Tag SVD → 20D.
     * Concatenate to 100D, scale, L2-normalize.

3. **Clustering**

   * v1.1.2–1.2.2: KMeans with fixed k in LSA space.
   * v1.4.2: spectral clustering + KMeans on eigenvector space, with k-scan.
   * v2.1.2–2.2.2: OPTICS with cosine distance on 100D fused space.
   * v2.3.2–2.4.2: Nyström spectral embedding (40D) + MiniBatchKMeans k-scan.

4. **Diagnostics**

   * Compute internal metrics, size stats, tag coherence.
   * Generate UMAP 2D scatter and dendrogram (for spectral variants).
   * Produce k-vs-metric plots for the Nyström+KMeans runs.

5. **Outputs**

   * Save a clustered assignments CSV with:

     * original question,
     * POS-filtered text,
     * tags,
     * cluster ID,
     * cluster hints from top terms and top tags.
   * Save per-cluster summaries and global model-selection metrics as CSV.
   * Save a timings JSON to understand runtime per stage.

---

## Conclusion

Looking across the seven versions:

* **v1.1.2–v1.2.2** provide a reliable, explainable top-level taxonomy in an LSA space. They are ideal for initial stakeholder discussions, but mega-clusters limit their use for routing and detailed analytics.
* **v1.4.2** shows that POS filtering and spectral clustering in LSA space can sharpen themes and reveal a hierarchy of topics, but it is still tied to TF-IDF representations.
* **v2.1.2–v2.2.2** push density-based clustering as far as possible: these OPTICS runs provide an upper bound on how sharp and clinically coherent islands can be, at the cost of treating most questions as noise.
* **v2.3.2–v2.4.2** deliver a scalable spectral manifold + KMeans approach. Among them, **v2.4.2** is the most “production-ready”: same strong BERT+tags representation, fully covering clusters, consistent cosine-based geometry, and rich diagnostics.

If I have to choose one version for deployment today, I pick **v2.4.2**: the Nyström + MiniBatchKMeans pipeline balances semantic coherence, tag alignment, runtime, and explainability in a way that fits real operational needs.

---

## Lessons Learned

A few lessons are clear after working through these versions:

1. **Representation beats clever clustering.**
   Moving from raw TF-IDF to POS-filtered text helps, but the big jump happens when BERT embeddings and tag features are combined. After that, most clustering differences are about how to slice an already good space.

2. **KMeans is surprisingly strong.**
   In both LSA and spectral spaces, KMeans gives stable, interpretable top-level categories. The main issues are mega-clusters and granularity, not algorithm failure.

3. **Density-based methods are powerful but unforgiving.**
   OPTICS (v2.1.2–2.2.2) produces some of the most coherent medical topics in the corpus, but at the price of labeling the majority of points as noise. That’s great for analysis and canonical examples, but not for a system that needs a label for every question.

4. **Nyström + KMeans is a practical middle ground.**
   The v2.4.2 spectral Nyström approximation keeps a graph-based notion of similarity while remaining scalable. Combined with MiniBatchKMeans and k-scan, it yields 14 clusters that cover the full dataset, are reasonably compact (intra-cluster cosine ≈ 0.67), and align well with tags (purity ≈ 0.62).

5. **High silhouette alone can be misleading.**
   Density-based runs have fantastic silhouette scores on their labeled subsets, but that hides the fact that most points are labeled as noise. For this task, moderate silhouette with full coverage and strong tag coherence (as in v2.4.2) is more useful than excellent silhouette on a tiny fraction of the data.

---

## Limitations and Future Work

Remaining limitations:

* **Metric comparability across spaces.**
  Silhouette and CH in LSA space are not directly comparable to the same metrics in the 40D spectral space. A more rigorous comparison would recompute cohesion and separation for all versions in a shared embedding (e.g., the 100D BERT+tags space).

* **Scale of human evaluation.**
  Manual audits of clusters are still limited. A more systematic evaluation with multiple raters and inter-annotator agreement would give stronger evidence of topic coherence and naming quality.

* **No integrated retrieval layer yet.**
  The current work stops at clustering. A natural next step is to add a retrieval stack on top of v2.4.2: dense similarity search within each cluster plus reranking to support answer reuse or RAG.

* **Domain drift and retraining.**
  Medical queries will drift over time. A production system would need:

  * scheduled retraining of embeddings and clusters,
  * drift monitors over embedding distributions and cluster size trends,
  * alerts when new themes appear or some clusters grow abnormally.

* **From clusters to intents.**
  Clusters are unsupervised proto-intents. To make them operational:

  * freeze v2.4.2 clusters as a starting taxonomy,
  * collect names/guidelines from clinicians or product owners,
  * and train supervised intent models that predict these labels for new questions.

Overall, the comparison across v1.1.2–v2.4.2 gives a clear map of the design space:

* **v1.x** = reliable, explainable TF-IDF/LSA taxonomies.
* **v2.1.2–v2.2.2** = upper bound of density-based micro-topics.
* **v2.4.2** = the version that balances all constraints and is ready to sit behind a real medical chatbot pipeline.

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


