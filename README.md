# Unsupervised Clustering of Medical Chatbot Questions with TF-IDF, BERT and Spectral Methods

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
* Average intra-cluster cosine similarity ≈ 0.67, mean inter-centroid cosine ≈ 0.00, and average tag purity ≈ 0.62.

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



## Comparison between v1 and v2


# Reference

[1] Scikit-learn Developers, "KMeans clustering," *scikit-learn documentation*, 2024. [Online]. Available: [https://scikit-learn.org/stable/modules/clustering.html#k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)

[2] Scikit-learn Developers, "Truncated SVD (LSA)," *scikit-learn documentation*, 2024. [Online]. Available:
[https://scikit-learn.org/stable/modules/decomposition.html#lsa](https://scikit-learn.org/stable/modules/decomposition.html#lsa)

[3] Scikit-learn Developers, "OPTICS clustering," *scikit-learn documentation*, 2024. [Online]. Available:
[https://scikit-learn.org/stable/modules/clustering.html#optics](https://scikit-learn.org/stable/modules/clustering.html#optics)

[4] HDBSCAN Authors, "HDBSCAN: Hierarchical density-based clustering," *Official Documentation*, 2024. [Online]. Available:
[https://hdbscan.readthedocs.io/](https://hdbscan.readthedocs.io/)

[5] UMAP-learn Developers, "UMAP: Uniform Manifold Approximation and Projection," *Official Documentation*, 2024. [Online]. Available:
[https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)

[6] Explosion AI, "spaCy: Industrial-strength NLP," *spaCy Documentation*, 2024. [Online]. Available:
[https://spacy.io/](https://spacy.io/)

[7] HuggingFace, "Sentence Transformers - all-MiniLM-L6-v2 model card," *HuggingFace Model Hub*, 2024. [Online]. Available:
[https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

[8] HuggingFace, "BERT base uncased model card," *HuggingFace Model Hub*, 2024. [Online]. Available:
[https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)

[9] M. Grootendorst, "BERTopic: Neural Topic Modeling with class-based TF-IDF," *BERTopic Documentation*, 2024. [Online]. Available:
[https://maartengr.github.io/BERTopic/](https://maartengr.github.io/BERTopic/)

[10] FAISS Team (Meta AI), "FAISS: Facebook AI Similarity Search," *GitHub Repository*, 2024. [Online]. Available:
[https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

[11] nltk.org, "Punkt Tokenizer Models," NLTK Documentation, 2024. [Online]. Available:
[https://www.nltk.org/](https://www.nltk.org/)

[12] StackOverflow Community, "Techniques for TF-IDF, vectorization, and clustering in Python," *StackOverflow Discussions*, accessed 2024. [Online]. Available:
[https://stackoverflow.com/](https://stackoverflow.com/)





>
> 
