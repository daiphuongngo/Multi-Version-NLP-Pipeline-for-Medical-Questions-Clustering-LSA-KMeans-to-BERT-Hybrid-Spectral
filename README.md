# Multi-Version-NLP-Pipeline-for-Medical-Questions-Clustering-LSA-KMeans-to-BERT-Hybrid-Spectral

![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master of Liberal Arts (ALM), Data Science**

## **CSCI E-108 Data Mining, Discovery and Exploration**

## Professor: **Stephen Elston, PhD**, Princeton University, Principal Consultant, Quantia Analytics LLC

## Name: **Dai Phuong Ngo (Liam)**

# ABSTRACT

My work compares eight clustering pipelines (v1.1-v1.4, v2.1-v2.4) on an approximate 47.5k-row corpus of short medical chatbot questions with associated tags. My goals were: (1) to build an explainable, business-ready taxonomy for routing and analytics, and (2) to understand how far I can push unsupervised structure before needing supervised intent models. I started with TF-IDF/LSA + KMeans baselines, moved to POS-filtered representations and then switched to BERT embeddings plus tag features. On the clustering side, I experimented with KMeans, spectral clustering, density-based OPTICS over a cosine space, and finally a scalable Nyström spectral embedding followed by MiniBatchKMeans. Across versions I found that simple LSA + KMeans (v1.x) gives clean, board-friendly top-level categories; OPTICS (v2.1-v2.3) uncovers extremely tight but sparse topic islands. And the final Nyström + KMeans pipeline (v2.4) strikes the best balance between semantic coherence, tag alignment, runtime and full coverage. I close with an integrated picture of how these clusters can support labeling, routing and future retrieval or RAG-style answering.

# Executive Summary

I ran eight clustering variants on the same inputs:

> v1.1-v1.2 (TF-IDF + LSA + KMeans)

Early baselines using bag-of-words and tags. These runs already produced a recognizable medical taxonomy, such as pregnancy, skin, dental, mental health, etc.—but they also suffered from one or two "mega-clusters" mixing general pain/symptom questions.

> v1.3-v1.4 (TF-IDF + LSA + Spectral Clustering)

Once I added stricter preprocessing (POS-based noun/proper-noun filtering) and moved to spectral clustering, clusters became more semantically focused and less spherical. v1.4 in particular, with k-scan and dendrograms, gave a good top-level view but still relied on TF-IDF features.

> v2.1-v2.3 (BERT + Tags + OPTICS)

In v2.x I switched to a joint representation: Sentence-Transformer embeddings for short_question_pos plus SVD-compressed tag vectors. OPTICS with a cosine metric discovered very dense, clinically coherent micro-topics (for example, shingles, hepatitis, thyroid), but labeled only a small fraction of the dataset and treated the rest as noise. v2.3 is the most polished of these density-based runs: 4 extremely tight clusters, 46,650/47,491 points as noise, very strong internal metrics on the labeled subset but not operationally convenient.

> v2.4 (BERT + Tags → Nyström Spectral + KMeans)

v2.4 keeps the same BERT+tags feature design as v2.3 but replaces OPTICS with a scalable Nyström spectral embedding (40D) followed by MiniBatchKMeans with a k-scan. The chosen solution has k = 15 clusters, covers all 47,491 questions, improves average tag purity (≈0.682 vs 0.450 in v2.3) and reduces clustering runtime by roughly an order of magnitude (≈272s total vs ≈3,138s). The resulting clusters line up with intuitive themes: pregnancy and fertility, newborn/baby care, skin and rash, dental questions, general infection/fever, insurance and Medicare/ACA, anxiety and depression, surgery and orthopedics, etc.

From my perspective, the story is that:

> v1.x: good top-level taxonomy, weaker semantics.

> v2.1-v2.3: excellent micro-topics, low coverage, slow.

> v2.4: best trade-off—semantically meaningful, tag-aligned clusters that are fast enough and easy to explain to non-technical stakeholders.


# Project Overview

I will treat this as a clustering-only project with a strong operational flavor (for now). The dataset is a large set of short, informal medical questions plus a multi-label tag field. What I wanted in the end was a taxonomy that makes sense to clinicians or product owners, clusters that can be reused for routing, reporting or later supervised models and an understanding of how different unsupervised methods behave on this kind of noisy, high-cardinality medical corpus.

To get there, I iterated through eight versions. Each version changed either the NLP pipeline (tokenization, POS filtering, TF-IDF vs BERT, tag handling), or the clustering backend (KMeans, spectral, OPTICS, Nyström + KMeans) or both.

Throughout, my north star was: "Would this clustering actually help the company organize and navigate these questions in a real system?"

# Problem Statement

Given tens of thousands of short, noisy medical questions with a large and messy tag universe, I needed to produce clusters that align with real medical themes, not just word frequency artifacts. Then I keep the clusters explainable enough that a domain expert could review and name them. And I also handle the long tail of rare conditions and paraphrases without exploding the number of clusters, as well as build a pipeline that can be rerun and audited as new questions arrive and distributions drift.

The key tension is between simple, top-level clusters (great for slides and dashboards) and fine-grained micro-topics (great for intent libraries and future retrieval/RAG). The v1.x line leans toward the first, v2.1-v2.3 toward the second and v2.4 tries to meet in the middle.

# Data Exploration

After preprocessing, I ended up with 47,491 rows after POS-based cleanup (starting from 47,603; a small number of zero-information rows dropped), a TF-IDF vocabulary of up to ~40k terms, depending on min_df, a multi-hot tag matrix with ~4k unique tags.

The distributions are very heavy-tailed. Some tags (for example, "pregnancy", "period", "pain") appear thousands of times. Many tags appear only a handful of times.

Cosine similarity histograms in both LSA and BERT spaces show dense "islands" for very specific topics (for example, shingles, hepatitis C, hypothyroidism), a broad low-similarity background for generic symptom and "what is this" questions.

That pattern explains later behavior when KMeans tends to form one or more large generic clusters plus several specialized ones. Density methods like OPTICS naturally pick out the islands and treat the rest as noise.

I also inspected outliers in the rows that turned into zero vectors or got filtered away were usually extremely short prompts ("help pls", "hi", etc.) or lines where POS filtering removed everything.

# Modelling

In this project I treated clustering and retrieval as two sides of the same problem: organizing a large corpus of short medical questions into coherent groups while still being able to retrieve similar questions on demand. On the clustering side, I started with more classical TF-IDF + LSA + KMeans baselines (v1.1-v1.4) and then moved toward embedding-based pipelines that combine BERT, POS-filtered text, and tag information (v2.1-v2.4). The earlier versions were mainly about getting a stable, explainable taxonomy with a fixed number of clusters and seeing how far I could push purely linear structure in LSA space. The later versions were more ambitious: they incorporated sentence embeddings, POS-driven content filtering, and multi-label tag signals to build a denser feature space and then experimented with different clustering logics, including OPTICS and a Nyström-based spectral approximation with a KMeans sweep on the spectral embedding. Throughout, I used internal metrics and tag coherence measures to decide which versions are better suited for high-level routing versus long-tail discovery.

# Data Preprocessing

All versions share the same basic preprocessing backbone, and then the later versions add a more sophisticated NLP layer on top. I start by loading the train_data_chatbot.csv file, making sure that short_question and tags are present and filling any missing text with empty strings. I normalize the raw question text by lowercasing it, normalizing curly quotes and dashes and stripping out non-alphanumeric characters except for apostrophes and question marks. This gives me a short_question_norm column that is clean enough for tokenization and TF-IDF. Because the dataset stores tags as a single string with quoted tag values, I parse them into a proper tag_list using a regular expression that extracts everything inside single quotes, normalizes whitespace and lowercases the result. From there, the v2.x versions add a POS-based layer: I run spaCy (en_core_web_sm) over the normalized questions and keep only lemmas that are nouns or proper nouns, with some special allowances for short medical tokens like "hpv" or "rbc". I also remove generic stopwords and domain-specific filler terms (like "today", "year", "time") and drop purely numeric-like tokens. This gives me a short_question_pos column that focuses on disease names, body parts, drugs and other medically meaningful nouns and I then filter out rows that have neither useful POS-filtered text nor tags. I also compute simple descriptive features like character length, token counts and tag counts for exploratory plots.

I relied on a shared preprocessing base, with some tightening over time. Here is an outline how I would do in my Data Preprocessing phase:

* Normalization

  * Lowercasing.
  * Normalizing quotes and dashes: `’ → '`, `– → -`, etc.
  * Removing non-alphanumeric symbols except `'` and `?` when helpful.

* POS filtering (v1.3 onward, fully stabilized by v1.4 and v2.x)

  * spaCy `en_core_web_sm`, with NER and textcat disabled for speed.
  * Keep tokens where:

    * POS ∈ {NOUN, PROPN}, or
    * lemma in a small whitelist of short medical tokens (e.g., `hpv`, `uti`, `flu`).
  * Strip stopwords (general + domain-specific), numeric-like tokens, and very short tokens not in the whitelist.

* Tags

  * Parse tags using a quoted pattern, normalize, and keep them as lists of strings.
  * Convert to multi-label binary matrix via `MultiLabelBinarizer(sparse_output=True)`.
  * Reduce dimensionality with TruncatedSVD for all v1.3+ and v2.x runs.

* Dimensionality reduction

  * v1.x:

    * TruncatedSVD(200) on TF-IDF(+tags) → LSA space.
  * v2.x:

    * PCA on BERT embeddings (80D in v2.3/v2.4).
    * SVD on tag matrix (20D).
    * Concatenation → 100D.

* Scaling and normalization

  * StandardScaler (with_mean=True) on reduced features.
  * Row-wise L2 normalization before OPTICS or spectral embedding.
 
# Model Architectures

The clustering architectures themselves evolved in two phases. In the v1.1-v1.4 series, I stick to a classical linear representation: tf-idf on the normalized text (with unigrams and a limited number of features) concatenated with a multi-hot encoding of the tags, followed by a truncated SVD to reduce the combined sparse matrix to 200 latent dimensions. All variants in this family are KMeans in that 200-dimensional LSA space; they differ mainly in k (for example, 15 in the more fine-grained taxonomy vs. 7 for a more executive summary) and in whether I use standard KMeans or MiniBatchKMeans with additional refinements. These models are straightforward, fast enough for tens of thousands of rows, and easy to explain to non-technical stakeholders.

In the v2.1-v2.4 series, the architecture becomes more "modern" and hybrid. I embed the POS-filtered questions using all-MiniLM-L6-v2, optionally normalize the embeddings to unit length, and then apply PCA down to 80 dimensions while keeping track of the explained variance (around 0.72 in the runs I report). In parallel, I project the tag multi-label matrix into a 20-dimensional SVD space, capturing roughly a quarter of the tag variance. I then horizontally stack these two dense blocks to form a 100-dimensional feature vector per question, which mixes semantic content and weak supervision from tags. The clustering head differs by version: in v2.1-v2.3 I use OPTICS with cosine distance directly in this space, tuning min_samples and min_cluster_size to control how much noise I allow versus how many clusters I discover. In v2.4 I introduce a scalable spectral step: I approximate the spectral embedding with a Nyström method based on 2,000 landmarks and a cosine kernel, then run MiniBatchKMeans over this 40-dimensional spectral embedding and scan k between 4 and 15 to choose the best solution by sampled silhouette plus CH and DB indices.

I can summarize the architecture choices like this:

* **KMeans family (v1.1-v1.3, and KMeans layer inside v2.4)**

  * Standard KMeans / MiniBatchKMeans in a low-dimensional, roughly linear space (LSA or spectral).
  * v1.x: used for the top-level taxonomy.
  * v2.4: used on spectral coordinates after Nyström embedding.

* **Spectral clustering (v1.3-v1.4)**

  * Build an affinity matrix (implicitly via sklearn).
  * Compute Laplacian eigenvectors.
  * Run KMeans on the eigenvector space.
  * v1.4 adds k-scan to find a sweet spot in cluster granularity.

* **OPTICS (v2.1-v2.3)**

  * Density-based clustering using reachability, ordering, and a cluster extraction step.
  * Uses cosine distance on normalized BERT+tags space.
  * Naturally identifies dense islands and marks the rest as noise.

* **Nyström spectral + MiniBatchKMeans (v2.4)**

  * Approximate spectral embedding using:

    * random landmark sampling,
    * dense eigen decomposition on K_mm (landmarks),
    * sparse S matrix for point-to-landmark similarities (topT).
  * Embed all points into r = 40 eigenvector coordinates.
  * Run MiniBatchKMeans + k-scan on this compact representation.

# Evaluation Strategy

Because I have multiple representation spaces and several clustering algorithms, evaluation has to be layered rather than reduced to a single score. Within each version, I compute classical internal metrics in the space where the clustering actually happens. For KMeans in LSA space (the v1.x line), I use silhouette, Calinski-Harabasz, and Davies-Bouldin directly on the 200-dimensional representations and combine those with basic stats on cluster size (min, max, mean, and standard deviation) to see whether there are any mega-clusters or pathological tiny groups. For OPTICS and the Nyström spectral + KMeans pipeline (v2.1-v2.4), I compute the same metrics in the underlying 100-dimensional fused space or in the 40-dimensional spectral embedding, sometimes using a sampled silhouette for scalability. To understand cluster structure, I also compute the average intra-cluster cosine similarity and the mean inter-centroid cosine similarity; the former acts as a cohesion proxy and the latter as a separation proxy. A big part of my evaluation is tag-driven: for each cluster, I look at the distribution of tags, compute the dominant-tag purity, the entropy over tags, and a mean intra-cluster Jaccard similarity between tag sets. These metrics tell me whether clusters are mixing unrelated intents or capturing clearly focused themes. Finally, I always inspect the top TF-IDF terms and a few representative questions per cluster by hand. In particular, I read through the outlier clusters and the ones with unusually low or high silhouette scores to understand what the model is doing, instead of relying purely on numeric criteria.

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
