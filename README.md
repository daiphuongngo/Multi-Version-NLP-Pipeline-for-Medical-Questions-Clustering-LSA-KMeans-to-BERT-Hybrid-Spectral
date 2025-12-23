# From Clusters to Retrieval: Hybrid BERT-Based Taxonomy and Similarity Search for Medical Chatbot Questions

![Harvard_University_logo svg](https://github.com/user-attachments/assets/cf1e57fb-fe56-4e09-9a8b-eb8a87343825)

![Harvard-Extension-School](https://github.com/user-attachments/assets/59ea7d94-ead9-47c0-b29f-f29b14edc1e0)

## **Master of Liberal Arts (ALM), Data Science**

## **CSCI E-108 Data Mining, Discovery and Exploration**

## Professor: **Stephen Elston, PhD**, Princeton University, Principal Consultant, Quantia Analytics LLC

## Name: **Dai Phuong Ngo (Liam)**

## Youtube:

[https://youtu.be/kyI9Hc2sSP0](https://youtu.be/kyI9Hc2sSP0)

## Abstract

My work compares six clustering pipelines (v1.1.3, v1.2.3, v1.4.3, v2.1.3, v2.3.3, v2.4.3) and two retrieval/reranking pipelines (Similarity v1 and v2) on an approximately 47.5 k-row corpus of short medical chatbot questions with associated tags. The goals are: 1) to build an explainable, business-ready taxonomy for routing and analytics, 2) to understand how far unsupervised structure can go before supervised intent models become necessary and 3) to design a similarity search and reranking layer that can reuse those representations for FAQ reuse, and future RAG-style answering.

I begin with BERT + tags + KMeans baselines (v1.1.3 and v1.2.3), add a graph-based spectral clustering variant (v1.4.3), then refine the representation space with PCA (v2.1.3) and finally apply a scalable manifold-aware clustering using a Nyström spectral embedding plus MiniBatchKMeans (v2.3.3, v2.4.3). On the clustering side, I test how spherical clustering, manifold clustering, and spectral manifold partitioning behave in this medical question space. On the retrieval side, I build two versions. Similarity Search & Reranking v1 uses dense semantic similarity plus tag Jaccard to rank neighbors. Similarity Search & Reranking v2 extends this with a hybrid architecture combining semantic cosine, BM25 lexical scores, and tag overlap in a blended scoring function. Across versions, I find that the Nyström + KMeans pipeline (v2.4.3), when combined with the hybrid retrieval stack, offers the best balance of semantic coherence, tag alignment, runtime efficiency, full coverage and retrieval quality. I conclude with an integrated view of how this taxonomy and similarity stack can support labeling, routing, retrieval, and future RAG-style answering.

---

## Executive Summary

I ran six main clustering variants on the same dataset and tags, and then layered two retrieval/reranking pipelines on top of the BERT + tags feature space used in the v2 line.

### v1.1.3 and v1.2.3 (BERT + tags + KMeans)

These early baselines embed POS-filtered questions using BERT(all-MiniLM-L6-v2) and reduce tags with SVD, concatenating both into a high-dimensional space. v1.1.3 uses a default KMeans; v1.2.3 refines that with a k-scan over cluster counts and thorough metric logging. Both produce a recognizable medical taxonomy—clusters for pregnancy/menstrual issues, vaccines, sexual health, musculoskeletal pain, mental health, medication questions, insurance and more—but also tend to produce one, or two very broad clusters mixing general symptom questions. Internal metrics (silhouette, intra-cluster cosine) are modest, and cluster size imbalance is significant. At this stage retrieval remains implicit: nearest-neighbor search in the same space is possible but no formal reranking or retrieval stack exists.

### v1.4.3 (BERT + tags + Spectral Clustering / Graph Embedding)

In v1.4.3 I keep the BERT + tags representation but switch clustering from KMeans to a graph-based spectral clustering over a similarity graph. This shifts the partition from spherical clusters to clusters aligned with the data manifold. With a k-scan and a dendrogram on cluster centroids, v1.4.3 gives a more nuanced, hierarchical view of major medical themes. However, results remain tied to the same embedding, and retrieval remains informal, without a dedicated reranker.

### v2.1.3 (BERT + tags + PCA + KMeans)

v2.1.3 represents a refinement in embedding: I apply PCA on BERT vectors to compress semantic variance, then fuse with tag SVD vectors to create a lower-noise 100-dimensional feature space. KMeans is applied over this normalized space. Compared to v1.2.3, clustering becomes more stable, silhouettes improve modestly, cluster boundaries sharpen and numerical behavior (diameter, inter-centroid separation) becomes more predictable. This sets the stage for more advanced manifold-aware clustering.

### v2.3.3 and v2.4.3 (BERT + tags → Nyström Spectral Embedding + MiniBatchKMeans)

In v2.3.3 and v2.4.3 I combine the mature BERT + tags embedding with a scalable spectral embedding via the Nyström method. I sample landmarks, build a cosine kernel approximate manifold, embed all points into a compact manifold space (e.g. 40 dimensions), then run MiniBatchKMeans over a k-scan. This approach produces full coverage (every question assigned to a cluster), improved geometry-aware partitioning, and more balanced cluster sizes. The v2.4.3 configuration yields many tight, semantically and tag-coherent clusters, while limiting over-large generic clusters. Cluster interpretability, silhouette, intra-cluster similarity, and tag purity reach levels that make this version production-ready.

### Similarity Search & Reranking v1 (Dense Semantic + Tags)

Similarity v1 is a retrieval stack built on the v2 representation. It uses dense semantic embeddings (BERT + PCA) for neighbor search, then reranks results based on a blend of semantic cosine similarity and tag Jaccard overlap. This design is simple and aligned with clustering geometry, but it has limitations with rare or exact-phrase queries where lexical overlap or specialized terms matter.

### Similarity Search & Reranking v2 (Hybrid Semantic + BM25 + Tags)

Similarity v2 improves on v1 by adding a BM25 lexical index over normalized question text. Candidates are drawn from the union of semantic neighbors and BM25 lexical matches. Reranking uses a blended score combining semantic similarity, tag overlap, and normalized BM25 score. This hybrid design preserves semantic generalization while capturing exact-phrase matches and rare-term relevance. In trials, v2 recovers many relevant neighbors missed by pure semantic retrieval, while maintaining coherence and tag alignment.

From my experiments: v1.1.3 and v1.2.3 give a rough, board-friendly taxonomy; v1.4.3 begins to incorporate manifold structure; v2.1.3 refines the representation; v2.3.3 and v2.4.3 produce a robust, scalable taxonomy; and in tandem, Similarity v2 yields the most balanced and practically useful retrieval stack.

---

## Project Overview

This project treats clustering as the backbone for understanding and organizing the medical question corpus—and builds a retrieval layer on top to support search, reuse, and future RAG workflows. The dataset is short, often noisy, informal medical questions with a multi-label tag field. The desired system must satisfy:

* human interpretability (clusters and categories make sense to domain experts),
* reusability (clusters and retrieval feed into routing, analytics, and intent definition),
* scalability (able to handle tens of thousands of questions efficiently), and
* adaptability (new questions can be inserted; clustering and retrieval pipelines can be rerun or updated).

I iterate over multiple clustering versions to explore trade-offs: representation noise, cluster shape (spherical vs manifold), coverage vs purity, stability vs flexibility. In parallel I develop retrieval pipelines that reuse the same semantic and tag representations, add lexical information, and provide reranking so that retrieval quality is aligned with both semantics and surface-level fidelity (useful for medical phrasing). The result is a taxonomy plus a retrieval stack that together enable routing, analytics, FAQ reuse, and future RAG-style answering.

---

## Problem Statement

Given a large, diverse corpus of short, informal medical questions plus messy, human-generated tag data, the challenge is to produce:

1. **Clusters** that:

   * reflect real medical themes rather than superficial lexical similarity,
   * remain explainable and reviewable by domain experts,
   * cover the long tail without exploding cluster count,
   * and can be recomputed and audited as new questions accumulate.

2. **A similarity search and reranking system** that:

   * reuses the same representation as clustering,
   * supports both semantic paraphrase matching and exact lexical matches,
   * leverages tag overlap as a weak supervision signal to preserve intent alignment,
   * and is fast enough for real-time retrieval (FAQ reuse, RAG pipelines, etc.).

There is an inherent tension between having simple, top-level clusters for dashboards and analytics; finer-grained micro-topics for intent libraries; and a flexible retrieval layer that can cross clusters. The architecture I build aims for a balanced compromise among these needs.

---

## Data Exploration

After preprocessing and POS-based cleanup, the dataset retains approximately 47,491 rows (out of ~47,600). The tag space remains broad, with certain tags appearing frequently (pregnancy, pain, flu, depression, etc.) and many tags only a handful of times—resulting in a heavy-tailed distribution.

When I embed the data into BERT+tags space and project pairwise cosine similarities, I observe a characteristic structure:

* Dense islands corresponding to well-defined medical topics or conditions (e.g. shingles, hepatitis C, thyroid disorders, certain drug regimes, discrete anatomical problems).
* A vast “background” cloud of low-similarity questions: generic symptoms, vague concerns, mixed complaints, or ambiguous phrasing.

This geometry explains much of the behavior I observe across clustering methods:

* In KMeans-based methods (v1.1.3, v1.2.3), large generic clusters emerge capturing much of the background, along with several more topical clusters.
* In spectral + manifold-based clustering (v2.3.3, v2.4.3), those dense islands tend to form coherent clusters, while the background is partitioned more evenly, limiting mega-cluster formation.
* Reranking and retrieval benefit from this structure: semantic + tag-based retrieval tends to surface relevant neighbors within islands; hybrid retrieval (with BM25) helps surface more lexical, or rare-term neighbors even if they inhabit the noisy background region.

Some rows drop out early in preprocessing: extremely short prompts (“help”, “hi”) or badly formatted questions that POS-filtering reduces to empty. These inevitably become problematic for both clustering and retrieval. I treat them as edge cases; they are either dropped or relegated to catch-all clusters, and manual review would likely be needed in production.

---

## Modelling

Clustering and retrieval are treated as two complementary sides of the organizing problem: clustering gives global structure, retrieval allows fine-grained navigation.

### Clustering line

**v1.1.3 and v1.2.3**

* Text normalized, POS-filtered, embedded with BERT all-MiniLM-L6-v2; tags vectorized via SVD; concatenated into high-dimensional fused vectors.
* KMeans clustering partitions the data into a fixed number of spherical clusters. v1.2.3 adds a k-scan and metric logging to pick a reasonable k.
* Outcome: a coarse taxonomy that covers the full dataset but tends to produce over-broad clusters.

**v1.4.3**

* Same representation, but clustering via graph-based spectral embedding: build a similarity graph, compute eigenvectors, cluster in that manifold space.
* A dendrogram of centroids offers a hierarchical view of macro-level topics.
* Outcome: a more manifold-aware taxonomy that improves topical coherence for many clusters, though some clusters remain broad.

**v2.1.3**

* Representation refined: BERT embeddings compressed via PCA (to ~80 dimensions), fused with tag SVD (20 dims), yielding a cleaner 100-dimensional space.
* KMeans over this normalized space produces more stable clusters, reducing representation noise and improving separation.
* Outcome: a more robust baseline clustering that improves over early v1.x runs.

**v2.3.3 and v2.4.3**

* Representation as in v2.1.3.
* Clustering via Nyström spectral approximation: sample landmarks, compute approximate spectral embedding (e.g. 40-D manifold), then apply MiniBatchKMeans with a k-scan.
* Outcome: full coverage clustering that respects data manifold, yields many tight, semantically and tag-coherent clusters, and avoids the mega-cluster problem seen earlier. v2.4.3 is the most refined and balanced version.

### Retrieval line

**Similarity Search & Reranking v1**

* Semantic tower: L2-normalized PCA-reduced BERT embeddings.
* Tag tower: tag sets per question.
* Candidate generation: nearest neighbors by semantic cosine.
* Reranking: blended score combining semantic cosine and tag Jaccard.
* Outcome: simple semantic + tag retrieval, aligned with clustering representation, but limited for rare lexical patterns.

**Similarity Search & Reranking v2**

* Adds a BM25 lexical index over normalized question text.
* Candidate generation: union of semantic neighbors and BM25 lexical top candidates.
* Reranking: blended score combining semantic cosine, tag Jaccard, and normalized BM25 score.
* Outcome: hybrid retrieval that balances semantic generalization, tag alignment, and lexical precision, improving recall for rare or exact-phrase queries while preserving semantic quality.

---

## Algorithm and Evaluation Strategy

Each clustering and retrieval version follows a structured, reproducible pipeline:

1. Build a shared feature space combining content and tags.
2. Optionally reduce dimensionality for stability and noise removal.
3. Run a clustering algorithm appropriate to the geometry (spherical or manifold).
4. Evaluate cluster structure both numerically (silhouette, intra/inter-cluster similarity, tag coherence) and qualitatively (cluster summaries, example questions).
5. For retrieval: build indexes (semantic, lexical, tag), generate candidates, rerank, and review neighbor quality qualitatively, optionally logging score components for analysis.

For clustering, I record global silhouette, per-cluster silhouette, cluster size distribution, intra-cluster similarity, inter-centroid similarity, and tag purity and entropy. For retrieval, I inspect sample queries and neighbor lists, compare semantic-only vs hybrid rankings, and verify that retrieval outputs align with medical relevance and tag consistency. In future iterations, a small labeled relevance set could allow computation of IR metrics (precision@k, recall@k, nDCG@k), but for now qualitative analysis suffices to guide design decisions.

---

## Conclusion

Across the evaluated versions:

* v1.1.3 and v1.2.3 provide a rough but usable taxonomy built with BERT + tags + KMeans.
* v1.4.3 explores manifold-aware clustering via spectral methods and delivers a more nuanced hierarchical taxonomy.
* v2.1.3 refines the embedding to a lower-noise space, improving cluster stability.
* v2.3.3 and v2.4.3 achieve the sweet spot: scalable clustering with manifold awareness and full coverage, yielding many coherent, medically meaningful clusters.
* Similarity Search & Reranking v1 and especially v2 transform the representation space into a retrieval stack suitable for FAQ reuse or RAG pipelines: semantic generalization, tag alignment, and lexical precision.

If I were to choose a single clustering and single retrieval configuration for deployment now, I would pick **v2.4.3 for clustering** and **Similarity Search & Reranking v2 for retrieval**. That combination balances semantics, tag alignment, runtime, coverage, explainability, and retrieval quality in a way that seems compatible with real-world usage for medical chat or documentation systems.

---

## Lessons Learned

* High-quality representation matters more than fancy clustering tricks. Once BERT + tags embedding is in place, clustering method tweaks produce incremental gains.
* KMeans remains a strong baseline, especially when paired with a clean, noise-reduced embedding space.
* Manifold-aware clustering (Nyström + spectral embedding) helps capture latent structure while avoiding over-broad, noisy clusters.
* Pure density-based clustering (OPTICS) can find very tight topic islands, but sacrifices coverage—less useful when full taxonomy is needed.
* A hybrid retrieval stack combining semantic, tag, and lexical signals works better than semantic-only, especially in domains with rare terms and variable phrasing (like medical questions).
* Internal clustering metrics (silhouette, intra-cluster similarity, tag purity) are helpful diagnostics, but final judgment benefits greatly from manual cluster inspection and retrieval tests.

---

## Limitations and Future Work

A few limitations remain:

* Metric comparability across different embedding spaces is imperfect; silhouette or cosine similarity in a spectral manifold is not strictly comparable to the same in a high-dimensional raw embedding space. To address this, a more formal evaluation framework (possibly with a small human-labeled relevance set) would help.
* Human evaluation of both clustering coherence and retrieval quality remains limited in scale. A more systematic review by domain experts would increase confidence in the taxonomy and retrieval stack.
* Retrieval evaluation remains qualitative. Next steps include building a small gold-standard set of question–answer or question–neighbor pairs to compute IR metrics such as recall@k, precision@k, and nDCG@k.
* The system may face drift over time: new medical topics, new slang, new formulations. A productionized system should include periodic re-embedding, re-clustering, and re-indexing, with monitoring of cluster stability, tag distribution shifts, and retrieval performance.
* Clusters and neighbor sets currently form a “proto-intent” space. To make them operational:

  1. fix the v2.4.3 taxonomy,
  2. use Similarity v2 to propose canonical exemplars,
  3. engage clinicians or product stakeholders to name and refine clusters,
  4. optionally build a supervised intent classification layer, with retrieval used to supplement answers via RAG or FAQ reuse when needed.

In sum, the revised pipeline from v1.1.3 through v2.4.3 and Similarity v1–v2 gives a practical, scalable, and explainable foundation for clustering and retrieval of medical chatbot questions.


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

# CLUSTERING

## v1.1.3

```
Rows after POS cleanup: 47491
modules.json: 100%
 349/349 [00:00<00:00, 82.0kB/s]
config_sentence_transformers.json: 100%
 116/116 [00:00<00:00, 28.7kB/s]
README.md: 
 10.5k/? [00:00<00:00, 2.37MB/s]
sentence_bert_config.json: 100%
 53.0/53.0 [00:00<00:00, 13.4kB/s]
config.json: 100%
 612/612 [00:00<00:00, 153kB/s]
model.safetensors: 100%
 90.9M/90.9M [00:00<00:00, 169MB/s]
tokenizer_config.json: 100%
 350/350 [00:00<00:00, 93.1kB/s]
vocab.txt: 
 232k/? [00:00<00:00, 22.0MB/s]
tokenizer.json: 
 466k/? [00:00<00:00, 68.1MB/s]
special_tokens_map.json: 100%
 112/112 [00:00<00:00, 30.5kB/s]
config.json: 100%
 190/190 [00:00<00:00, 48.1kB/s]
[Features] BERT(all-MiniLM-L6-v2) 384D + Tags→SVD(20D, EV=0.245) = 404D
[Shapes] Combined feature matrix: (47491, 404)
Searching for a good k (cosine metrics only)...
Model selection summary (first rows):
 k  silhouette  max_cluster_diameter
10    0.068711              1.326555
11    0.069707              1.290245
12    0.074148              1.290245
13    0.076263              1.310299
14    0.075475              1.291242
15    0.077149              1.310299
16    0.077758              1.310299
17    0.078318              1.310299
18    0.078135              1.304229
19    0.079234              1.281636
Selected k = 45 with silhouette (cosine) = 0.095
Saved model-selection table to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_model_selection.csv

[Plot] Saved silhouette vs k plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v1.1.2_silhouette_vs_k.png

[Plot] Saved max cluster diameter vs k plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v1.1.2_max_cluster_diameter_vs_k.png

=== Cluster Summaries ===

Cluster 0  (n=756)
 Top terms: exercise, muscle, weight, workout, program, type, activity, gym, heat, fat, aerobics, minute
 Top tags: exercise, muscle, walking, workout, weight, diet, coldness, running, pregnancy, hunger
 Examples (original text):
  - what exercise does not aggravate achilles tendonitis  i am trying to rest my achilles so i can return to playing tennis in the meantime i want to exercise to stay in shape what types of exercise can i do
  - what is resistance exercise
  - what is a weight loss exercise for people with arthritis in feet ankles and disc herniation in neck and low back  swimming irritates the neck and pain shoots down my arms walking in good sneakers pains my feet and ankles which have arthritis i can stretch but that is not enough to lose weight
  - how do you know what the best exercise routine is  i have had bariatric bypass surgery in 2010 i went from 340 to 232 and have a lot of access skin i also have fibromyalgia and arthritis that is not able to be controlled at the present time it is my desire to run a mini marathon but i do not even know where to begin on setting myself on the proper program i do not have the money to go to a trainer so need some direction please thank you  vicki
  - will exercising the vaginal muscles make my vagina tighter

Cluster 1  (n=809)
 Top terms: flu, vaccine, shot, injection, vaccination, swine, child, chickenpox, influenza, virus, baby, depo
 Top tags: flu, injection, vaccines, vaccination, swine flu, shingles, virus, pregnancy, coldness, chickenpox
 Examples (original text):
  - can you test positive from having the hep b vaccine
  - why would a rn choose not to get her kids a flu shot as the grandparent is there anything i can do
  - my son had dtap polio chicken pox and mmr vaccines now can barely move
  - what reactions are likely after an immunization
  - can allergy shots be used to treat asthma

Cluster 2  (n=2482)
 Top terms: tooth, child, baby, mouth, daughter, son, lip, water, wart, lice, mole, gum
 Top tags: tooth, pregnancy, baby, mouth, coldness, drinking, wart, tongue, vision, burn
 Examples (original text):
  - okay so i am 16 and i want to grow about 3 more inches if i smoke hookah once or twice will i grow to my goal height
  - what are some warning signs for pregnant women when they are exercising
  - can you use egg whites on a burn  i read an article that said you can use egg whites to sooth and help heal burns like if you burn yourself with fire but not real bad is this true
  - would braces close wide gap between front teeth
  - 3 yr old son has small specks of blood on face after napping

Cluster 3  (n=1402)
 Top terms: period, pregnancy, test, sex, pill, control, birth, symptom, cycle, cramp, condom, breast
 Top tags: period, pregnancy, pregnancy test, sexual intercourse, birth control, spotting, condom, breast, nausea, cramps
 Examples (original text):
  - could i possibly be pregnant  last period may 16th unprotected sex on june 9th supposed to start june 16th still have not if you think i am pregnant when should i take a test  side notes  feel as if i start but do not   i was throwing up at 2 am on saturday the 15th i was nauseous the rest of the day   light cramps  the guy i had sex with says he only has a 3 chance of getting someone pregnant  i have been tired lately i also have been having light heartburn i think  if anyone can help me it would be greatly appreciated
  - pregnant unprotected sex a week before period period came on time and heavy with bad cramps as usual reg 28 day 4 yrs i had unprotected sex a week before my period started he ejaculated awa from me but im worried a little bit may have got it before he pulled out my period came on the dot when it was supposed to get it and was heavy at first then to moderate with bad cramps like i normally have basically my period came on time and was normal in length flow and cramps my periods have been regular for years i do not know when i ovulate or my latueal phase what are my chances of being pregnant
  - i am on the pill and a condom was used pregnant  i have been on the pill for over 5 years i am pretty good with taking it on time but occasionally i will forget a day but immediately take it when i realize ive missed it last weekend saturday i was about an hour late taking my bc the next day i had sex he was wearing a condom a week later getting cramps and what not like a period i usually take the pills continuously but i am scared so i am going to let myself have a period hopefully i am about to start the sugar pills today
  - i have been off depo shot for over a year and have no period i recently began having pregnancy symptoms am i pregnant  i was on the depo shot for 2 years and have not gotten the shot since september 2011 i have still not had a period and have been having pregnancy symptoms my fiance and i use the withdraw method i realize that you can still get pregnant this way after over a year of not receiving the shot i do not think the symptoms could be attributed to the shot i had a pap smear 9 12 12 and it came back fine so could i be pregnant and if not what could be causing the pregnancy symptoms
  - i skip my periods with my birth control trinessa i am sexually active and very busy so if it is unnecessary for me to have a period u would rather not have it my boyfriend and i use both condoms and my birth control is there a higher chance of pregnancy if i skip my pills or is there even a chance of pregnancy i take my pill pretty religously but i have had times where i have missed the time to take it and have taken it hours later

Cluster 4  (n=955)
 Top terms: cancer, prostate, tumor, brain, treatment, colon, symptom, radiation, lymphoma, chemotherapy, biopsy, risk
 Top tags: cancer, prostate cancer, colon cancer, brain tumor, chemotherapy, prostate, breast cancer, lung cancer, biopsy, radiation surgery
 Examples (original text):
  - how to treat leiomyosarcoma and rectal cancer at the same time
  - i have a basal cell carcinoma what specialist should i consult
  - i have recently had breast surgery removal of calcium deposits with cancer cells
  - can hot showers a day cause cancer  i love to take hot showers and take three 5 10 minute showers a day   morning noon and evening is there a danger of cell damage or cancer from this practice also are three too many i like the water very hot is this bad our water in our city is chlorinated should i be concerned
  - abscessed tooth symptoms and oral cancer symptoms similar

Cluster 5  (n=1275)
 Top terms: chlamydia, hiv, infection, yeast, sex, vaginosis, discharge, hpv, partner, std, vagina, symptom
 Top tags: vagina, chlamydia, yeast infection, burn, sexual intercourse, hiv, bacterial vaginosis, bacterium, sexually transmitted disease, wart
 Examples (original text):
  - i was curious about anal used mothers sex toy didnt clean it at risk for stds do not think it was used in a while i was curious and i found a vibrator and i used it i put a condom on it but condom broke i got tested for chlamidia and ghonorea both negative do you think i am at risk for hiv or anything else  also i used other sorta home made toys over a year ago and i just got worried i could have done damage to my body have not had any negative symptoms and havnt used them since last year should i be worried everything is normal and during use nothing negative happened like bleeding of anything
  - can battery acid from a vibrator burn my genitals i thought i had a vaginal yeast infection but it turns out my vibrator was leaking battery acid and has burned my vaginal area included and up to my rectum what can i do to ease my pain
  - what causes bacterial vaginosis should i be worried if it this infection keeps popping up
  - every time i cough or sneeze i pass blood clots also through vaginal discharge and during intercourse
  - gonorrhea long term effects i have aged and look frail

Cluster 6  (n=726)
 Top terms: medication, drug, medicine, counter, prescription, med, treatment, blood, pressure, ibuprofen, allergy, effect
 Top tags: drug, over the counter, coldness, pregnancy, ibuprofen, pain, drug test, tuberculosis, depression, allergy
 Examples (original text):
  - i have been thinking about buying drugs online to save money is that a good idea
  - i am extremely agoraphobic despite help from my psychologist psychiatrist and medication why else can i do
  - is avinza a narcotic medication
  - does soma show up on a drug urinalysis
  - what medications and treatments are available to treat pagets disease of bone

Cluster 7  (n=850)
 Top terms: hand, arm, finger, shoulder, pain, wrist, elbow, thumb, rash, arthritis, numbness, joint
 Top tags: hand, arm, finger, pain, wrist, shoulder, rash, arthritis, coldness, swelling
 Examples (original text):
  - pain ring finger to the middle of arm before elbow for 45 days started after i held on to stop falling worse on lifting its not bad3 on 10 earlier it was more painful but now it does not hurt as much unless i type or write i have barely used my right hand for anything for the past month else the pain increases it starts hurting at one point about five fingers from my wrist but pain goes away completely if i press down on it it also hurts in the area below middle to pinkie i got an xray done already so nothing there shd i get an mri if so only for wrist or forearm also or shd anti inflammatories be enough thanks
  - i have got a wrist and palm injury in my right hand
  - i am kite surf instructor but have spine condition pain too much too handle
  - would the arm of a 1yr old swell immedietely if it were broken
  - why does the pain from frozen shoulder go down my arm and sometimes makes my hand tingle

Cluster 8  (n=489)
 Top terms: herpe, outbreak, partner, sex, sore, simplex, virus, area, bump, sign, symptom, type
 Top tags: genital herpes, herpes, vagina, virus, sexual intercourse, cold sore, lip, herpes simplex, sore, acne
 Examples (original text):
  - can herpes be spread by bed bugs if a person infected with herpes is bitten by a bed bug can another person bitten by the same bug get infected with herpes
  - what other than a yeast infection could it be if the medicine does not work and more symptoms start occuring  for a few months now i have had constant vaginal itching and burning and after using yeast infection medicine multiple times it still has not gone away and now blisters have begun forming after looking up genital herpes i have noticed that i have nearly all the symptoms for it but i have yet to have sexual activity could it still be possible or is it something else
  - how is herpes simplex treated
  - herpes transmission somewhat complicated question i have genital herpes but no outbreaks after the initial one my partner got it from me and has genital outbreaks we are wondering if we are able to spread it to each other in areas where we have not had outbreaks during shedding not active outbreak seems obvious that we could spread it around while there are lesions so we are careful during her outbreaks but the big mystery is  can we spread it to new areas during shedding    thanks a million we have been searching hard for the answer 
  - are there any ointments that can speed up the healing process of genitle herpes already on 3 day valtrex should i be worried if it dosent clear up by then

Cluster 9  (n=381)
 Top terms: shingle, pox, chicken, vaccine, pain, rash, risk, child, scalp, husband, area, treatment
 Top tags: shingles, pain, chickenpox, rash, vaccination, virus, blister, scalp, burn, injection
 Examples (original text):
  - can shingles occur in the scalp
  - i am 33 years old i need shingles vaccine but am i too young to have it
  - can i get shingles after having scarlet fever as a child  my mother is 80 and has been iin a lot of pain the last few weeks with her hip now she has a red rash very painful just appeared tonight it has a burning sensation is is possible she could have shingles should i take her to the er for treatment  thank you
  - is shingles contagious
  - i am 55 what  of getting shingles after getting shot as a side effect want to get the shot but not if i have a high risk of giving me shingles

Cluster 10  (n=793)
 Top terms: penis, sex, testicle, erection, vagina, std, tip, area, bump, shaft, condom, foreskin
 Top tags: penis, vagina, burn, pregnancy, pain, ejaculation, testicle, sexual intercourse, rash, masturbation
 Examples (original text):
  - pain when urinating inconsistent urination painfull ejaculation painfull mastrubation etc oh gosh im in all sort of trouble here and its given me anxiety over the past 1 5years ive been having this condition it all began when my urine penis started smelling cheesy after urination then later on when i was laying in bed and rising up i could feel from pelvic area like something is almost pushing my urine out it happened all the time then came premature ejaculation painfull urination painfull mastrubation painfull ejaculation also when i drag back my foreskin pain help please
  - i am experiencing a problem keeping an erection are there natural remedies that can be taken for this
  - will my glans burn recover  accidentally i got a very hot water on my penis the water hit a small area of glans and the area under it now rubbing it either by hand or cloths has some sort of annoying feeling i would like to know whether recovery is possible or not if yes please tell me how
  - why will not my penis stay hard when in pregame  i get hard quite easily when around my gf but then all of a sudden when it comes to me taking my jeans off it goes down why  when it does decide to work i really do love sex with her so what is causing this also i can not cum when she tries to give bj
  - can a yeast infection cause pain in the urethra as well as the head of the penis  originially thought to have prostatitis a month ago have been experiencing penis soreness urethra pain some redness gonnohrea and chlamydia negative and 3 weeks of cipro provided no relief now thinking it may be fungal could this be a yeast infection and if not what could it be

Cluster 11  (n=1040)
 Top terms: diarrhea, stool, colitis, bowel, movement, constipation, blood, colon, disease, crohn, gastroenteritis, colonoscopy
 Top tags: diarrhea, bowel movement, ulcerative colitis, constipation, diet, "crohns disease", pain, colonoscopy, stomach, colon cancer
 Examples (original text):
  - i have started a low sugar and low wheat diet and i keep going to the toilet more than normal is my diet the reason
  - how to heal crohn is ulcers  my husband has crohns he has no pain and is mostly symptom free he was diagnosed with crohns from a biopsy after intestinal cancer he has had two surgeries to remove ulcers and he has deep ulcers again the doctor wants to put him back on endocort but that was the drug he was on when he got the ulcers back after the second surgery he has been checked for inflammation but the tests always come back well within normal limits what can we do to heal the ulcers
  - my butthole hurts and there is a bump and it hurts whenever i sit or move i assume it happened a few days ago when i was trying to hold in my bowel movement but i could not so i went to the bathroom but i do not remember anything happening to me for the rest of the day or the next but on sun today8th 9th its been bothering me severely and it hurts whenever in doing anything that requires me to move my lower body
  - what are the risk factors that cause malnutrition for someone with ulcerative colitis
  - how to treat diarrhea in a 6 month old baby who has had over a week and his bottom is raw

Cluster 12  (n=787)
 Top terms: kidney, urine, uti, bladder, infection, stone, pain, urination, blood, antibiotic, tract, test
 Top tags: pain, urination, kidney stone, kidney, bladder, vision, burn, urinary tract infection, pregnancy, antibiotic
 Examples (original text):
  - how to manage large kidney stones that can not be passed on its own while being 16 weeks pregnant and not harm the baby stones are roughly 5 8 mm in size recurrent episodes of pain over the past 2 3 weeks should i stent how safe are the pain meds percocet and dilaudid for the baby
  - why does my urine smell
  - is it safe to swim in a lake if you have a uti  i have a bladder infection and began antibiotics two days ago we are going to the lake this week can i swim in the lake or should i avoid that
  - my urine has a bad smell and is cloudy what can be wrong
  - is there a drug to strengthen the bladder muscle

Cluster 13  (n=1791)
 Top terms: sex, condom, sperm, intercourse, ejaculation, pill, boyfriend, chance, birth, semen, girlfriend, control
 Top tags: pregnancy, sexual intercourse, condom, ejaculation, sperm, vagina, period, birth control, masturbation, anus
 Examples (original text):
  - how effective are male condoms at birth control
  - i am pregant does everything i wear have to have cotton in it  my boyfriend is sure he read that now that i am pregnant everything i wear has to have at least a percentage of cotton in it i know that my panties should be cotton but i can not find answers about the rest of my wardrobe this is making it difficult to find suitable pants for my work uniform
  - gave boyfriend oral then ate an hour later got home and felt mucus in throat and spit it went down my body pregnant  gave my boyfriend oral and swallowed some then ate a small burger and some coke an hour later i took a shower while in the shower i felt then need to spit i felt like mucus so i did spit worried that spit contained sperm and traveled down my body and got me pregnant i did not insert anything in not even with fingers and i was obviously standing while in shower but spit went through middle of my stomach so i am sure it passed by my vagina i am only 19 help can i be pregnant
  - my boyfriend and i had sex using both the pill and a condom the condom broke should i worry about rubber inside me  we think most stayed on his shaft and i found a small piece about 34 inch x 12 inch like it ripped and a portion tore off do i have to worry about this causing tss 
  - sexual health how can i increase my semen due to much masturbation i m starting masturbation in the age of 14 yr regularly many time in a day now i am 28 yr i have following major problems 1 erectile dysfunction 2 premature ejaculation 3 low sperm count after masturbation semen has comes only 4 5 drops and very slander can i pregnant my wife in this situation please sir help me

Cluster 14  (n=1720)
 Top terms: foot, leg, knee, arthritis, osteoarthritis, pain, ankle, osteoporosis, bone, toe, hip, doctor
 Top tags: foot, arthritis, leg, knee, pain, osteoarthritis, osteoporosis, ankle, walking, toe
 Examples (original text):
  - can i sit in a sauna and steam room with a broken ankle
  - is hand foot and mouth the same as rubella is hand foot nd mouth the same as rubella
  - i have lump in the soft tissue of my left leg about 1 inch above my ankle i have had an ultrasound and a xray and i have been told it is not life threatening but the doctor will not tell une but it is hurting and swelling and i am taking pain killers every day me what it is until he sees me again in june
  - i have had 2 knee operations knee feels like loose
  - i am tired legs feel heavy and have disequilibrium doctors can not pinpoint problem

Cluster 15  (n=1141)
 Top terms: surgery, hernia, cyst, option, hysterectomy, surgeon, procedure, fusion, operation, doctor, opinion, bypass
 Top tags: surgery, hernia, hysterectomy, pregnancy, cyst, insurance, smoking, ovary, pain, healing
 Examples (original text):
  - i just find out by a x ray i have 10 stables inside of me is it normal after surgery
  - just discovered a 4cm x 4cm cyst inside my left ovary via ultrasound will ovary need removal
  - i had a back fusion at l4 s1 now have anterlothesis of l4 on l5 do i need surgery
  - what are the treatments for a hernia
  - is it normal to break out after a laser resurfacing procedure and what should i do

Cluster 16  (n=1450)
 Top terms: pregnancy, baby, test, trimester, miscarriage, fertility, birth, woman, abortion, symptom, infertility, risk
 Top tags: pregnancy, baby, miscarriage, pregnancy test, fertility, infertility, sexual intercourse, postpartum depression, ovulation, ultrasound
 Examples (original text):
  - how effective are foam and male condoms in preventing pregnancy
  - can newborn babies be born addicted to prednisone if the mom took it for asthma in the last trimester
  - i take steroid prednisolone for ivfpregnancy due to auto immune issue but i have herpes 2 will this affect baby
  - inserting finger into girls vagina leads to pregnancy
  - i had sex in the 5th week of pregnancy and saw vaginal bleeding why

Cluster 17  (n=2166)
 Top terms: skin, acne, rash, face, spot, eczema, cream, bump, product, dermatitis, scar, area
 Top tags: skin, rash, acne, arm, atopic dermatitis, itch, vision, burn, scar, blister
 Examples (original text):
  - guest in my home has scabies do i have house sterilized or will a good cleaning do it we have not had skin contact he is getting treated and i am having a general cleaner come in this afternoon to change linens etc is this sufficient i have never had anything like this in my home somewhat disturbed
  - what is the best moisturizer for older skin
  - what medicines can cause your skin to turn blue
  - i have blisters on the inside of my lips and on my tongue what kind of doctor should i see  i also have multiple sclerosis hypothyroidism incontinence bipolar borderline personality disorder and an addict i smoke i do not drink or eat spicy anything i am a 37 year old woman with no children
  - i use a high spf lotion on my face daily especially when going outdoors why do i have sun spots and white freckles

Cluster 18  (n=454)
 Top terms: sleep, bed, bedbug, aid, insomnia, nap, bug, apnea, ambien, fatigue, baby, melatonin
 Top tags: bedbug, pregnancy, baby, over the counter, drinking, insomnia, melatonin, stress, exercise, coldness
 Examples (original text):
  - had a stroke on the brain in 2012 its 2016 i cant get no more than 5 hours of sleep a day
  - how much nap time does my baby need
  - when is the best time to take vitamins in the morning or at bedtime
  - how much sleep should my 4 yr old be getting a night
  - major sleeping difficulties for the past month due to anxiety restless legs meds dont help no caffine reg schedule sleeping pills do not help stopped melatonin since i was getting pounding headaches concussion about one month ago but all symptons have disappeared daily activities same often up for days at a time major depression for years but now symptoms seems to be leaning more toward bipolar it seems like a can fall asleep on the couch better than in the bedroom  please help  meds include xanax paxil topomax and requip in addition to occasional percoset or oxycodone to treact chronic pancreatitis

Cluster 19  (n=715)
 Top terms: ear, infection, hearing, pain, doctor, head, tinnitus, neck, fluid, earache, eardrum, sound
 Top tags: ears, pain, ear infection, tinnitus, head, antibiotic, coldness, neck, vision, pressure
 Examples (original text):
  - why am i hearing my heartbeat in my right ear  just recently i have started hearing my heartbeat in my right ear this came on suddenly i am a 66 year old female with no particular health issues what could be the cause of this anything to worry about
  - should i go to the er for severe right earache or wait until monday is appointment  i am 20 on thursday i woke with dull throbbing it has gotten worse i have a 101 f fever i think the throbbing is still there but now my ear canal is so blocked i can not feel anything in the canal but there is still sharp stinging pain in what feels like the back my outer ear is swollen the pain extended to my throat chewing or hiccups feels like my ear is tearing i have almost zero hearing in this ear er or wait
  - i have had white noise with corresponding hearing loss in my left ear for 2 months what causes it and can it be fixed i have had an mri    negative for tumor ms or anything else that might be causing it
  - does lipo flavonoid reduce or cure tinnitus
  - i am having ruptured eardrum one doctor suggested to have surgery and other suggested to wait for 3 months

Cluster 20  (n=913)
 Top terms: fever, pain, throat, body, symptom, headache, grade, cough, son, chill, temperature, flu
 Top tags: fever, running, pain, cough, headache, flu, antibiotic, coldness, swelling, sweating
 Examples (original text):
  - my baby ate her on poop my baby ate poop 4 days later she is sick weezing coughing and high fever for 4days straight i took her to the doctor and they said shes fine just a normal cold i told them what happen and they just said she should be fine but if she still has a fever next week come back what should i do and is her symptoms related to her eating her poop
  - i have persistent headache and i feel like i have lowgrade fever help  hi so i am a headache everyday it is not too bad though it is completely bearable but a little distracting and i have noticed that i have lowgrade fever most of the time or mostly everyday but just like the headache it is bearable i can not just shrug this feeling off this have been occurring for two or three months now i am hopefully going to the doctor in a few days and get myself checked
  - why do i feel lightheaded fatigued and sweat during sleeping no fever  i am an almost 37 yr old female with a lot of stress right now dr put me on effexor and i started not being able to sleep having bad headaches feeling lightheaded and constipated i took it for 2 weeks and he told me to stop when i called him he called me in something else but i am afraid to get it i have been on paxil prozac and celexa and never felt this horrible i have not taken anything in almost a week but feel lightheaded many times thoughout the day any ideas what could be wrong
  - can i have strep without fever  my 4 year old son was diagnosed with strep throat 3 days ago last night i was fine one minute and suddenly felt like i would been hit by a ton of bricks body aches headache  sore throat and general feeling crappy but no fever is there any point in dragging myself out to doctor when i feel so miserable is it possible to have strep without fever i do not have any runny nose stuffy nose or cough not a cold 
  - how long should i wait before bringing my 11 yr old with flu symptoms to our family dr it has been 8 days initial symptoms were nausea high fever severe headache loss of appetite and fatigue those lasted about 2 days now she is very tired little appetite sore bellynausea and has a sore throat and cough

Cluster 21  (n=981)
 Top terms: headache, migraine, head, pain, nausea, dizziness, sensitivity, side, pressure, symptom, neck, vision
 Top tags: headache, migraine, nausea, pain, head, dizziness, vision, neck, ears, photosensitivity
 Examples (original text):
  - i have heavy pain in both side of my head that causes dizziness sometimes in my back and neck
  - i have been taking propranolol for the chest pains now have headaches and pain on left side of head and body
  - i have been having very sharp stabbing pains down through the top rtrear of my head the pain almost knocks me down i have been having these pains for 6 7 weeks i have had no previous head injuries they just started out of the blue they are not headaches they are in  a dime sized spot on top of my head right side just off center back portion top of head does that make any sense these pains happen wether  i am standing or laying down thank you for your time
  - can diabetes cause you to have chronic migraines
  - could clonazepam be causing nausea and dizziness or is it bupropion

Cluster 22  (n=631)
 Top terms: marijuana, smoking, test, drug, cigarette, smoke, urine, smoker, cocaine, system, nicotine, weed
 Top tags: smoking, marijuana, drug test, drinking, pregnancy, quit smoking, drug, cocaine, lung, nicotine
 Examples (original text):
  - i smoked cigs for 1 month averaging about 3 a day just wondering if any irreversible was done i did quit since then i started smoking for a month after a period of depression a couple of months ago 2 3 cigs most days with a couple more on bad days i would estimate i probably had 4 packs total over the period i went cold turkey as i started to get my life together and hated the ill feeling from them i exercise regularly and eat healthy and i am still young i would just like to clear my head and hear that i did no damage permanent to my lungs i know it takes a bit to recover hopefully to 100
  - what is less damaging smoking 3 cigarettes per day or picking two days of the week and smoking 10 on each of those days  i know that light smoking still poses significant risks i know i can still get cancer one day from doing it so please do not answer by telling me that i am looking to significantly cut down in order to lessen the damage i am doing to myself while still indulging a little in one of my favorite activities so here are my two options a smoke about 3 cigsday about 20week or b pick 2 nightsweek say two nights when i am out socializing and smoke about 10 on each of those about 20week
  - drug test where to find
  - does quiting smoking improve semen motility and morphality levels or just prevent any further deterioration
  - problems keeping an erection i am an 18 year old male i weight 133 pounds and i am 58 i have problems keeping an erection it will get hard but it gets soft rather quickly without stimulation like a few seconds when it is hard it usually stays a sort of mostly hard but still soft penis i smoke cigarettes and marijuana but not to the extent where it would cause major damage i exercise 2 3 times a week and eat healthy although i do indulge in some junk food could i really have erectile disfunction at my age

Cluster 23  (n=1346)
 Top terms: heart, pressure, blood, chest, pain, failure, disease, bp, rate, hypertension, artery, attack
 Top tags: high blood pressure, blood pressure, heart, heart disease, chest, congestive heart failure, pain, exercise, heart attack, low blood pressure
 Examples (original text):
  - ekg says there was “moderate right axis deviation ” “normal sinus rhythm with marked sinus arrythmia ”
  - is there evidence that statins increase life expectancy for people without heart disease
  - i felt like electric shock like feeling which became sharp on my left chest
  - would lad lesion cause tachycardia
  - heart attack i am a parapalegic and the other night when i was going to bed i had a burning sensation that started in the chest area and moved to the back to the point my upper torso was burnig all the way around i was in total discomfort i felt a heaviness in my chest this lasted for about 3 hours are these symptoms of a heart attack or are these symptoms that have to do with my spinal cord injury i have had these symptoms before but never to the degree i had the other night

Cluster 24  (n=1314)
 Top terms: disease, disorder, parkinson, symptom, epilepsy, dementia, seizure, schizophrenia, depression, lupus, syndrome, people
 Top tags: bipolar disorder, "parkinsons disease", epilepsy, dementia, schizophrenia, seizure, celiac disease, depression, lupus, lactose intolerance
 Examples (original text):
  - what are the dietary restrictions for celiac disease gluten
  - where can i go for help for bipolar disorder
  - what is the treatment for the common cold
  - is erythema multiforme an autoimmune disorder
  - i was diagnosed with mononucleosis 7 months ago since then i have not been able to recover

Cluster 25  (n=1548)
 Top terms: cough, pneumonia, sinus, throat, nose, asthma, allergy, infection, bronchitis, cold, symptom, sinusitis
 Top tags: coldness, cough, sinus infection, throat, pneumonia, cold, nose, chest, allergy, asthma
 Examples (original text):
  - i have had a pneumonia shot can i get either a sinus infection or walking pneumonia from my 6 year old grand daughter  and can i be a carrier to others in my age group
  - how soon should my 14 year old wait before returning to school having been diagnosed with pneumonia
  - my husband has had a fever and a cough for over 10days now not pneumonia had xray 4 days ago what could it be  his fever ranges between 100 103 he is coughing up lots of mucous but does not feel any sinus congestion no runny nose he has been on an antibiotic for 5days now he says it is helping a little but it is not taking care of the fever or cough just making the mucous less green he had a chest x ray 5days ago but it did not show pnumonia should he be x rayed again he is also very pale in color as well as is experiencing dizziness headache and nausea had a cbc blood work came back normal
  - what are the symptoms of pneumonia
  - is viral pneumonia contagious

Cluster 26  (n=575)
 Top terms: breast, lump, nipple, mammogram, chest, cancer, pain, milk, armpit, ultrasound, biopsy, side
 Top tags: breast, nipple, pregnancy, lump, mammogram, breast cancer, pain, milk, swelling, ultrasound
 Examples (original text):
  - hi my husband has a lump on his head that has been there for 2 weeks and now has a lump under his armpit hi my husband has a lump on the back of his head for about 2 weeks now and it appeared there without injury we were a little concerned but now i am really concerned because he has another lump under his armpit that has been there for 2days
  - i have had rapid breast growth want breast reduction surgery what age should i typically get this surgery
  - why do my breasts hurt and feel and look like they are getting bigger  for almsot 2 weeks now my breasts have been painful and feel like and look like they are getting bigger i instantly thought of being pregnant but i am on the mirena birth control and although its not 100 effective i still did not believe i were pregnant i took a home pregnancy test from my local drugstore and it was negative why are my breasts feeling this way people i know have been commenting on them looking huge or bigger than normal
  - will health insurance through the marketplace cover genetic testing for breast cancer
  - i am 44 yrs tubal ligation 18yrs ago last period came for 1 day my breast hurt for 3 wks and vomiting am i pregnant

Cluster 27  (n=955)
 Top terms: weight, diet, loss, pound, calorie, lbs, gain, fat, body, lot, protein, exercise
 Top tags: weight, diet, exercise, weight loss, weight gain, food, calorie, height, pregnancy, corpulence
 Examples (original text):
  - what weighs more muscle or fat  im just wondering about weight gain due to muscle growth my wife has been working out for some time with weights and cardio training but she is finding that her weight has been fluctuating and at times gains weight a little bit
  - are diet pills safe for teenagers if so which ones are
  - had total knee replacements i am not feeling good no energy depressed no appetite have lost weight
  - what can i do to gain back my missing pounds and feel healthy again  i have been sick and lost 17 pounds i am fatigued all the time and look poorly i want to gain my wieght back and feel good again as quickly as possible
  - would you kindly suggest diet for a pregnant woman

Cluster 28  (n=1074)
 Top terms: antibiotic, infection, throat, strep, amoxicillin, penicillin, treatment, doctor, bacteria, meningitis, pneumonia, virus
 Top tags: antibiotic, amoxicillin, throat, penicillin, virus, bacterium, strep throat, sore throat, pain, sinus infection
 Examples (original text):
  - can an antibiotic through an iv give you a rash a couple days later
  - can taking multiple antibiotics cause redness and dryness of vagina
  - i need relief from chronic epididymitis
  - is clindamycin effective in treating syphilis
  - why it is necessary to take 2 antibiotics for a diverticuilitis infection will taking just the cipro work

Cluster 29  (n=1104)
 Top terms: blood, vitamin, test, anemia, level, iron, result, atherosclerosis, cell, hemoglobin, work, disease
 Top tags: anemia, blood test, vitamin, pregnancy, atherosclerosis, vitamin d, exercise, iron, coldness, low blood pressure
 Examples (original text):
  - is all vitamin d the same
  - what is black measles when i was young i had them now in my fifties i have a lot of health problems could it be because of them and what damage do they do to your body i know they have not been heard of in god know how long is there any way to know after all of these years after having them to get information on them
  - can menstrual cycle affect diabetes blood work
  - is it safe to give a 2 month old pedialyte on a daily basis
  - how to correct vitamin b12 deficiency

Cluster 30  (n=1755)
 Top terms: pain, side, rib, back, neck, shoulder, muscle, chest, spine, leg, fibromyalgia, area
 Top tags: pain, back pain, burn, fibromyalgia, neck, leg, exercise, shoulder, muscle, arthritis
 Examples (original text):
  - broken collarbone 3 5 cm overlap its been three weeks after break and still feels broken or loose
  - i have an acute dextroscoliosis i feel pain when i skip meals
  - does lidocaine cure canker sores on your throat  i found a small white sore on my throat and i was prescribed lidocaine and it numbs the pain but i was wondering if it cures it at the same time before it gets worse
  - was skiing and fell on my knee cap now 3 days later i just heard a pop and its throbbing and its severe pain  they did an xray but they did not have an mri machine available i will not be able to access a doctor for another 3 9 days a half hour the pop happened and it really hurts on a scale of 1 10 10 being like you have just been shot its more of an 8 7
  - can age appropriate arthritis be exaserbated by barometric pressure change or other weather conditions  im 55 very fit and athletic and often suffering with joint pain and stiffness like never before and all over my body not sure where it came from or how to get rid of

Cluster 31  (n=891)
 Top terms: period, sex, cycle, menopause, birth, pill, boyfriend, control, woman, may, discharge, ovulation
 Top tags: period, pregnancy, sexual intercourse, ejaculation, spotting, menopause, ovulation, birth control, irregularity, vagina
 Examples (original text):
  - my period only last 36 48 hours which is my norm is that why i have had 2 yrs of no luck getting pregnant  my husband and i have been trying for two years to have a child i am turning 30 next month and in my family after 30 equals issues my normal period is only 36 48 hours could this be preventing me from getting pregnant
  - spotting on day two of my period could i be pregnant i have had my loop taken out 71015 and have had unprotected sex a few times after that hoping to fall pregnant my period was meant to start 12102015 it now day two on my period and i have only been spotting which is very un usual could i be pregnant
  - my period has been late by 4 days i am trying to conceive please help
  - had sex a week later instead of my normal period had alittle blood when wiped now im sick to stomach and nipples itch could i be pregnant
  - i have not had a period for 4 months

Cluster 32  (n=801)
 Top terms: eye, vision, cataract, eyelid, circle, doctor, pain, discharge, pinkeye, nose, problem, face
 Top tags: eyes, vision, pink eye, cataract, eyelid, swelling, burn, coldness, skin, sty
 Examples (original text):
  - seems i have got chemical in eye from eye cream anything i can do to get relief from burning i have tryed to rinse
  - i have been suffering from pressure and pain behind eyes for almost three years
  - my 2 friends also underwent through lasik procedures but i am afraid about any side effects can anyone guide me about the possible risks thanks
  - i was checking my husband is testicles and i felt a peanut size lump above his left testicle not sure if we should worry  we are both without insurance right now it would be nice to know if i should just keep an eye on it or if it will just go away
  - my eyelids sag and i have heard that botox can lift my eyelids how does this work will not it look strange

Cluster 33  (n=1098)
 Top terms: stomach, pain, nausea, abdomen, diarrhea, side, cramp, doctor, feeling, vomiting, symptom, daughter
 Top tags: stomach, nausea, pain, diarrhea, vomit, pregnancy, cramps, fever, bloating, vision
 Examples (original text):
  - after a bowel movement abdominal pain puts me on my knees severe nausea to the point i do vomit
  - how long can flucold causing bacteria live outside the human body  pretty much what i asked my mom had a nasty stomach bug which i am pretty sure was the flu with all the nasty symptoms that go along with it not thinking she used my computer for something to do i have been avoiding it using a friend is computer now but how long do i need to i have stuff i need to do on it so how long do i need to wait before i do not need to worry about contracting it myself
  - i began to feel a sudden twitchspasm feeling in my stomach went to er twice no solution
  - pain in upper right quadrant of abdomen may indicate what 18 year old grand daughter had gallbladder stmptoms but no stones doctor said bile was crystalizing and bladder was removed still having problems food test showed food slow at passing from stomach to intestines smaller portions have not helped couple crackers may induce vomiting what are possible causes
  - pain in testicles lower abdomen rectum is it fatty liver

Cluster 34  (n=581)
 Top terms: hair, loss, scalp, head, shampoo, woman, treatment, chin, cause, growth, man, beard
 Top tags: hair, hair loss, scalp, head, ringworm, stress, lice, pregnancy, acne, itch
 Examples (original text):
  - my hair has been thinning over the past few years taking minoxidil is it safe
  - how effective is laser hair removal at treating keratosis pilarus especially on arms
  - does xanax show up in hair sample
  - does vinegar stop hair loss in women
  - can perming a girls hair kill head lice and nits

Cluster 35  (n=513)
 Top terms: diabete, sugar, diabetes, type, blood, insulin, level, diabetic, disease, glucose, diet, doctor
 Top tags: diabetes, blood sugar, type 1 diabetes, insulin, type 2 diabetes, exercise, diet, high blood pressure, injection, pregnancy
 Examples (original text):
  - do doctors immediately prescribe medication to control blood sugar or do they wait to see if diet and exercise help
  - i was screened early for gestational diabetes 10w it came back high so now i do the 3 hour i am worried and scared my mother and brother both have diabetes and i am just really worried about this i had a hard time getting pregnant over 2 years and just do not want to have to worry about this is there any thing i can do to calm me down or help me manage it if i do have it i am feeling very stressed
  - is garlic safe for my diabetic dog 8 year old bichon with atypical cushings and diabetes he takes 7 units of novalin insulin bid flax hull and melatonin 3mg bid sugars are consistantly high and is going blind
  - does high sugar intake while pregnant mean a big baby
  - does niacin affect the bodys ability to produce insulin

Cluster 36  (n=1605)
 Top terms: insurance, health, plan, medicare, coverage, income, care, exchange, medicaid, marketplace, employer, company
 Top tags: insurance, health insurance, medicare, affordable care act, medicaid, health insurance exchange, health insurance marketplace, obamacare, dental, vision
 Examples (original text):
  - i am a disabled veteran on medicare am i affected by the affordable care act
  - is liposuction covered by insurance
  - i manage a medical office with 3 employees  rather than offer a health insurance plan we pay 50 of the employees premium so if they purchase their insurance through the marketplace will we no longer be able to do that
  - will they accept obamacare at any hospital
  - i am 50 years old and am currently on medicaressdi i have part a b and d  i also have aarp hospital indemnity by united healthcare i also have an aarp rx drug plan is this enough to cover what is needed under the affordable care act

Cluster 37  (n=806)
 Top terms: anxiety, depression, attack, panic, disorder, symptom, stress, medication, med, heart, pain, xanax
 Top tags: anxiety, depression, panic attack, stress, pain, heart, fear, chest, drowsiness, drug
 Examples (original text):
  - i would appreciate advice on a drug regimen that will restore serotonin while dealing with short term anxiety symptoms
  - should medical marijuana be used to treat anxiety disorder
  - my primary doctor diagnosed me with anxiety disorder and prescribed xanax and zoloft why do i still have panic attacks
  - why do my hands shake involuntarily it happens when i am trying to do something precise like pour sugar in my cup it also happens after i have eaten something so my blood sugar is not low and i do drink but i have ruled out delirium tremors i had eaten something tonight and then when i went to bring my chicken wings to the table i dropped them on the floor when i had to drive because my gf was too tired my anxiety was through the roof i feel like my anxiety goes hand in hand with my shaky hands but i do not know the cause
  - i served in the navy and now have panic disorder my family does not believe in mental illness are they right

Cluster 38  (n=1954)
 Top terms: overdose, pill, medication, control, birth, effect, prescription, dose, tylenol, counter, aspirin, tablet
 Top tags: drug overdose, pregnancy, injection, birth control pill, ibuprofen, over the counter, acetaminophen, birth control, coldness, drinking
 Examples (original text):
  - does prozac cause weight gain what about zoloft
  - what are the ingredients inibuprofen  i take a 600mg ibuprofen only as needed for nerve pain my question is what ingredients are in this medication i have a legal prescription for it
  - my husband is taking 40 mg of prozac and is really depressed and has thought of suicide what do we do
  - i swallowed 20 tablets of 40mg citalopram whay should i do
  - sleeping pills for traveling what should i take to sleep on the plane im traveling to israel from miami fl and am super scared of flying i would like to sleep on the plane so i dont get nerve recked whats safe to take that i can bring on the plane with me and that i will be able to wake up after my flight lands

Cluster 39  (n=477)
 Top terms: gallstone, reflux, heartburn, hernia, gerd, gallbladder, disease, pain, symptom, liver, gall, doctor
 Top tags: heartburn, gallstone, pain, hernia, burn, pregnancy, gallbladder, stomach, acid reflux, diet
 Examples (original text):
  - i had ovarian cancer and reflux surgery i still deal with constant nausea i can barely eat and i am unable to live life and go anywhere
  - my fart hurt the back of my head on the left side  this is a stupid question but i farted and the left side of the back of my head started hurting for a few seconds did this fart damage my brain or anything
  - how are gallstones diagnosed
  - is it normal to feel liver pain after gallbladder removal  i had viral hepatitis from bad drinking water in a 3rd world country when i was 9 years old now i wonder if the gallbladder stones are associated with this because after 5 days of having the gallbladder removed i am feeling liver pain thank you roxana casas
  - how can i cure acid reflux  i ate a large meal and went to bed right after and 4 hours later woke up with a burning throat gargled some cold water and saw some blood got worried gargled couple of more times but the bleeding got worse rushed to a doctor and got some medications but i still sometimes feel its hard to sleep and when i wake up my throat feels very dry even though i have woken up couple of times to drink water

Cluster 40  (n=372)
 Top terms: hypothyroidism, thyroid, hyperthyroidism, synthroid, weight, nodule, medicine, level, medication, hypothyroid, hormone, tsh
 Top tags: hypothyroidism, thyroid, hyperthyroidism, weight, exercise, pregnancy, hives, hormone, diet, tsh levels and pregnancy
 Examples (original text):
  - what should i do if i suspect an overdose of thyroid
  - i have all the classic signs of hypothyroidism but lab results normal where to go from here
  - im 27 and ive had almost all of the symptoms since i was about 6 years old could it be possible i have hypothyroidism
  - what would make my tsh levels suddenly jump to critically high level when i feel the same  i take synthroid 1 25 daily and yet my tsh is now 28 3 increased from 23 on friday why would it jump so quickly my dr is on vacation and i just want to make sure this is ok to wait until thursday when he comes back 
  - how is hypothyroidism treated

Cluster 41  (n=1608)
 Top terms: baby, problem, wart, way, hive, cold, child, woman, tip, body, daughter, face
 Top tags: pregnancy, exercise, coldness, cold, baby, wart, hives, transient ischemic attack, my daughter cries, acne
 Examples (original text):
  - can i transmit genital warts seventeen years after having them removed
  - i have been feeling extremely exhausted and unable to do basic tasks need advice
  - what causes hives
  - my body has not been feeling good at all what can be wrong
  - what do hives look like when they start to clear up

Cluster 42  (n=259)
 Top terms: hepatitis, liver, virus, infection, symptom, cirrhosis, risk, prognosis, effect, hepititis, mess, vaccine
 Top tags: hepatitis c, hepatitis, virus, liver, hepatitis b, infection, hepatitis a, concerned about atrophic liver, just found out i have hepatitis c?, sexual intercourse
 Examples (original text):
  - how do you get hepatitis c
  - is cirrhosis a form of liver cancer
  - who should receive antiviral therapy for hepatitis c virus
  - is it true that the hepatitis c virus cannot survive for more than two hours outside the human body
  - how many kinds of viral hepatitis are there

Cluster 43  (n=1111)
 Top terms: food, diet, allergy, meal, milk, cholesterol, water, meat, vegetable, fruit, baby, supplement
 Top tags: food, diet, drinking, food allergy, meal, milk, baby, fruit, vegetable, feeding
 Examples (original text):
  - can you be allergic to mold in your food
  - is it better for a type ii diabetic to eat corn or bread stuffing
  - why does coffee give me such an energy boost
  - i had energy supplements now feeling dizzy and passed out today
  - i am a woman 41 and eat a balanced diet and take a multivitamin do i need fish oil and calcium too

Cluster 44  (n=1037)
 Top terms: doctor, opinion, hospital, prostatitis, patient, medicine, ultrasound, test, exam, problem, neurologist, symptom
 Top tags: vision, pregnancy, prostatitis, ultrasound, pain, family, wart, exercise, i have many symptoms need second doctor opinion, bipolar disorder
 Examples (original text):
  - my 12th week scan showed everything is o k but radiologist suggested there is caudal regression syndrome
  - i have or think i have parkinsons disease when should i contact my doctor
  - admitted into the hospital for angioedema how to get swelling to go down
  - i have a patient of mine has pitting oedema difficult diagnosis any help would be appreciated
  - what kind of doctor do i need to see if my clavicle did not heal properly

Wrote assignments to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/clustered_short_questions.csv

=== Assessment Metrics (COSINE) ===
 Features: BERT(all-MiniLM-L6-v2) 384D + Tags→SVD(20D, EV=0.245) = 404D
 Global: silhouette (cosine)=0.0947

--- Cluster Size Stats ---
 k=45 | min=259 max=2482 mean=1055.4 median=955.0 std=497.8

--- Per-Cluster Averages ---
 cluster | size | mean silhouette (cosine)
       0 |  756 |         0.1496
       1 |  809 |         0.1826
       2 | 2482 |        -0.0442
       3 | 1402 |         0.0629
       4 |  955 |         0.1046
       5 | 1275 |         0.0293
       6 |  726 |         0.1313
       7 |  850 |         0.0705
       8 |  489 |         0.2225
       9 |  381 |         0.3454
      10 |  793 |         0.1147
      11 | 1040 |         0.0930
      12 |  787 |         0.1188
      13 | 1791 |         0.0540
      14 | 1720 |         0.0609
      15 | 1141 |         0.0612
      16 | 1450 |         0.1256
      17 | 2166 |         0.0671
      18 |  454 |         0.1302
      19 |  715 |         0.2222
      20 |  913 |         0.1614
      21 |  981 |         0.2006
      22 |  631 |         0.0795
      23 | 1346 |         0.1048
      24 | 1314 |         0.0291
      25 | 1548 |         0.0492
      26 |  575 |         0.1231
      27 |  955 |         0.0943
      28 | 1074 |         0.0357
      29 | 1104 |         0.0167
      30 | 1755 |         0.0497
      31 |  891 |         0.1478
      32 |  801 |         0.1747
      33 | 1098 |         0.1085
      34 |  581 |         0.2235
      35 |  513 |         0.1948
      36 | 1605 |         0.2404
      37 |  806 |         0.2023
      38 | 1954 |        -0.0196
      39 |  477 |         0.1062
      40 |  372 |         0.2742
      41 | 1608 |         0.1300
      42 |  259 |         0.4273
      43 | 1111 |         0.0503
      44 | 1037 |         0.0148

--- Structure (cosine on scaled space) ---
 Avg intra-cluster cosine similarity: 0.2640 (higher = tighter)
 Mean inter-centroid cosine similarity: -0.0196 (lower = better)

--- Tag Coherence (avg) ---
 Purity=0.371 | Entropy=4.383 | Intra-Jaccard=0.080

--- Model Selection Top Rows (by silhouette, cosine) ---
 k  silhouette  max_cluster_diameter
45    0.094747              1.283184
44    0.094228              1.283184
43    0.091827              1.283184
40    0.091323              1.281219
41    0.091055              1.265386
42    0.090828              1.253987
39    0.090085              1.258280
30    0.089970              1.283184
29    0.089522              1.283184
31    0.088863              1.283184

[UMAP] Saved 2D scatter to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_scatter.png
[UMAP] Saved embedding CSV to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_embedding.csv

[HClust] Saved dendrogram to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/hierarchical_dendrogram.png

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_timings.json

=== Runtime (seconds) ===
               load_csv_s:   6.4572
         nlp_preprocess_s: 118.6167
    vectorize_text_tags_s: 111.0925
         scale_features_s:   0.1756
        model_selection_s: 476.6413
     interpret_clusters_s:   0.4321
           save_outputs_s:   0.9072
        metrics_compute_s:  27.5919
                   umap_s:  48.4826
           hierarchical_s:   0.7016
          total_runtime_s: 791.1047
```

In version v1.1.3, my workflow begins with a thorough normalization of the raw short medical questions. After lowercasing, punctuation stripping, and whitespace cleanup, each question is passed through a strict spaCy POS filter that retains only nouns and proper nouns, along with a small set of medically relevant short tokens such as "uti" or "hpv." This transformation converts long narrative sentences into concise noun-phrase-style representations that emphasize medical entities, conditions, and body parts. By discarding verbs, adjectives, adverbs, and fillers, the dataset becomes far more suitable for topic clustering, even though this also removes intent-style linguistic signals such as "how long," "can I," or "risk if." For my purposes, building a taxonomy for routing medical inquiries rather than modeling intent, this trade-off works well. After this process, I am left with about 47,491 usable questions, each accompanied by a cleaned list of tags parsed into a normalized list of lowercase strings.

The feature representation in v1.1.3 relies on a dense embedding core provided by the all-MiniLM-L6-v2 model, applied to the POS-filtered text. This produces 384-dimensional semantic vectors. To incorporate domain knowledge, I encode the tag lists using a multilabel binarizer followed by SVD, reducing them to a compact 20-dimensional representation. Concatenating the text and tag vectors yields a 404-dimensional embedding, which I standardize and L2-normalize before clustering. I also train a separate TF-IDF model on the POS-filtered text solely for interpretability, allowing me to inspect top terms per cluster in a sparse, human-readable form while keeping the clustering itself driven by more expressive dense embeddings.

Clustering in v1.1.3 uses KMeans on a range of values of k from 10 to 45. To select the best model, I rely on cosine-based silhouette scores and measurements of maximum intra-cluster distances. The silhouette curve gradually climbs as k increases, and the best score appears at k = 45. Although the absolute silhouette values are small, as is common in high-noise medical short-text datasets, the relative improvements across k provide useful guidance. Cluster diameters shrink modestly as k increases, and the UMAP projections reveal a mixture of dense central regions surrounded by smaller, more coherent islands. The dendrogram of centroid embeddings offers a clear hierarchical structure that can be cut at higher levels to produce broader super-topics if needed.

The metrics for this run show a global cosine silhouette of about 0.0947, which is not unusual for datasets with overlapping symptom descriptions, multi-condition questions, and ambiguous phrasing. Cluster sizes range from a few hundred to several thousand samples, with a mix of broad catch-all clusters and tightly cohesive medical subdomains. Average intra-cluster similarity is noticeably higher than inter-centroid similarity, indicating that clusters have meaningful internal structure despite the overall difficulty of the dataset. Tag coherence metrics tell a similar story: although purity values near 0.37 and modest Jaccard overlaps reflect the inherently multi-label nature of the data, the clusters still capture medically coherent themes suitable for routing or taxonomy building rather than strict single-label classification.

Qualitatively, v1.1.3 produces clusters that reflect natural medical groupings. Conditions related to pregnancy, reproductive health, and menstrual cycles split into multiple interpretable subclusters covering pregnancy tests, contraception, trimesters, and vaginal infections. Musculoskeletal concerns also decompose cleanly into clusters relating to back pain, joint injuries, arthritis, or exercise-related issues. Internal organ systems form other stable groups, such as those dominated by gastrointestinal symptoms, UTIs and kidney problems, or liver and gallbladder disorders. Chronic diseases, mental health topics, and insurance or healthcare navigation questions each find their own regions in the clustering space. These clusters collectively form a broad medical taxonomy that can be used for downstream routing or data organization.

The runtime for the entire pipeline is around thirteen minutes, dominated by KMeans model selection, POS processing, and embedding generation. If runtime reduction becomes necessary, narrowing the k-range or using approximate clustering for model selection would reduce computation significantly.

From a practical standpoint, the prediction workflow for a new question is straightforward. After passing through the same normalization, POS filtering, embedding, and tag encoding steps, the question is assigned to one of the learned clusters. Each cluster has an interpretable label based on its top TF-IDF terms and a characteristic distribution of medical tags. This makes v1.1.3 suitable as a baseline taxonomy engine, especially for analytics, routing, or weak supervision. It has several high-quality clusters, many usable medium-cohesion clusters, and a few broad groups that could be refined in later versions.

## v1.2.3

```
Rows after POS cleanup: 47491
[Features] BERT(all-MiniLM-L6-v2) 384D + Tags→SVD(20D, EV=0.245) = 404D
[Shapes] Combined feature matrix: (47491, 404)
Searching for a good k (cosine metrics only)...
Model selection summary (first rows):
 k  silhouette  max_cluster_diameter
10    0.068711              1.326555
11    0.069707              1.290245
12    0.074148              1.290245
13    0.076263              1.310299
14    0.075475              1.291242
15    0.077149              1.310299
16    0.077758              1.310299
17    0.078318              1.310299
18    0.078135              1.304229
19    0.079234              1.281636
Selected k = 45 with silhouette (cosine) = 0.095
Saved model-selection table to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_model_selection.csv

[Plot] Saved silhouette vs k plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v1.2.2_silhouette_vs_k.png

[Plot] Saved max cluster diameter vs k plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v1.2.2_max_cluster_diameter_vs_k.png

=== Cluster Summaries ===

Cluster 0  (n=756)
 Top terms: exercise, muscle, weight, workout, program, type, activity, gym, heat, fat, aerobics, minute
 Top tags: exercise, muscle, walking, workout, weight, diet, coldness, running, pregnancy, hunger
 Examples (original text):
  - what exercise does not aggravate achilles tendonitis  i am trying to rest my achilles so i can return to playing tennis in the meantime i want to exercise to stay in shape what types of exercise can i do
  - what is resistance exercise
  - what is a weight loss exercise for people with arthritis in feet ankles and disc herniation in neck and low back  swimming irritates the neck and pain shoots down my arms walking in good sneakers pains my feet and ankles which have arthritis i can stretch but that is not enough to lose weight
  - how do you know what the best exercise routine is  i have had bariatric bypass surgery in 2010 i went from 340 to 232 and have a lot of access skin i also have fibromyalgia and arthritis that is not able to be controlled at the present time it is my desire to run a mini marathon but i do not even know where to begin on setting myself on the proper program i do not have the money to go to a trainer so need some direction please thank you  vicki
  - will exercising the vaginal muscles make my vagina tighter

Cluster 1  (n=809)
 Top terms: flu, vaccine, shot, injection, vaccination, swine, child, chickenpox, influenza, virus, baby, depo
 Top tags: flu, injection, vaccines, vaccination, swine flu, shingles, virus, pregnancy, coldness, chickenpox
 Examples (original text):
  - can you test positive from having the hep b vaccine
  - why would a rn choose not to get her kids a flu shot as the grandparent is there anything i can do
  - my son had dtap polio chicken pox and mmr vaccines now can barely move
  - what reactions are likely after an immunization
  - can allergy shots be used to treat asthma

Cluster 2  (n=2482)
 Top terms: tooth, child, baby, mouth, daughter, son, lip, water, wart, lice, mole, gum
 Top tags: tooth, pregnancy, baby, mouth, coldness, drinking, wart, tongue, vision, burn
 Examples (original text):
  - okay so i am 16 and i want to grow about 3 more inches if i smoke hookah once or twice will i grow to my goal height
  - what are some warning signs for pregnant women when they are exercising
  - can you use egg whites on a burn  i read an article that said you can use egg whites to sooth and help heal burns like if you burn yourself with fire but not real bad is this true
  - would braces close wide gap between front teeth
  - 3 yr old son has small specks of blood on face after napping

Cluster 3  (n=1402)
 Top terms: period, pregnancy, test, sex, pill, control, birth, symptom, cycle, cramp, condom, breast
 Top tags: period, pregnancy, pregnancy test, sexual intercourse, birth control, spotting, condom, breast, nausea, cramps
 Examples (original text):
  - could i possibly be pregnant  last period may 16th unprotected sex on june 9th supposed to start june 16th still have not if you think i am pregnant when should i take a test  side notes  feel as if i start but do not   i was throwing up at 2 am on saturday the 15th i was nauseous the rest of the day   light cramps  the guy i had sex with says he only has a 3 chance of getting someone pregnant  i have been tired lately i also have been having light heartburn i think  if anyone can help me it would be greatly appreciated
  - pregnant unprotected sex a week before period period came on time and heavy with bad cramps as usual reg 28 day 4 yrs i had unprotected sex a week before my period started he ejaculated awa from me but im worried a little bit may have got it before he pulled out my period came on the dot when it was supposed to get it and was heavy at first then to moderate with bad cramps like i normally have basically my period came on time and was normal in length flow and cramps my periods have been regular for years i do not know when i ovulate or my latueal phase what are my chances of being pregnant
  - i am on the pill and a condom was used pregnant  i have been on the pill for over 5 years i am pretty good with taking it on time but occasionally i will forget a day but immediately take it when i realize ive missed it last weekend saturday i was about an hour late taking my bc the next day i had sex he was wearing a condom a week later getting cramps and what not like a period i usually take the pills continuously but i am scared so i am going to let myself have a period hopefully i am about to start the sugar pills today
  - i have been off depo shot for over a year and have no period i recently began having pregnancy symptoms am i pregnant  i was on the depo shot for 2 years and have not gotten the shot since september 2011 i have still not had a period and have been having pregnancy symptoms my fiance and i use the withdraw method i realize that you can still get pregnant this way after over a year of not receiving the shot i do not think the symptoms could be attributed to the shot i had a pap smear 9 12 12 and it came back fine so could i be pregnant and if not what could be causing the pregnancy symptoms
  - i skip my periods with my birth control trinessa i am sexually active and very busy so if it is unnecessary for me to have a period u would rather not have it my boyfriend and i use both condoms and my birth control is there a higher chance of pregnancy if i skip my pills or is there even a chance of pregnancy i take my pill pretty religously but i have had times where i have missed the time to take it and have taken it hours later

Cluster 4  (n=955)
 Top terms: cancer, prostate, tumor, brain, treatment, colon, symptom, radiation, lymphoma, chemotherapy, biopsy, risk
 Top tags: cancer, prostate cancer, colon cancer, brain tumor, chemotherapy, prostate, breast cancer, lung cancer, biopsy, radiation surgery
 Examples (original text):
  - how to treat leiomyosarcoma and rectal cancer at the same time
  - i have a basal cell carcinoma what specialist should i consult
  - i have recently had breast surgery removal of calcium deposits with cancer cells
  - can hot showers a day cause cancer  i love to take hot showers and take three 5 10 minute showers a day   morning noon and evening is there a danger of cell damage or cancer from this practice also are three too many i like the water very hot is this bad our water in our city is chlorinated should i be concerned
  - abscessed tooth symptoms and oral cancer symptoms similar

Cluster 5  (n=1275)
 Top terms: chlamydia, hiv, infection, yeast, sex, vaginosis, discharge, hpv, partner, std, vagina, symptom
 Top tags: vagina, chlamydia, yeast infection, burn, sexual intercourse, hiv, bacterial vaginosis, bacterium, sexually transmitted disease, wart
 Examples (original text):
  - i was curious about anal used mothers sex toy didnt clean it at risk for stds do not think it was used in a while i was curious and i found a vibrator and i used it i put a condom on it but condom broke i got tested for chlamidia and ghonorea both negative do you think i am at risk for hiv or anything else  also i used other sorta home made toys over a year ago and i just got worried i could have done damage to my body have not had any negative symptoms and havnt used them since last year should i be worried everything is normal and during use nothing negative happened like bleeding of anything
  - can battery acid from a vibrator burn my genitals i thought i had a vaginal yeast infection but it turns out my vibrator was leaking battery acid and has burned my vaginal area included and up to my rectum what can i do to ease my pain
  - what causes bacterial vaginosis should i be worried if it this infection keeps popping up
  - every time i cough or sneeze i pass blood clots also through vaginal discharge and during intercourse
  - gonorrhea long term effects i have aged and look frail

Cluster 6  (n=726)
 Top terms: medication, drug, medicine, counter, prescription, med, treatment, blood, pressure, ibuprofen, allergy, effect
 Top tags: drug, over the counter, coldness, pregnancy, ibuprofen, pain, drug test, tuberculosis, depression, allergy
 Examples (original text):
  - i have been thinking about buying drugs online to save money is that a good idea
  - i am extremely agoraphobic despite help from my psychologist psychiatrist and medication why else can i do
  - is avinza a narcotic medication
  - does soma show up on a drug urinalysis
  - what medications and treatments are available to treat pagets disease of bone

Cluster 7  (n=850)
 Top terms: hand, arm, finger, shoulder, pain, wrist, elbow, thumb, rash, arthritis, numbness, joint
 Top tags: hand, arm, finger, pain, wrist, shoulder, rash, arthritis, coldness, swelling
 Examples (original text):
  - pain ring finger to the middle of arm before elbow for 45 days started after i held on to stop falling worse on lifting its not bad3 on 10 earlier it was more painful but now it does not hurt as much unless i type or write i have barely used my right hand for anything for the past month else the pain increases it starts hurting at one point about five fingers from my wrist but pain goes away completely if i press down on it it also hurts in the area below middle to pinkie i got an xray done already so nothing there shd i get an mri if so only for wrist or forearm also or shd anti inflammatories be enough thanks
  - i have got a wrist and palm injury in my right hand
  - i am kite surf instructor but have spine condition pain too much too handle
  - would the arm of a 1yr old swell immedietely if it were broken
  - why does the pain from frozen shoulder go down my arm and sometimes makes my hand tingle

Cluster 8  (n=489)
 Top terms: herpe, outbreak, partner, sex, sore, simplex, virus, area, bump, sign, symptom, type
 Top tags: genital herpes, herpes, vagina, virus, sexual intercourse, cold sore, lip, herpes simplex, sore, acne
 Examples (original text):
  - can herpes be spread by bed bugs if a person infected with herpes is bitten by a bed bug can another person bitten by the same bug get infected with herpes
  - what other than a yeast infection could it be if the medicine does not work and more symptoms start occuring  for a few months now i have had constant vaginal itching and burning and after using yeast infection medicine multiple times it still has not gone away and now blisters have begun forming after looking up genital herpes i have noticed that i have nearly all the symptoms for it but i have yet to have sexual activity could it still be possible or is it something else
  - how is herpes simplex treated
  - herpes transmission somewhat complicated question i have genital herpes but no outbreaks after the initial one my partner got it from me and has genital outbreaks we are wondering if we are able to spread it to each other in areas where we have not had outbreaks during shedding not active outbreak seems obvious that we could spread it around while there are lesions so we are careful during her outbreaks but the big mystery is  can we spread it to new areas during shedding    thanks a million we have been searching hard for the answer 
  - are there any ointments that can speed up the healing process of genitle herpes already on 3 day valtrex should i be worried if it dosent clear up by then

Cluster 9  (n=381)
 Top terms: shingle, pox, chicken, vaccine, pain, rash, risk, child, scalp, husband, area, treatment
 Top tags: shingles, pain, chickenpox, rash, vaccination, virus, blister, scalp, burn, injection
 Examples (original text):
  - can shingles occur in the scalp
  - i am 33 years old i need shingles vaccine but am i too young to have it
  - can i get shingles after having scarlet fever as a child  my mother is 80 and has been iin a lot of pain the last few weeks with her hip now she has a red rash very painful just appeared tonight it has a burning sensation is is possible she could have shingles should i take her to the er for treatment  thank you
  - is shingles contagious
  - i am 55 what  of getting shingles after getting shot as a side effect want to get the shot but not if i have a high risk of giving me shingles

Cluster 10  (n=793)
 Top terms: penis, sex, testicle, erection, vagina, std, tip, area, bump, shaft, condom, foreskin
 Top tags: penis, vagina, burn, pregnancy, pain, ejaculation, testicle, sexual intercourse, rash, masturbation
 Examples (original text):
  - pain when urinating inconsistent urination painfull ejaculation painfull mastrubation etc oh gosh im in all sort of trouble here and its given me anxiety over the past 1 5years ive been having this condition it all began when my urine penis started smelling cheesy after urination then later on when i was laying in bed and rising up i could feel from pelvic area like something is almost pushing my urine out it happened all the time then came premature ejaculation painfull urination painfull mastrubation painfull ejaculation also when i drag back my foreskin pain help please
  - i am experiencing a problem keeping an erection are there natural remedies that can be taken for this
  - will my glans burn recover  accidentally i got a very hot water on my penis the water hit a small area of glans and the area under it now rubbing it either by hand or cloths has some sort of annoying feeling i would like to know whether recovery is possible or not if yes please tell me how
  - why will not my penis stay hard when in pregame  i get hard quite easily when around my gf but then all of a sudden when it comes to me taking my jeans off it goes down why  when it does decide to work i really do love sex with her so what is causing this also i can not cum when she tries to give bj
  - can a yeast infection cause pain in the urethra as well as the head of the penis  originially thought to have prostatitis a month ago have been experiencing penis soreness urethra pain some redness gonnohrea and chlamydia negative and 3 weeks of cipro provided no relief now thinking it may be fungal could this be a yeast infection and if not what could it be

Cluster 11  (n=1040)
 Top terms: diarrhea, stool, colitis, bowel, movement, constipation, blood, colon, disease, crohn, gastroenteritis, colonoscopy
 Top tags: diarrhea, bowel movement, ulcerative colitis, constipation, diet, "crohns disease", pain, colonoscopy, stomach, colon cancer
 Examples (original text):
  - i have started a low sugar and low wheat diet and i keep going to the toilet more than normal is my diet the reason
  - how to heal crohn is ulcers  my husband has crohns he has no pain and is mostly symptom free he was diagnosed with crohns from a biopsy after intestinal cancer he has had two surgeries to remove ulcers and he has deep ulcers again the doctor wants to put him back on endocort but that was the drug he was on when he got the ulcers back after the second surgery he has been checked for inflammation but the tests always come back well within normal limits what can we do to heal the ulcers
  - my butthole hurts and there is a bump and it hurts whenever i sit or move i assume it happened a few days ago when i was trying to hold in my bowel movement but i could not so i went to the bathroom but i do not remember anything happening to me for the rest of the day or the next but on sun today8th 9th its been bothering me severely and it hurts whenever in doing anything that requires me to move my lower body
  - what are the risk factors that cause malnutrition for someone with ulcerative colitis
  - how to treat diarrhea in a 6 month old baby who has had over a week and his bottom is raw

Cluster 12  (n=787)
 Top terms: kidney, urine, uti, bladder, infection, stone, pain, urination, blood, antibiotic, tract, test
 Top tags: pain, urination, kidney stone, kidney, bladder, vision, burn, urinary tract infection, pregnancy, antibiotic
 Examples (original text):
  - how to manage large kidney stones that can not be passed on its own while being 16 weeks pregnant and not harm the baby stones are roughly 5 8 mm in size recurrent episodes of pain over the past 2 3 weeks should i stent how safe are the pain meds percocet and dilaudid for the baby
  - why does my urine smell
  - is it safe to swim in a lake if you have a uti  i have a bladder infection and began antibiotics two days ago we are going to the lake this week can i swim in the lake or should i avoid that
  - my urine has a bad smell and is cloudy what can be wrong
  - is there a drug to strengthen the bladder muscle

Cluster 13  (n=1791)
 Top terms: sex, condom, sperm, intercourse, ejaculation, pill, boyfriend, chance, birth, semen, girlfriend, control
 Top tags: pregnancy, sexual intercourse, condom, ejaculation, sperm, vagina, period, birth control, masturbation, anus
 Examples (original text):
  - how effective are male condoms at birth control
  - i am pregant does everything i wear have to have cotton in it  my boyfriend is sure he read that now that i am pregnant everything i wear has to have at least a percentage of cotton in it i know that my panties should be cotton but i can not find answers about the rest of my wardrobe this is making it difficult to find suitable pants for my work uniform
  - gave boyfriend oral then ate an hour later got home and felt mucus in throat and spit it went down my body pregnant  gave my boyfriend oral and swallowed some then ate a small burger and some coke an hour later i took a shower while in the shower i felt then need to spit i felt like mucus so i did spit worried that spit contained sperm and traveled down my body and got me pregnant i did not insert anything in not even with fingers and i was obviously standing while in shower but spit went through middle of my stomach so i am sure it passed by my vagina i am only 19 help can i be pregnant
  - my boyfriend and i had sex using both the pill and a condom the condom broke should i worry about rubber inside me  we think most stayed on his shaft and i found a small piece about 34 inch x 12 inch like it ripped and a portion tore off do i have to worry about this causing tss 
  - sexual health how can i increase my semen due to much masturbation i m starting masturbation in the age of 14 yr regularly many time in a day now i am 28 yr i have following major problems 1 erectile dysfunction 2 premature ejaculation 3 low sperm count after masturbation semen has comes only 4 5 drops and very slander can i pregnant my wife in this situation please sir help me

Cluster 14  (n=1720)
 Top terms: foot, leg, knee, arthritis, osteoarthritis, pain, ankle, osteoporosis, bone, toe, hip, doctor
 Top tags: foot, arthritis, leg, knee, pain, osteoarthritis, osteoporosis, ankle, walking, toe
 Examples (original text):
  - can i sit in a sauna and steam room with a broken ankle
  - is hand foot and mouth the same as rubella is hand foot nd mouth the same as rubella
  - i have lump in the soft tissue of my left leg about 1 inch above my ankle i have had an ultrasound and a xray and i have been told it is not life threatening but the doctor will not tell une but it is hurting and swelling and i am taking pain killers every day me what it is until he sees me again in june
  - i have had 2 knee operations knee feels like loose
  - i am tired legs feel heavy and have disequilibrium doctors can not pinpoint problem

Cluster 15  (n=1141)
 Top terms: surgery, hernia, cyst, option, hysterectomy, surgeon, procedure, fusion, operation, doctor, opinion, bypass
 Top tags: surgery, hernia, hysterectomy, pregnancy, cyst, insurance, smoking, ovary, pain, healing
 Examples (original text):
  - i just find out by a x ray i have 10 stables inside of me is it normal after surgery
  - just discovered a 4cm x 4cm cyst inside my left ovary via ultrasound will ovary need removal
  - i had a back fusion at l4 s1 now have anterlothesis of l4 on l5 do i need surgery
  - what are the treatments for a hernia
  - is it normal to break out after a laser resurfacing procedure and what should i do

Cluster 16  (n=1450)
 Top terms: pregnancy, baby, test, trimester, miscarriage, fertility, birth, woman, abortion, symptom, infertility, risk
 Top tags: pregnancy, baby, miscarriage, pregnancy test, fertility, infertility, sexual intercourse, postpartum depression, ovulation, ultrasound
 Examples (original text):
  - how effective are foam and male condoms in preventing pregnancy
  - can newborn babies be born addicted to prednisone if the mom took it for asthma in the last trimester
  - i take steroid prednisolone for ivfpregnancy due to auto immune issue but i have herpes 2 will this affect baby
  - inserting finger into girls vagina leads to pregnancy
  - i had sex in the 5th week of pregnancy and saw vaginal bleeding why

Cluster 17  (n=2166)
 Top terms: skin, acne, rash, face, spot, eczema, cream, bump, product, dermatitis, scar, area
 Top tags: skin, rash, acne, arm, atopic dermatitis, itch, vision, burn, scar, blister
 Examples (original text):
  - guest in my home has scabies do i have house sterilized or will a good cleaning do it we have not had skin contact he is getting treated and i am having a general cleaner come in this afternoon to change linens etc is this sufficient i have never had anything like this in my home somewhat disturbed
  - what is the best moisturizer for older skin
  - what medicines can cause your skin to turn blue
  - i have blisters on the inside of my lips and on my tongue what kind of doctor should i see  i also have multiple sclerosis hypothyroidism incontinence bipolar borderline personality disorder and an addict i smoke i do not drink or eat spicy anything i am a 37 year old woman with no children
  - i use a high spf lotion on my face daily especially when going outdoors why do i have sun spots and white freckles

Cluster 18  (n=454)
 Top terms: sleep, bed, bedbug, aid, insomnia, nap, bug, apnea, ambien, fatigue, baby, melatonin
 Top tags: bedbug, pregnancy, baby, over the counter, drinking, insomnia, melatonin, stress, exercise, coldness
 Examples (original text):
  - had a stroke on the brain in 2012 its 2016 i cant get no more than 5 hours of sleep a day
  - how much nap time does my baby need
  - when is the best time to take vitamins in the morning or at bedtime
  - how much sleep should my 4 yr old be getting a night
  - major sleeping difficulties for the past month due to anxiety restless legs meds dont help no caffine reg schedule sleeping pills do not help stopped melatonin since i was getting pounding headaches concussion about one month ago but all symptons have disappeared daily activities same often up for days at a time major depression for years but now symptoms seems to be leaning more toward bipolar it seems like a can fall asleep on the couch better than in the bedroom  please help  meds include xanax paxil topomax and requip in addition to occasional percoset or oxycodone to treact chronic pancreatitis

Cluster 19  (n=715)
 Top terms: ear, infection, hearing, pain, doctor, head, tinnitus, neck, fluid, earache, eardrum, sound
 Top tags: ears, pain, ear infection, tinnitus, head, antibiotic, coldness, neck, vision, pressure
 Examples (original text):
  - why am i hearing my heartbeat in my right ear  just recently i have started hearing my heartbeat in my right ear this came on suddenly i am a 66 year old female with no particular health issues what could be the cause of this anything to worry about
  - should i go to the er for severe right earache or wait until monday is appointment  i am 20 on thursday i woke with dull throbbing it has gotten worse i have a 101 f fever i think the throbbing is still there but now my ear canal is so blocked i can not feel anything in the canal but there is still sharp stinging pain in what feels like the back my outer ear is swollen the pain extended to my throat chewing or hiccups feels like my ear is tearing i have almost zero hearing in this ear er or wait
  - i have had white noise with corresponding hearing loss in my left ear for 2 months what causes it and can it be fixed i have had an mri    negative for tumor ms or anything else that might be causing it
  - does lipo flavonoid reduce or cure tinnitus
  - i am having ruptured eardrum one doctor suggested to have surgery and other suggested to wait for 3 months

Cluster 20  (n=913)
 Top terms: fever, pain, throat, body, symptom, headache, grade, cough, son, chill, temperature, flu
 Top tags: fever, running, pain, cough, headache, flu, antibiotic, coldness, swelling, sweating
 Examples (original text):
  - my baby ate her on poop my baby ate poop 4 days later she is sick weezing coughing and high fever for 4days straight i took her to the doctor and they said shes fine just a normal cold i told them what happen and they just said she should be fine but if she still has a fever next week come back what should i do and is her symptoms related to her eating her poop
  - i have persistent headache and i feel like i have lowgrade fever help  hi so i am a headache everyday it is not too bad though it is completely bearable but a little distracting and i have noticed that i have lowgrade fever most of the time or mostly everyday but just like the headache it is bearable i can not just shrug this feeling off this have been occurring for two or three months now i am hopefully going to the doctor in a few days and get myself checked
  - why do i feel lightheaded fatigued and sweat during sleeping no fever  i am an almost 37 yr old female with a lot of stress right now dr put me on effexor and i started not being able to sleep having bad headaches feeling lightheaded and constipated i took it for 2 weeks and he told me to stop when i called him he called me in something else but i am afraid to get it i have been on paxil prozac and celexa and never felt this horrible i have not taken anything in almost a week but feel lightheaded many times thoughout the day any ideas what could be wrong
  - can i have strep without fever  my 4 year old son was diagnosed with strep throat 3 days ago last night i was fine one minute and suddenly felt like i would been hit by a ton of bricks body aches headache  sore throat and general feeling crappy but no fever is there any point in dragging myself out to doctor when i feel so miserable is it possible to have strep without fever i do not have any runny nose stuffy nose or cough not a cold 
  - how long should i wait before bringing my 11 yr old with flu symptoms to our family dr it has been 8 days initial symptoms were nausea high fever severe headache loss of appetite and fatigue those lasted about 2 days now she is very tired little appetite sore bellynausea and has a sore throat and cough

Cluster 21  (n=981)
 Top terms: headache, migraine, head, pain, nausea, dizziness, sensitivity, side, pressure, symptom, neck, vision
 Top tags: headache, migraine, nausea, pain, head, dizziness, vision, neck, ears, photosensitivity
 Examples (original text):
  - i have heavy pain in both side of my head that causes dizziness sometimes in my back and neck
  - i have been taking propranolol for the chest pains now have headaches and pain on left side of head and body
  - i have been having very sharp stabbing pains down through the top rtrear of my head the pain almost knocks me down i have been having these pains for 6 7 weeks i have had no previous head injuries they just started out of the blue they are not headaches they are in  a dime sized spot on top of my head right side just off center back portion top of head does that make any sense these pains happen wether  i am standing or laying down thank you for your time
  - can diabetes cause you to have chronic migraines
  - could clonazepam be causing nausea and dizziness or is it bupropion

Cluster 22  (n=631)
 Top terms: marijuana, smoking, test, drug, cigarette, smoke, urine, smoker, cocaine, system, nicotine, weed
 Top tags: smoking, marijuana, drug test, drinking, pregnancy, quit smoking, drug, cocaine, lung, nicotine
 Examples (original text):
  - i smoked cigs for 1 month averaging about 3 a day just wondering if any irreversible was done i did quit since then i started smoking for a month after a period of depression a couple of months ago 2 3 cigs most days with a couple more on bad days i would estimate i probably had 4 packs total over the period i went cold turkey as i started to get my life together and hated the ill feeling from them i exercise regularly and eat healthy and i am still young i would just like to clear my head and hear that i did no damage permanent to my lungs i know it takes a bit to recover hopefully to 100
  - what is less damaging smoking 3 cigarettes per day or picking two days of the week and smoking 10 on each of those days  i know that light smoking still poses significant risks i know i can still get cancer one day from doing it so please do not answer by telling me that i am looking to significantly cut down in order to lessen the damage i am doing to myself while still indulging a little in one of my favorite activities so here are my two options a smoke about 3 cigsday about 20week or b pick 2 nightsweek say two nights when i am out socializing and smoke about 10 on each of those about 20week
  - drug test where to find
  - does quiting smoking improve semen motility and morphality levels or just prevent any further deterioration
  - problems keeping an erection i am an 18 year old male i weight 133 pounds and i am 58 i have problems keeping an erection it will get hard but it gets soft rather quickly without stimulation like a few seconds when it is hard it usually stays a sort of mostly hard but still soft penis i smoke cigarettes and marijuana but not to the extent where it would cause major damage i exercise 2 3 times a week and eat healthy although i do indulge in some junk food could i really have erectile disfunction at my age

Cluster 23  (n=1346)
 Top terms: heart, pressure, blood, chest, pain, failure, disease, bp, rate, hypertension, artery, attack
 Top tags: high blood pressure, blood pressure, heart, heart disease, chest, congestive heart failure, pain, exercise, heart attack, low blood pressure
 Examples (original text):
  - ekg says there was “moderate right axis deviation ” “normal sinus rhythm with marked sinus arrythmia ”
  - is there evidence that statins increase life expectancy for people without heart disease
  - i felt like electric shock like feeling which became sharp on my left chest
  - would lad lesion cause tachycardia
  - heart attack i am a parapalegic and the other night when i was going to bed i had a burning sensation that started in the chest area and moved to the back to the point my upper torso was burnig all the way around i was in total discomfort i felt a heaviness in my chest this lasted for about 3 hours are these symptoms of a heart attack or are these symptoms that have to do with my spinal cord injury i have had these symptoms before but never to the degree i had the other night

Cluster 24  (n=1314)
 Top terms: disease, disorder, parkinson, symptom, epilepsy, dementia, seizure, schizophrenia, depression, lupus, syndrome, people
 Top tags: bipolar disorder, "parkinsons disease", epilepsy, dementia, schizophrenia, seizure, celiac disease, depression, lupus, lactose intolerance
 Examples (original text):
  - what are the dietary restrictions for celiac disease gluten
  - where can i go for help for bipolar disorder
  - what is the treatment for the common cold
  - is erythema multiforme an autoimmune disorder
  - i was diagnosed with mononucleosis 7 months ago since then i have not been able to recover

Cluster 25  (n=1548)
 Top terms: cough, pneumonia, sinus, throat, nose, asthma, allergy, infection, bronchitis, cold, symptom, sinusitis
 Top tags: coldness, cough, sinus infection, throat, pneumonia, cold, nose, chest, allergy, asthma
 Examples (original text):
  - i have had a pneumonia shot can i get either a sinus infection or walking pneumonia from my 6 year old grand daughter  and can i be a carrier to others in my age group
  - how soon should my 14 year old wait before returning to school having been diagnosed with pneumonia
  - my husband has had a fever and a cough for over 10days now not pneumonia had xray 4 days ago what could it be  his fever ranges between 100 103 he is coughing up lots of mucous but does not feel any sinus congestion no runny nose he has been on an antibiotic for 5days now he says it is helping a little but it is not taking care of the fever or cough just making the mucous less green he had a chest x ray 5days ago but it did not show pnumonia should he be x rayed again he is also very pale in color as well as is experiencing dizziness headache and nausea had a cbc blood work came back normal
  - what are the symptoms of pneumonia
  - is viral pneumonia contagious

Cluster 26  (n=575)
 Top terms: breast, lump, nipple, mammogram, chest, cancer, pain, milk, armpit, ultrasound, biopsy, side
 Top tags: breast, nipple, pregnancy, lump, mammogram, breast cancer, pain, milk, swelling, ultrasound
 Examples (original text):
  - hi my husband has a lump on his head that has been there for 2 weeks and now has a lump under his armpit hi my husband has a lump on the back of his head for about 2 weeks now and it appeared there without injury we were a little concerned but now i am really concerned because he has another lump under his armpit that has been there for 2days
  - i have had rapid breast growth want breast reduction surgery what age should i typically get this surgery
  - why do my breasts hurt and feel and look like they are getting bigger  for almsot 2 weeks now my breasts have been painful and feel like and look like they are getting bigger i instantly thought of being pregnant but i am on the mirena birth control and although its not 100 effective i still did not believe i were pregnant i took a home pregnancy test from my local drugstore and it was negative why are my breasts feeling this way people i know have been commenting on them looking huge or bigger than normal
  - will health insurance through the marketplace cover genetic testing for breast cancer
  - i am 44 yrs tubal ligation 18yrs ago last period came for 1 day my breast hurt for 3 wks and vomiting am i pregnant

Cluster 27  (n=955)
 Top terms: weight, diet, loss, pound, calorie, lbs, gain, fat, body, lot, protein, exercise
 Top tags: weight, diet, exercise, weight loss, weight gain, food, calorie, height, pregnancy, corpulence
 Examples (original text):
  - what weighs more muscle or fat  im just wondering about weight gain due to muscle growth my wife has been working out for some time with weights and cardio training but she is finding that her weight has been fluctuating and at times gains weight a little bit
  - are diet pills safe for teenagers if so which ones are
  - had total knee replacements i am not feeling good no energy depressed no appetite have lost weight
  - what can i do to gain back my missing pounds and feel healthy again  i have been sick and lost 17 pounds i am fatigued all the time and look poorly i want to gain my wieght back and feel good again as quickly as possible
  - would you kindly suggest diet for a pregnant woman

Cluster 28  (n=1074)
 Top terms: antibiotic, infection, throat, strep, amoxicillin, penicillin, treatment, doctor, bacteria, meningitis, pneumonia, virus
 Top tags: antibiotic, amoxicillin, throat, penicillin, virus, bacterium, strep throat, sore throat, pain, sinus infection
 Examples (original text):
  - can an antibiotic through an iv give you a rash a couple days later
  - can taking multiple antibiotics cause redness and dryness of vagina
  - i need relief from chronic epididymitis
  - is clindamycin effective in treating syphilis
  - why it is necessary to take 2 antibiotics for a diverticuilitis infection will taking just the cipro work

Cluster 29  (n=1104)
 Top terms: blood, vitamin, test, anemia, level, iron, result, atherosclerosis, cell, hemoglobin, work, disease
 Top tags: anemia, blood test, vitamin, pregnancy, atherosclerosis, vitamin d, exercise, iron, coldness, low blood pressure
 Examples (original text):
  - is all vitamin d the same
  - what is black measles when i was young i had them now in my fifties i have a lot of health problems could it be because of them and what damage do they do to your body i know they have not been heard of in god know how long is there any way to know after all of these years after having them to get information on them
  - can menstrual cycle affect diabetes blood work
  - is it safe to give a 2 month old pedialyte on a daily basis
  - how to correct vitamin b12 deficiency

Cluster 30  (n=1755)
 Top terms: pain, side, rib, back, neck, shoulder, muscle, chest, spine, leg, fibromyalgia, area
 Top tags: pain, back pain, burn, fibromyalgia, neck, leg, exercise, shoulder, muscle, arthritis
 Examples (original text):
  - broken collarbone 3 5 cm overlap its been three weeks after break and still feels broken or loose
  - i have an acute dextroscoliosis i feel pain when i skip meals
  - does lidocaine cure canker sores on your throat  i found a small white sore on my throat and i was prescribed lidocaine and it numbs the pain but i was wondering if it cures it at the same time before it gets worse
  - was skiing and fell on my knee cap now 3 days later i just heard a pop and its throbbing and its severe pain  they did an xray but they did not have an mri machine available i will not be able to access a doctor for another 3 9 days a half hour the pop happened and it really hurts on a scale of 1 10 10 being like you have just been shot its more of an 8 7
  - can age appropriate arthritis be exaserbated by barometric pressure change or other weather conditions  im 55 very fit and athletic and often suffering with joint pain and stiffness like never before and all over my body not sure where it came from or how to get rid of

Cluster 31  (n=891)
 Top terms: period, sex, cycle, menopause, birth, pill, boyfriend, control, woman, may, discharge, ovulation
 Top tags: period, pregnancy, sexual intercourse, ejaculation, spotting, menopause, ovulation, birth control, irregularity, vagina
 Examples (original text):
  - my period only last 36 48 hours which is my norm is that why i have had 2 yrs of no luck getting pregnant  my husband and i have been trying for two years to have a child i am turning 30 next month and in my family after 30 equals issues my normal period is only 36 48 hours could this be preventing me from getting pregnant
  - spotting on day two of my period could i be pregnant i have had my loop taken out 71015 and have had unprotected sex a few times after that hoping to fall pregnant my period was meant to start 12102015 it now day two on my period and i have only been spotting which is very un usual could i be pregnant
  - my period has been late by 4 days i am trying to conceive please help
  - had sex a week later instead of my normal period had alittle blood when wiped now im sick to stomach and nipples itch could i be pregnant
  - i have not had a period for 4 months

Cluster 32  (n=801)
 Top terms: eye, vision, cataract, eyelid, circle, doctor, pain, discharge, pinkeye, nose, problem, face
 Top tags: eyes, vision, pink eye, cataract, eyelid, swelling, burn, coldness, skin, sty
 Examples (original text):
  - seems i have got chemical in eye from eye cream anything i can do to get relief from burning i have tryed to rinse
  - i have been suffering from pressure and pain behind eyes for almost three years
  - my 2 friends also underwent through lasik procedures but i am afraid about any side effects can anyone guide me about the possible risks thanks
  - i was checking my husband is testicles and i felt a peanut size lump above his left testicle not sure if we should worry  we are both without insurance right now it would be nice to know if i should just keep an eye on it or if it will just go away
  - my eyelids sag and i have heard that botox can lift my eyelids how does this work will not it look strange

Cluster 33  (n=1098)
 Top terms: stomach, pain, nausea, abdomen, diarrhea, side, cramp, doctor, feeling, vomiting, symptom, daughter
 Top tags: stomach, nausea, pain, diarrhea, vomit, pregnancy, cramps, fever, bloating, vision
 Examples (original text):
  - after a bowel movement abdominal pain puts me on my knees severe nausea to the point i do vomit
  - how long can flucold causing bacteria live outside the human body  pretty much what i asked my mom had a nasty stomach bug which i am pretty sure was the flu with all the nasty symptoms that go along with it not thinking she used my computer for something to do i have been avoiding it using a friend is computer now but how long do i need to i have stuff i need to do on it so how long do i need to wait before i do not need to worry about contracting it myself
  - i began to feel a sudden twitchspasm feeling in my stomach went to er twice no solution
  - pain in upper right quadrant of abdomen may indicate what 18 year old grand daughter had gallbladder stmptoms but no stones doctor said bile was crystalizing and bladder was removed still having problems food test showed food slow at passing from stomach to intestines smaller portions have not helped couple crackers may induce vomiting what are possible causes
  - pain in testicles lower abdomen rectum is it fatty liver

Cluster 34  (n=581)
 Top terms: hair, loss, scalp, head, shampoo, woman, treatment, chin, cause, growth, man, beard
 Top tags: hair, hair loss, scalp, head, ringworm, stress, lice, pregnancy, acne, itch
 Examples (original text):
  - my hair has been thinning over the past few years taking minoxidil is it safe
  - how effective is laser hair removal at treating keratosis pilarus especially on arms
  - does xanax show up in hair sample
  - does vinegar stop hair loss in women
  - can perming a girls hair kill head lice and nits

Cluster 35  (n=513)
 Top terms: diabete, sugar, diabetes, type, blood, insulin, level, diabetic, disease, glucose, diet, doctor
 Top tags: diabetes, blood sugar, type 1 diabetes, insulin, type 2 diabetes, exercise, diet, high blood pressure, injection, pregnancy
 Examples (original text):
  - do doctors immediately prescribe medication to control blood sugar or do they wait to see if diet and exercise help
  - i was screened early for gestational diabetes 10w it came back high so now i do the 3 hour i am worried and scared my mother and brother both have diabetes and i am just really worried about this i had a hard time getting pregnant over 2 years and just do not want to have to worry about this is there any thing i can do to calm me down or help me manage it if i do have it i am feeling very stressed
  - is garlic safe for my diabetic dog 8 year old bichon with atypical cushings and diabetes he takes 7 units of novalin insulin bid flax hull and melatonin 3mg bid sugars are consistantly high and is going blind
  - does high sugar intake while pregnant mean a big baby
  - does niacin affect the bodys ability to produce insulin

Cluster 36  (n=1605)
 Top terms: insurance, health, plan, medicare, coverage, income, care, exchange, medicaid, marketplace, employer, company
 Top tags: insurance, health insurance, medicare, affordable care act, medicaid, health insurance exchange, health insurance marketplace, obamacare, dental, vision
 Examples (original text):
  - i am a disabled veteran on medicare am i affected by the affordable care act
  - is liposuction covered by insurance
  - i manage a medical office with 3 employees  rather than offer a health insurance plan we pay 50 of the employees premium so if they purchase their insurance through the marketplace will we no longer be able to do that
  - will they accept obamacare at any hospital
  - i am 50 years old and am currently on medicaressdi i have part a b and d  i also have aarp hospital indemnity by united healthcare i also have an aarp rx drug plan is this enough to cover what is needed under the affordable care act

Cluster 37  (n=806)
 Top terms: anxiety, depression, attack, panic, disorder, symptom, stress, medication, med, heart, pain, xanax
 Top tags: anxiety, depression, panic attack, stress, pain, heart, fear, chest, drowsiness, drug
 Examples (original text):
  - i would appreciate advice on a drug regimen that will restore serotonin while dealing with short term anxiety symptoms
  - should medical marijuana be used to treat anxiety disorder
  - my primary doctor diagnosed me with anxiety disorder and prescribed xanax and zoloft why do i still have panic attacks
  - why do my hands shake involuntarily it happens when i am trying to do something precise like pour sugar in my cup it also happens after i have eaten something so my blood sugar is not low and i do drink but i have ruled out delirium tremors i had eaten something tonight and then when i went to bring my chicken wings to the table i dropped them on the floor when i had to drive because my gf was too tired my anxiety was through the roof i feel like my anxiety goes hand in hand with my shaky hands but i do not know the cause
  - i served in the navy and now have panic disorder my family does not believe in mental illness are they right

Cluster 38  (n=1954)
 Top terms: overdose, pill, medication, control, birth, effect, prescription, dose, tylenol, counter, aspirin, tablet
 Top tags: drug overdose, pregnancy, injection, birth control pill, ibuprofen, over the counter, acetaminophen, birth control, coldness, drinking
 Examples (original text):
  - does prozac cause weight gain what about zoloft
  - what are the ingredients inibuprofen  i take a 600mg ibuprofen only as needed for nerve pain my question is what ingredients are in this medication i have a legal prescription for it
  - my husband is taking 40 mg of prozac and is really depressed and has thought of suicide what do we do
  - i swallowed 20 tablets of 40mg citalopram whay should i do
  - sleeping pills for traveling what should i take to sleep on the plane im traveling to israel from miami fl and am super scared of flying i would like to sleep on the plane so i dont get nerve recked whats safe to take that i can bring on the plane with me and that i will be able to wake up after my flight lands

Cluster 39  (n=477)
 Top terms: gallstone, reflux, heartburn, hernia, gerd, gallbladder, disease, pain, symptom, liver, gall, doctor
 Top tags: heartburn, gallstone, pain, hernia, burn, pregnancy, gallbladder, stomach, acid reflux, diet
 Examples (original text):
  - i had ovarian cancer and reflux surgery i still deal with constant nausea i can barely eat and i am unable to live life and go anywhere
  - my fart hurt the back of my head on the left side  this is a stupid question but i farted and the left side of the back of my head started hurting for a few seconds did this fart damage my brain or anything
  - how are gallstones diagnosed
  - is it normal to feel liver pain after gallbladder removal  i had viral hepatitis from bad drinking water in a 3rd world country when i was 9 years old now i wonder if the gallbladder stones are associated with this because after 5 days of having the gallbladder removed i am feeling liver pain thank you roxana casas
  - how can i cure acid reflux  i ate a large meal and went to bed right after and 4 hours later woke up with a burning throat gargled some cold water and saw some blood got worried gargled couple of more times but the bleeding got worse rushed to a doctor and got some medications but i still sometimes feel its hard to sleep and when i wake up my throat feels very dry even though i have woken up couple of times to drink water

Cluster 40  (n=372)
 Top terms: hypothyroidism, thyroid, hyperthyroidism, synthroid, weight, nodule, medicine, level, medication, hypothyroid, hormone, tsh
 Top tags: hypothyroidism, thyroid, hyperthyroidism, weight, exercise, pregnancy, hives, hormone, diet, tsh levels and pregnancy
 Examples (original text):
  - what should i do if i suspect an overdose of thyroid
  - i have all the classic signs of hypothyroidism but lab results normal where to go from here
  - im 27 and ive had almost all of the symptoms since i was about 6 years old could it be possible i have hypothyroidism
  - what would make my tsh levels suddenly jump to critically high level when i feel the same  i take synthroid 1 25 daily and yet my tsh is now 28 3 increased from 23 on friday why would it jump so quickly my dr is on vacation and i just want to make sure this is ok to wait until thursday when he comes back 
  - how is hypothyroidism treated

Cluster 41  (n=1608)
 Top terms: baby, problem, wart, way, hive, cold, child, woman, tip, body, daughter, face
 Top tags: pregnancy, exercise, coldness, cold, baby, wart, hives, transient ischemic attack, my daughter cries, acne
 Examples (original text):
  - can i transmit genital warts seventeen years after having them removed
  - i have been feeling extremely exhausted and unable to do basic tasks need advice
  - what causes hives
  - my body has not been feeling good at all what can be wrong
  - what do hives look like when they start to clear up

Cluster 42  (n=259)
 Top terms: hepatitis, liver, virus, infection, symptom, cirrhosis, risk, prognosis, effect, hepititis, mess, vaccine
 Top tags: hepatitis c, hepatitis, virus, liver, hepatitis b, infection, hepatitis a, concerned about atrophic liver, just found out i have hepatitis c?, sexual intercourse
 Examples (original text):
  - how do you get hepatitis c
  - is cirrhosis a form of liver cancer
  - who should receive antiviral therapy for hepatitis c virus
  - is it true that the hepatitis c virus cannot survive for more than two hours outside the human body
  - how many kinds of viral hepatitis are there

Cluster 43  (n=1111)
 Top terms: food, diet, allergy, meal, milk, cholesterol, water, meat, vegetable, fruit, baby, supplement
 Top tags: food, diet, drinking, food allergy, meal, milk, baby, fruit, vegetable, feeding
 Examples (original text):
  - can you be allergic to mold in your food
  - is it better for a type ii diabetic to eat corn or bread stuffing
  - why does coffee give me such an energy boost
  - i had energy supplements now feeling dizzy and passed out today
  - i am a woman 41 and eat a balanced diet and take a multivitamin do i need fish oil and calcium too

Cluster 44  (n=1037)
 Top terms: doctor, opinion, hospital, prostatitis, patient, medicine, ultrasound, test, exam, problem, neurologist, symptom
 Top tags: vision, pregnancy, prostatitis, ultrasound, pain, family, wart, exercise, i have many symptoms need second doctor opinion, bipolar disorder
 Examples (original text):
  - my 12th week scan showed everything is o k but radiologist suggested there is caudal regression syndrome
  - i have or think i have parkinsons disease when should i contact my doctor
  - admitted into the hospital for angioedema how to get swelling to go down
  - i have a patient of mine has pitting oedema difficult diagnosis any help would be appreciated
  - what kind of doctor do i need to see if my clavicle did not heal properly

Wrote assignments to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/clustered_short_questions.csv

=== Assessment Metrics (COSINE) ===
 Features: BERT(all-MiniLM-L6-v2) 384D + Tags→SVD(20D, EV=0.245) = 404D
 Global: silhouette (cosine)=0.0947

--- Cluster Size Stats ---
 k=45 | min=259 max=2482 mean=1055.4 median=955.0 std=497.8

--- Per-Cluster Averages ---
 cluster | size | mean silhouette (cosine)
       0 |  756 |         0.1496
       1 |  809 |         0.1826
       2 | 2482 |        -0.0442
       3 | 1402 |         0.0629
       4 |  955 |         0.1046
       5 | 1275 |         0.0293
       6 |  726 |         0.1313
       7 |  850 |         0.0705
       8 |  489 |         0.2225
       9 |  381 |         0.3454
      10 |  793 |         0.1147
      11 | 1040 |         0.0930
      12 |  787 |         0.1188
      13 | 1791 |         0.0540
      14 | 1720 |         0.0609
      15 | 1141 |         0.0612
      16 | 1450 |         0.1256
      17 | 2166 |         0.0671
      18 |  454 |         0.1302
      19 |  715 |         0.2222
      20 |  913 |         0.1614
      21 |  981 |         0.2006
      22 |  631 |         0.0795
      23 | 1346 |         0.1048
      24 | 1314 |         0.0291
      25 | 1548 |         0.0492
      26 |  575 |         0.1231
      27 |  955 |         0.0943
      28 | 1074 |         0.0357
      29 | 1104 |         0.0167
      30 | 1755 |         0.0497
      31 |  891 |         0.1478
      32 |  801 |         0.1747
      33 | 1098 |         0.1085
      34 |  581 |         0.2235
      35 |  513 |         0.1948
      36 | 1605 |         0.2404
      37 |  806 |         0.2023
      38 | 1954 |        -0.0196
      39 |  477 |         0.1062
      40 |  372 |         0.2742
      41 | 1608 |         0.1300
      42 |  259 |         0.4273
      43 | 1111 |         0.0503
      44 | 1037 |         0.0148

--- Structure (cosine on scaled space) ---
 Avg intra-cluster cosine similarity: 0.2640 (higher = tighter)
 Mean inter-centroid cosine similarity: -0.0196 (lower = better)

--- Tag Coherence (avg) ---
 Purity=0.371 | Entropy=4.383 | Intra-Jaccard=0.081

--- Model Selection Top Rows (by silhouette, cosine) ---
 k  silhouette  max_cluster_diameter
45    0.094747              1.283184
44    0.094228              1.283184
43    0.091827              1.283184
40    0.091323              1.281219
41    0.091055              1.265386
42    0.090828              1.253987
39    0.090085              1.258280
30    0.089970              1.283184
29    0.089522              1.283184
31    0.088863              1.283184
[UMAP] Saved 2D scatter to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_scatter.png
[UMAP] Saved embedding CSV to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_embedding.csv
[HClust] Saved dendrogram to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/hierarchical_dendrogram.png

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_timings.json

=== Runtime (seconds) ===
               load_csv_s:   0.2587
         nlp_preprocess_s: 114.6835
    vectorize_text_tags_s:  99.9546
         scale_features_s:   0.1707
        model_selection_s: 476.9480
     interpret_clusters_s:   0.4418
           save_outputs_s:   0.2511
        metrics_compute_s:  27.9097
                   umap_s:  31.8654
           hierarchical_s:   0.1774
          total_runtime_s: 752.6672

```

<img width="790" height="490" alt="CSCi E-108 v1 2 3 plot 1" src="https://github.com/user-attachments/assets/6037bc15-68a5-477a-a26b-2d82668bee5f" />

<img width="790" height="490" alt="CSCi E-108 v1 2 3 plot 2" src="https://github.com/user-attachments/assets/2a9780d8-5024-4874-950a-aae898ef032b" />

<img width="989" height="790" alt="CSCi E-108 v1 2 3 plot 3" src="https://github.com/user-attachments/assets/7f8111a5-ffa1-4117-815a-71fc28325cd2" />

<img width="1189" height="590" alt="CSCi E-108 v1 2 3 plot 4" src="https://github.com/user-attachments/assets/fc2f211c-34a0-4369-9463-d5395a4d3b19" />


In v1.2.3, I'm generally conducting a semi-supervised topic discovery pipeline over about 47,491 short medical questions with same techniques of data preprocessing and Natural Language Processing as I did in v1.1.3.

**Feature representation and modelling**

On the representation side, v1.2.3 is the first "BERT + tags" variant without dimensionality reduction on the BERT part. It encodes the POS-filtered sentences with all-MiniLM-L6-v2 into 384-dimensional sentence embeddings, then optionally L2-normalizes them. Tags are vectorized using a MultiLabelBinarizer and reduced via TruncatedSVD to a 20-dimensional tag space, capturing about 24.5 percent of tag variance. The final feature vector is a 404-dimensional concatenation of BERT and tag SVD. This is then standardized (StandardScaler) and L2-normalized to give a cosine-friendly feature space. Clustering is done with classic KMeans directly in this scaled space, with model selection over k from 10 to 45 based purely on cosine silhouette and a diagnostic measure of maximum cluster diameter.

The model selection behavior is quite consistent: silhouette grows very slowly with k, from around 0.069 at k=10 to around 0.095 at k=45, and maximum cluster diameter stays high (around 1.28-1.32) across the range. There is no obvious elbow; more clusters simply keep carving slightly tighter pockets out of a very overlapped medical space. The best k by silhouette is 45, so v1.2.3 ends up as a 45-cluster partition of the whole corpus, with no noise label. UMAP and a hierarchical dendrogram on cluster centroids are used only for visualization and qualitative structure, not for the clustering itself.

**Measurement, metrics, and structure**

Global cosine silhouette at the chosen k=45 is 0.0947, which is modest but not surprising given the density and overlap of medical questions. Cluster sizes range from 259 to 2,482, with a mean around 1,055 and a median of 955; in other words, this is a relatively "flat" partition where many clusters are big. Per-cluster silhouettes show a mixed picture: some clusters are clearly better than others, with a handful reaching 0.3-0.4 (for example cluster 9 at 0.3454 and cluster 42 at 0.4273). But two clusters, 2 and 38, even have negative mean silhouette, indicating that many of their points are closer to other centroids than to their own assigned centroid. The average intra-cluster cosine similarity is only about 0.264, which suggests relatively loose clusters compared to later v2.x pipelines that push this number far higher. The mean inter-centroid cosine is slightly negative at around -0.02, so centroids are, on average, almost orthogonal and reasonably separated in direction, but the within-cluster spread is still large.

Tag coherence metrics give more nuance. Average purity is about 0.371, meaning that roughly 37 percent of points in a typical cluster share the dominant tag. Entropy is 4.383, and average intra-cluster Jaccard over tag sets is 0.081. This tells me clusters are somewhat label-aligned but still cover a wide variety of tags, which is expected given that questions often have multiple tags and the space is highly multi-topic. Compared to v2.3.3 or v2.4.3, v1.2.3 is noticeably weaker on tag coherence: later pipelines, especially with spectral embedding or density-based detection, manage to carve out much purer islands.

**Runtime and performance**

From a performance perspective, v1.2.3 sits in the middle between the very heavy OPTICS run and the later Nyström + KMeans pipeline. Total runtime is about 753 seconds, a bit over 12 minutes. POS NLP with spaCy is the largest fixed cost, around 115 seconds, and BERT plus tag SVD vectorization adds another ~100 seconds. The real bottleneck is model_selection_s at about 477 seconds: repeatedly fitting full KMeans for every k from 10 to 45 on nearly 48k points in a 404-dimensional space is expensive. UMAP takes around 32 seconds, and metrics computation about 28 seconds, which are manageable. In my head, this version feels like a "brute-force" KMeans-on-BERT baseline: accurate embeddings, but relatively inefficient model selection and no additional geometry smoothing.

**Predicted semantic structure and behavior**

Despite the modest global silhouette, the qualitative semantics of the clusters are surprisingly good. Many of the 45 clusters line up with intuitive medical themes. There is a clear exercise and physical activity cluster, a vaccines and shots cluster, a dental and mouth cluster, multiple reproductive health and pregnancy clusters, a broad cancer and oncology cluster, distinct STD and vaginal infection clusters, a medications and OTC drugs cluster, musculoskeletal pain clusters for arms, legs, knees, and back, herpes and shingles clusters, male genital issues, diarrhea and colitis, kidney and urinary tract issues, respiratory and sinus infections, headaches and migraines, smoking and drug use, cardio and blood pressure, neuro and systemic diseases, diabetes, thyroid, gastrointestinal reflux and gallstones, anxiety and depression, and so on. Some clusters, like the hepatitis or shingles groups, look particularly coherent, which matches their higher silhouettes.

At the same time, a few large clusters behave like "hub" or catch-all regions. Cluster 2 and cluster 38, for example, show negative silhouette and combine multiple themes: children's mouth and skin issues, lifestyle questions, overdose and medication side effects, and mixed symptom cases. Cluster 41 is another broad "baby / hives / general problem" region. These clusters drag down global metrics and indicate areas where the embedding does not yet separate subtopics cleanly, so KMeans ends up mixing distinct disease systems into single groups. If I imagine using this version for downstream work, I would treat v1.2.3 as a first-generation, BERT-based taxonomy: it gives a broad, reasonably interpretable map of the corpus with many recognisable medical topics, but it still leaves substantial overlap and several overloaded clusters that later v2.x models, with PCA, spectral methods, and more sophisticated density logic, are designed to address.

## v1.4.3

```
Rows after POS cleanup: 47491
[Features] BERT(all-MiniLM-L6-v2) 384D + Tags→SVD(20D, EV=0.245) = 404D
[Shapes] Combined feature matrix: (47491, 404)
Searching for a good k (MiniBatchKMeans on sample, cosine metrics only)...
K-scan on sample (MiniBatchKMeans, silhouette cosine):
 k  silhouette
43    0.085863
45    0.085418
44    0.085281
42    0.085039
41    0.084271
40    0.084149
39    0.083447
36    0.083212
37    0.082797
34    0.082668
Selected k = 43 (sample silhouette = 0.086)
N=47491 > 12000. Running Spectral on subset then kNN propagate...
Final clustering: Spectral | k=43

=== Cluster Summaries ===

Cluster 0  (n=2297)
 Top terms: stomach, pain, diarrhea, nausea, stool, colitis, bowel, constipation, movement, blood, abdomen, doctor
 Top tags: stomach, diarrhea, nausea, pain, vomit, bowel movement, diet, ulcerative colitis, food, constipation
 Examples:
  - i have started a low sugar and low wheat diet and i keep going to the toilet more than normal is my diet the reason
  - how to heal crohn is ulcers  my husband has crohns he has no pain and is mostly symptom free he was diagnosed with crohns from a biopsy after intestinal cancer he has had two surgeries to remove ulcers and he has deep ulcers again the doctor wants to put him back on endocort but that was the drug he was on when he got the ulcers back after the second surgery he has been checked for inflammation but the tests always come back well within normal limits what can we do to heal the ulcers
  - how long does it take antibiotics to flush from your system after you stop taking when taking cefdinir 300 mg for a sinus infection i developed diarrhea and can not seem to stop it nurse line said to stop taking and it would take some time to get it out of my system how long is that time
  - my butthole hurts and there is a bump and it hurts whenever i sit or move i assume it happened a few days ago when i was trying to hold in my bowel movement but i could not so i went to the bathroom but i do not remember anything happening to me for the rest of the day or the next but on sun today8th 9th its been bothering me severely and it hurts whenever in doing anything that requires me to move my lower body
  - after a bowel movement abdominal pain puts me on my knees severe nausea to the point i do vomit

Cluster 1  (n=3610)
 Top terms: pain, leg, foot, arthritis, knee, hand, surgery, shoulder, arm, ankle, muscle, toe
 Top tags: pain, leg, arthritis, foot, exercise, knee, hand, surgery, swelling, finger
 Examples:
  - pain ring finger to the middle of arm before elbow for 45 days started after i held on to stop falling worse on lifting its not bad3 on 10 earlier it was more painful but now it does not hurt as much unless i type or write i have barely used my right hand for anything for the past month else the pain increases it starts hurting at one point about five fingers from my wrist but pain goes away completely if i press down on it it also hurts in the area below middle to pinkie i got an xray done already so nothing there shd i get an mri if so only for wrist or forearm also or shd anti inflammatories be enough thanks
  - can i sit in a sauna and steam room with a broken ankle
  - i need relief from chronic epididymitis
  - i have got a wrist and palm injury in my right hand
  - i am kite surf instructor but have spine condition pain too much too handle

Cluster 2  (n=256)
 Top terms: hepatitis, liver, virus, infection, symptom, cirrhosis, risk, prognosis, effect, enzyme, hepititis, mess
 Top tags: hepatitis c, hepatitis, virus, liver, hepatitis b, infection, hepatitis a, cancer, sexual intercourse, concerned about atrophic liver
 Examples:
  - how do you get hepatitis c
  - is cirrhosis a form of liver cancer
  - who should receive antiviral therapy for hepatitis c virus
  - is it true that the hepatitis c virus cannot survive for more than two hours outside the human body
  - how many kinds of viral hepatitis are there

Cluster 3  (n=2262)
 Top terms: weight, exercise, diet, diabete, type, sugar, loss, cholesterol, blood, diabetes, pound, level
 Top tags: exercise, weight, diet, diabetes, weight loss, food, cholesterol, muscle, pregnancy, blood sugar
 Examples:
  - what weighs more muscle or fat  im just wondering about weight gain due to muscle growth my wife has been working out for some time with weights and cardio training but she is finding that her weight has been fluctuating and at times gains weight a little bit
  - are diet pills safe for teenagers if so which ones are
  - had total knee replacements i am not feeling good no energy depressed no appetite have lost weight
  - what can i do to gain back my missing pounds and feel healthy again  i have been sick and lost 17 pounds i am fatigued all the time and look poorly i want to gain my wieght back and feel good again as quickly as possible
  - would you kindly suggest diet for a pregnant woman

Cluster 4  (n=13250)
 Top terms: pregnancy, doctor, blood, pain, test, surgery, medication, infection, drug, vitamin, sex, baby
 Top tags: pregnancy, drug, pain, penis, coldness, drinking, smoking, burn, vagina, injection
 Examples:
  - can an antibiotic through an iv give you a rash a couple days later
  - is all vitamin d the same
  - can taking multiple antibiotics cause redness and dryness of vagina
  - had a stroke on the brain in 2012 its 2016 i cant get no more than 5 hours of sleep a day
  - i smoked cigs for 1 month averaging about 3 a day just wondering if any irreversible was done i did quit since then i started smoking for a month after a period of depression a couple of months ago 2 3 cigs most days with a couple more on bad days i would estimate i probably had 4 packs total over the period i went cold turkey as i started to get my life together and hated the ill feeling from them i exercise regularly and eat healthy and i am still young i would just like to clear my head and hear that i did no damage permanent to my lungs i know it takes a bit to recover hopefully to 100

Cluster 5  (n=254)
 Top terms: lot, cause, man, xarelto, xdr, xolair, xray, yard, yasmin, yaz, yeast, wreck
 Top tags: exercise, coldness, cold, epilepsy, body rash, malaria, i have been feeling very depressed, i do i feel nauseous when i eat, hepatitis c, heartburn
 Examples:
  - is leucoderma curable
  - i am hardly eating anything and when i do i feel nauseous
  - what is arthropathy
  - how can ringworm be prevented
  - once i am stressed how can i calm myself

Cluster 6  (n=664)
 Top terms: migraine, headache, pain, head, sensitivity, eye, brain, vision, medication, aura, excedrin, weakness
 Top tags: migraine, headache, pain, head, vision, photosensitivity, drug, nausea, eyes, neck
 Examples:
  - can diabetes cause you to have chronic migraines
  - how many excedrins are safe to take in one day
  - i suffer with headachesmigraines frequently
  - unexplained headaches … does mri show problem
  - is lipitor used to treat migraines

Cluster 7  (n=174)
 Top terms: osteoarthritis, hip, arthritis, knee, costochondritis, tendonitis, area, pain, pac, problem, coxarthrosis, arthrosis
 Top tags: osteoarthritis, pain, arthritis, knee, could psoriatic arthritis be a more systemic problem, arthrosis of the hip joint (coxarthrosis), costochondritis, exercise, rheumatoid arthritis, thigh
 Examples:
  - is operation on en arthritisc bunnion recomended what are the results from it if having osteoarthitis thanks
  - i am 60 yr female fairly active the last yr having extreme pain in & around knees or feet frequently unable to walk intermittant pain started 1 12 yrs ago progressively more frequent ortho diagnosed minor arthritis not enough to cause pain & advised was inflammation of joints used topical anti inflammatories aleve & several modalities without consistant relief has drastically lowered quality of life weight gain & depression setting in as a result unable to obtain adequate help from local doctors or specialist accept that it will get worse any suggestions would be helpful
  - do i have costochondritis
  - what is the difference between ostio and rhumitoid arthitis
  - can osteoarthritis be prevented

Cluster 8  (n=1676)
 Top terms: insurance, health, plan, medicare, income, coverage, care, exchange, medicaid, marketplace, company, employer
 Top tags: insurance, medicare, health insurance, affordable care act, medicaid, health insurance exchange, health insurance marketplace, obamacare, pregnancy, dental
 Examples:
  - i am a disabled veteran on medicare am i affected by the affordable care act
  - i manage a medical office with 3 employees  rather than offer a health insurance plan we pay 50 of the employees premium so if they purchase their insurance through the marketplace will we no longer be able to do that
  - will they accept obamacare at any hospital
  - i am 50 years old and am currently on medicaressdi i have part a b and d  i also have aarp hospital indemnity by united healthcare i also have an aarp rx drug plan is this enough to cover what is needed under the affordable care act
  - if you already have a non  employer based health plan can you still go to the exchange  can you see if you can get a better plan at a lower rate

Cluster 9  (n=91)
 Top terms: tuberculosis, tb, xdr, drug, diagnosis, test, family, relapse, mother, disease, cure, risk
 Top tags: tuberculosis, drug, tuberculosis (tb) and fever, diagnosed with tuberculosis disease, bone ttberculosis (tb) diagnosis, pregnancy, baby, injection, skin, multi-drug-resistant tuberculosis (mdr-tb) and coughing blood
 Examples:
  - i have been diagnosed with tuberculosis disease but me and my family doubts this diagnosis
  - can you hold a baby after having a tb test
  - what can health care providers do to prevent extensively drug resistant xdr tuberculosis
  - is it harmful to get more than 1 tb test in a months time  received first test 2 weeks ago the 3 days ago received 2nd half and now they are saying they did them both wrong and i have to do them again is this safe
  - how often do you need a tb shot

Cluster 10  (n=184)
 Top terms: wart, plantar, doctor, skin, treatment, warfarin, tape, pant, duct, garlic, type, papillomavirus
 Top tags: wart, skin, plantar wart, vision, human papillomavirus, garlic, penis, foreskin, blister, mri results
 Examples:
  - can i transmit genital warts seventeen years after having them removed
  - what do warts look like
  - how will a doctor treat my warts
  - i frequently poop my pants accidentally please help me
  - what is hypertrophic cardiomyopathy hcm

Cluster 11  (n=2315)
 Top terms: skin, rash, acne, face, spot, eczema, bump, lip, cream, product, body, dermatitis
 Top tags: skin, rash, acne, arm, penis, atopic dermatitis, hand, itch, vision, thigh
 Examples:
  - guest in my home has scabies do i have house sterilized or will a good cleaning do it we have not had skin contact he is getting treated and i am having a general cleaner come in this afternoon to change linens etc is this sufficient i have never had anything like this in my home somewhat disturbed
  - what is the best moisturizer for older skin
  - is liposuction covered by insurance
  - 3 yr old son has small specks of blood on face after napping
  - what medicines can cause your skin to turn blue

Cluster 12  (n=60)
 Top terms: soma, odd, xarelto, xdr, xolair, xray, yard, yasmin, yaz, yeast, wreck, wrestling
 Top tags: pregnancy, ovulation, fertility, travel, vision, drinking
 Examples:
  - could i be pregnant please help
  - can you get pregnant if you take adderall
  - what if i become pregnant while taking metaglip
  - can you get pregnant any time
  - can i get pregnant if i am not ovulating

Cluster 13  (n=404)
 Top terms: hair, loss, scalp, woman, man, cause, shampoo, vitamin, minoxidil, dandruff, treatment, option
 Top tags: hair loss, hair, stress, head, pregnancy, scalp, hairfall, diet, acne, blood test
 Examples:
  - my hair has been thinning over the past few years taking minoxidil is it safe
  - does xanax show up in hair sample
  - does vinegar stop hair loss in women
  - can you have babies if a dog hair touches you
  - my hair has been falling out for almost a year now i am 26 yr old female my blood tests came back normal along wexcessive shedding my hair has been very dry i have not changed the way i treat my hair i do see growth but shedding is so excessive that my hair thinned considerably since last yr i have used spironolactone for acne 2 12 yrs ago i have been off it for a little over a yr did not experience any bad sideeffects wspiro i am not on birth control i also have been experiencing low sex drive my gp did not have an answer for me i feel hopeless<negative_smiley>

Cluster 14  (n=126)
 Top terms: parkinson, disease, unsteadiness, symptom, requip, problem, imbalance, dizziness, sign, drug, issue, inhibitor
 Top tags: "parkinsons disease", drug, "experiencing neurological problems... parkinsons disease", spine issues or parkinsons or both, brain, dizziness, foot, walking, muscle, pain
 Examples:
  - what is prolopa for parkinson is disease
  - i have or think i have parkinsons disease when should i contact my doctor
  - what increases the risk of getting parkinsons disease
  - my mother has parkinson is disease she has been having episodes of unresponsiveness are these related to parkinson is  these episodes vary in their duration she just stares and does not respond to stimuli verbal or physical it has been suggested these are freezing episodes but i am not familiar with brain freezing associated with parkinson is thank you for your assistance
  - how is parkinson is disease diagnosed

Cluster 15  (n=2425)
 Top terms: pill, sex, condom, control, birth, period, hiv, sperm, chance, ejaculation, boyfriend, pregnancy
 Top tags: pregnancy, condom, sexual intercourse, period, ejaculation, birth control, vagina, sperm, birth control pill, penis
 Examples:
  - how effective are foam and male condoms in preventing pregnancy
  - how effective are male condoms at birth control
  - i was curious about anal used mothers sex toy didnt clean it at risk for stds do not think it was used in a while i was curious and i found a vibrator and i used it i put a condom on it but condom broke i got tested for chlamidia and ghonorea both negative do you think i am at risk for hiv or anything else  also i used other sorta home made toys over a year ago and i just got worried i could have done damage to my body have not had any negative symptoms and havnt used them since last year should i be worried everything is normal and during use nothing negative happened like bleeding of anything
  - why will not my penis stay hard when in pregame  i get hard quite easily when around my gf but then all of a sudden when it comes to me taking my jeans off it goes down why  when it does decide to work i really do love sex with her so what is causing this also i can not cum when she tries to give bj
  - i am pregant does everything i wear have to have cotton in it  my boyfriend is sure he read that now that i am pregnant everything i wear has to have at least a percentage of cotton in it i know that my panties should be cotton but i can not find answers about the rest of my wardrobe this is making it difficult to find suitable pants for my work uniform

Cluster 16  (n=164)
 Top terms: epilepsy, seizure, diagnosis, event, sort, lobe, teenager, memory, people, surgery, term, daughter
 Top tags: epilepsy, seizure, drug, generalized epilepsy, short stature, memory loss, surgery, suffered some sort of seizure, i have temporal lobe epilepsy, benign
 Examples:
  - what is a seizure and what is epilepsy
  - i have seizure like events
  - how is epilepsy treated
  - i have seizure like events
  - i have a 9 year old daughter who suffered some sort of seizure yesterday

Cluster 17  (n=110)
 Top terms: vaginosis, sex, bacteria, partner, woman, symptom, vaginitis, treatment, man, tmp, infection, std
 Top tags: bacterial vaginosis, bacterium, sexual intercourse, vagina, sexually transmitted disease, vaginitis, screening, hysterectomy, gonoccocal urethritis, supplement
 Examples:
  - what causes bacterial vaginosis should i be worried if it this infection keeps popping up
  - can men get bacterial vaginosis
  - what happens to someone when they get bacterial vaginosis
  - what can i do to treat bacterial vaginosis at home
  - after i have been diagnosed with an std and treated for it should i go back and get tested again to make sure its gone  i got tested for std and came back possitive for chymidia and bacterial vaginosis i have been given treatments including antibodies for both and it has now been a week should i go back in for another screening to make sure it is gone before i have sex again i do not want to pass it on to another sex partner

Cluster 18  (n=203)
 Top terms: hernia, surgery, pain, button, groin, herniation, belly, bag, tear, treatment, operation, laparoscopic
 Top tags: hernia, surgery, pain, belly button, hiatal hernia, pregnancy, hernia repair, central posterior disc herniation, vision, scrotum
 Examples:
  - what are the treatments for a hernia
  - does hernia caouse infertility  hey there i have hernia there is pain in my scrotum does it cause to infertility
  - i have central posterior disc herniation with annular tear
  - how is a hernia repaired
  - how is a hernia diagnosed

Cluster 19  (n=2646)
 Top terms: fever, throat, cough, sinus, nose, infection, strep, pain, bronchitis, cold, asthma, symptom
 Top tags: fever, cough, coldness, throat, sinus infection, antibiotic, pain, nose, sore throat, headache
 Examples:
  - my baby ate her on poop my baby ate poop 4 days later she is sick weezing coughing and high fever for 4days straight i took her to the doctor and they said shes fine just a normal cold i told them what happen and they just said she should be fine but if she still has a fever next week come back what should i do and is her symptoms related to her eating her poop
  - i have persistent headache and i feel like i have lowgrade fever help  hi so i am a headache everyday it is not too bad though it is completely bearable but a little distracting and i have noticed that i have lowgrade fever most of the time or mostly everyday but just like the headache it is bearable i can not just shrug this feeling off this have been occurring for two or three months now i am hopefully going to the doctor in a few days and get myself checked
  - why do i feel lightheaded fatigued and sweat during sleeping no fever  i am an almost 37 yr old female with a lot of stress right now dr put me on effexor and i started not being able to sleep having bad headaches feeling lightheaded and constipated i took it for 2 weeks and he told me to stop when i called him he called me in something else but i am afraid to get it i have been on paxil prozac and celexa and never felt this horrible i have not taken anything in almost a week but feel lightheaded many times thoughout the day any ideas what could be wrong
  - can i have strep without fever  my 4 year old son was diagnosed with strep throat 3 days ago last night i was fine one minute and suddenly felt like i would been hit by a ton of bricks body aches headache  sore throat and general feeling crappy but no fever is there any point in dragging myself out to doctor when i feel so miserable is it possible to have strep without fever i do not have any runny nose stuffy nose or cough not a cold 
  - how long should i wait before bringing my 11 yr old with flu symptoms to our family dr it has been 8 days initial symptoms were nausea high fever severe headache loss of appetite and fatigue those lasted about 2 days now she is very tired little appetite sore bellynausea and has a sore throat and cough

Cluster 20  (n=167)
 Top terms: sex, intercourse, virgin, lot, teen, oct, color, man, butt, drive, occasionaly, victim
 Top tags: sexual intercourse, pregnancy, anus, ovulation, fertility, burn, bleeding after intercourse, swallow, itch, bleeding
 Examples:
  - can i have sex after having the lining burned out
  - i find myself eating from boredom a lot even if i am not hungry how can i make myself stop eating mindlessly
  - what would cause someone is brain to start hurting during intercourse
  - unprotected sex on ov day in oct then got my p one day early this month my p is 4 days late could i be pregnant
  - had sex on fertile days when should i check if pregnant

Cluster 21  (n=315)
 Top terms: shingle, shin, area, breakout, vaccine, scalp, risk, treatment, leg, shot, husband, thank
 Top tags: shingles, i had mild pitting edema, chickenpox, vaccination, pregnancy, scalp, pain, virus, baby, infant
 Examples:
  - can shingles occur in the scalp
  - i am 33 years old i need shingles vaccine but am i too young to have it
  - can i get shingles after having scarlet fever as a child  my mother is 80 and has been iin a lot of pain the last few weeks with her hip now she has a red rash very painful just appeared tonight it has a burning sensation is is possible she could have shingles should i take her to the er for treatment  thank you
  - at 21 i am still having breakouts what can i do to heal them
  - is shingles contagious

Cluster 22  (n=406)
 Top terms: hypothyroidism, thyroid, hyperthyroidism, synthroid, hypotension, weight, medicine, nodule, level, medication, hypothyroid, diet
 Top tags: hypothyroidism, thyroid, hyperthyroidism, weight, low blood pressure, exercise, pregnancy, hives, diet, weight loss
 Examples:
  - is garlic safe for my diabetic dog 8 year old bichon with atypical cushings and diabetes he takes 7 units of novalin insulin bid flax hull and melatonin 3mg bid sugars are consistantly high and is going blind
  - what causes peyronie is disease
  - i have all the classic signs of hypothyroidism but lab results normal where to go from here
  - im 27 and ive had almost all of the symptoms since i was about 6 years old could it be possible i have hypothyroidism
  - what would make my tsh levels suddenly jump to critically high level when i feel the same  i take synthroid 1 25 daily and yet my tsh is now 28 3 increased from 23 on friday why would it jump so quickly my dr is on vacation and i just want to make sure this is ok to wait until thursday when he comes back 

Cluster 23  (n=209)
 Top terms: overdose, poisoning, vitamin, case, heroin, potassium, metoprolol, capsule, alcohol, xanax, depakote, lantus
 Top tags: drug overdose, poisoning, vitamin d, magnesium, hydrocortisone, kidney failure, throat, tightness, temperature, acetaminophen
 Examples:
  - what should i do if i suspect an overdose of lidopain
  - what should i do if i suspect an overdose of thyroid
  - what should i do if i suspect an overdose of agesic
  - what should i do if i suspect an overdose of tinzaparin porcine
  - what should i do if i suspect an overdose of atorvastatin

Cluster 24  (n=837)
 Top terms: anxiety, depression, attack, panic, disorder, medication, symptom, med, pain, stress, xanax, heart
 Top tags: anxiety, depression, panic attack, pain, stress, heart, drug, chest, fear, antidepressant
 Examples:
  - my husband is taking 40 mg of prozac and is really depressed and has thought of suicide what do we do
  - pain when urinating inconsistent urination painfull ejaculation painfull mastrubation etc oh gosh im in all sort of trouble here and its given me anxiety over the past 1 5years ive been having this condition it all began when my urine penis started smelling cheesy after urination then later on when i was laying in bed and rising up i could feel from pelvic area like something is almost pushing my urine out it happened all the time then came premature ejaculation painfull urination painfull mastrubation painfull ejaculation also when i drag back my foreskin pain help please
  - i would appreciate advice on a drug regimen that will restore serotonin while dealing with short term anxiety symptoms
  - i am extremely agoraphobic despite help from my psychologist psychiatrist and medication why else can i do
  - should medical marijuana be used to treat anxiety disorder

Cluster 25  (n=199)
 Top terms: osteoporosis, scoliosis, bone, risk, ossification, treatment, density, medication, woman, health, perimenopause, care
 Top tags: osteoporosis, scoliosis, exercise, drug, heterotopic ossification, i have scoliosis, spine, back pain, pain, calcium
 Examples:
  - what can be done when heterotopic ossification breaks off
  - my osteoporosis specialist told me that i could take either reclast prolia or forteo which is safest  i am 65 and have never had a fracture i took fosamax for 7 years with no benefit other than things might have been worse without fosamax  i live in a second floor walk up with a dog and i still work full time a medication related fracture would be devastating but so would bone cancer advice please  my doctor provides only informational brochures from the drug manufacturers
  - scoliosis does it effect the stomach and breathing
  - without the scoliosis i would be about 56 tall should i use this height when calculating bmi
  - volunteer firefighting and scoliosis i am 16 and want to be a volunteer firefighter but have a slight case of scoliosis would i still be able to join 

Cluster 26  (n=647)
 Top terms: ear, infection, hearing, pain, doctor, tinnitus, wax, earache, fluid, sound, antibiotic, eardrum
 Top tags: ears, ear infection, tinnitus, pain, antibiotic, coldness, head, pressure, vision, neck
 Examples:
  - why am i hearing my heartbeat in my right ear  just recently i have started hearing my heartbeat in my right ear this came on suddenly i am a 66 year old female with no particular health issues what could be the cause of this anything to worry about
  - should i go to the er for severe right earache or wait until monday is appointment  i am 20 on thursday i woke with dull throbbing it has gotten worse i have a 101 f fever i think the throbbing is still there but now my ear canal is so blocked i can not feel anything in the canal but there is still sharp stinging pain in what feels like the back my outer ear is swollen the pain extended to my throat chewing or hiccups feels like my ear is tearing i have almost zero hearing in this ear er or wait
  - i have had white noise with corresponding hearing loss in my left ear for 2 months what causes it and can it be fixed i have had an mri    negative for tumor ms or anything else that might be causing it
  - does lipo flavonoid reduce or cure tinnitus
  - i am having ruptured eardrum one doctor suggested to have surgery and other suggested to wait for 3 months

Cluster 27  (n=156)
 Top terms: ringworm, worm, skin, scalp, beard, cream, ring, son, phimosis, tomato, school, earlobe
 Top tags: ringworm, skin, worm, scalp, itch, worms in stomavh, acne, blister, "im 39 weeks pregnant and have just discovered i have thread worm", ring worms
 Examples:
  - my ringworm fungal culture came back negative what else could it be if it is not fungus it is scaley but does not itch
  - how is ringworm of the scalp or beard treated
  - i ate rotten tomato – small white worms now inside me – how to get them out
  - i have or think i have ringworm of the scalp or beard when should i contact my doctor
  - how is phimosis treated

Cluster 28  (n=209)
 Top terms: chlamydia, sex, partner, treatment, gonorrhea, symptom, test, boyfriend, canbe, child, guy, pid
 Top tags: chlamydia, sexual intercourse, gonorrhea, penis, antibiotic, anus, drinking, mouth, vagina, pregnancy
 Examples:
  - is clindamycin effective in treating syphilis
  - can i use oil of oregano iv or im as an antibiotic  hello  a while ago i got chlamydia in three places i get horrible reactions to pharmaceutical antibiotics so decided to try oil of oregano locally and sublingually the chlamydia is cured below but has remained in my throat i was thinking of injecting pure essential oil of oregano im or iv in perhaps mct oil as a carrier i am experienced in the medical application of these injections so in that regard it is safe what do you think
  - can you drink alcohol while taking clidamycin  the clindamycin was given for bv and yeast infection
  - what increases the risk of getting chlamydia
  - is clindamycin effective in treating syphilis

Cluster 29  (n=221)
 Top terms: pneumonia, lung, doctor, son, shot, antibiotic, cough, pain, sepsis, symptom, treatment, fluid
 Top tags: pneumonia, walking, antibiotic, lung, bacterial pneumonia, "doctors dont know what is wrong", pneumonia wheezing, i had pneumonia, neck, sweating
 Examples:
  - i have had a pneumonia shot can i get either a sinus infection or walking pneumonia from my 6 year old grand daughter  and can i be a carrier to others in my age group
  - how soon should my 14 year old wait before returning to school having been diagnosed with pneumonia
  - can you have a stiff neck with pneumonia  i was treated for pneumonia with an antibiotic and prednisone for my chronic asthma i have recently noticed a stiff neck that seems to worsen by the end of the day can this be a side effect from the medicine or pneumonia or should i be concerned with a completely different illness
  - what are the symptoms of pneumonia
  - is viral pneumonia contagious

Cluster 30  (n=726)
 Top terms: eye, cataract, vision, circle, eyelid, stye, doctor, pain, pinkeye, discharge, bag, conjunctivitis
 Top tags: eyes, vision, cataract, pink eye, eyelid, swelling, burn, sty, skin, antibiotic
 Examples:
  - seems i have got chemical in eye from eye cream anything i can do to get relief from burning i have tryed to rinse
  - i have been suffering from pressure and pain behind eyes for almost three years
  - my 2 friends also underwent through lasik procedures but i am afraid about any side effects can anyone guide me about the possible risks thanks
  - swollen upper eyelid on the right side of my right eye no itching burning discharge redness its just swollen  i am not sure it is something that i should be worried about or even how to fix it but i just woke up and realized my eye felt of swollen so i looked at it and it is huge and it is only the top eyelid i know i have not hits my eye on anything and i have not had any allergies in a few months
  - should a cook with an eye stye work

Cluster 31  (n=698)
 Top terms: flu, vaccine, shot, vaccination, chickenpox, child, swine, influenza, baby, pox, chicken, symptom
 Top tags: flu, injection, vaccines, shingles, vaccination, swine flu, chickenpox, coldness, baby, pregnancy
 Examples:
  - can you test positive from having the hep b vaccine
  - why would a rn choose not to get her kids a flu shot as the grandparent is there anything i can do
  - my son had dtap polio chicken pox and mmr vaccines now can barely move
  - what reactions are likely after an immunization
  - what happens to someone when they get influenza

Cluster 32  (n=211)
 Top terms: depo, shot, period, provera, sex, birth, control, pill, pregnancy, symptom, effect, test
 Top tags: injection, pregnancy, period, depo-provera, sexual intercourse, birth control, spotting, pregnancy test, cramps, fertility
 Examples:
  - i have been off depo shot for over a year and have no period i recently began having pregnancy symptoms am i pregnant  i was on the depo shot for 2 years and have not gotten the shot since september 2011 i have still not had a period and have been having pregnancy symptoms my fiance and i use the withdraw method i realize that you can still get pregnant this way after over a year of not receiving the shot i do not think the symptoms could be attributed to the shot i had a pap smear 9 12 12 and it came back fine so could i be pregnant and if not what could be causing the pregnancy symptoms
  - depo provera didnt work well for mecontinuos bleeding will implanon be the same way tired of using the pill i would prefer to have as few menstrual cycles as possible none would be great when on depo provera the first few months were wonderful but it spiraled downward from there continuous never ending bleeding i would like to not have to take a pill everyday my schedule is hectic being active duty military and a student any advice on whether implanon would be a good choice is welcome
  - pregnant after depo  i recieved my first depo shot in may and just recently received my third one in november my husband and i decided we would like to have a child should i wait till february to start fertility drugs to help us or would i be able to start now i was told that fertibella is a great start but i just do not know when exactly thank you
  - i was due for the depo shot december 21st did not get it and had unprotected sex the next day am i pregnant
  - 17yr missed period in april spotted in march nothing in june pregnacy test negative on depo shot last year hello  my daughter is 17 she has been sexually active and claims a condoms was used in the act she did not receive a period in april and than spotted in march no period yet for june she did take a test and it came back negative she was on the depo briefly  she only took two shots and than stopped her last shot was last year i want to say in july 2012 please note that she had been getting regular periods up until april when she did not receive one

Cluster 33  (n=151)
 Top terms: anemia, hemoglobin, iron, thalassemia, blood, reduction, age, anaemia, infusion, injection, beta, anitratus
 Top tags: anemia, iron, normocytic anemia, i am a veteran who has beta thalassemia minor and asthma, iron deficiency anemia, low hemoglobin, not sure my doctor is doing it right, pernicious anaemia and require regular b12 injections, acinetobacter anitratus, advice regarding thalassemia and pregnancy
 Examples:
  - female age 5 diagnosed with normocytic anemia what can cause this
  - i am a veteran who has beta thalassemia minor and asthma while in deployment in afghanistan
  - does ferrous gluconate in my multi vitamin help iron deficiency anemia  i also take ferrous sulfate for my anemia i have been on them for years and wanna figure out if i can stop taking one or the other safely
  - i have pernicious anaemia and require regular b12 injections not sure my doctor is doing it right
  - how is anemia related to lead poisoning treated

Cluster 34  (n=56)
 Top terms: trimester, pregnancy, problem, movement, twin, symptom, placenta, percocet, woman, excedrin, baby, pinkish
 Top tags: pregnancy, first trimester, third trimester, second trimester, movement, baby, hunger, suicide, miscarriage, birth
 Examples:
  - are there any over the counter cough meds that are safe in the 3rd trimester to use
  - do pregnancy symptoms come and go or fade away during your first trimester
  - what causes bleeding in the first trimester of pregnancy
  - i am 14 5 weeks pregnant and i am not as hungry as i was in my 1st trimester very is this worrisome  during my first three months i was constantly hungry and eating constantly at night as well i probably gained 10 pounds by the end of my first trimester now i am much less hungry maybe eating less than i was before i got pregnant should i be worried about the baby is development
  - why is the second trimester of pregnancy the easiest for some women

Cluster 35  (n=93)
 Top terms: lupus, infection, erythematosus, sle, disease, symptom, flare, effect, control, sun, medication, cell
 Top tags: lupus, do i have lupus, drug, walking, i have rare sarcoid disease, probable lupus, swelling, diabetes, pain, seizure
 Examples:
  - do i have lupus or some chronic viral infection
  - who gets lupus
  - what increases the risk of getting lupus systemic lupus erythematosus
  - how do i know if i have lupus
  - what medications are used for lupus

Cluster 36  (n=2264)
 Top terms: tooth, baby, disorder, disease, sleep, child, problem, reflux, symptom, heartburn, depression, dementia
 Top tags: baby, tooth, heartburn, bipolar disorder, pregnancy, bedbug, schizophrenia, dementia, depression, postpartum depression
 Examples:
  - what are the dietary restrictions for celiac disease gluten
  - where can i go for help for bipolar disorder
  - can herpes be spread by bed bugs if a person infected with herpes is bitten by a bed bug can another person bitten by the same bug get infected with herpes
  - would braces close wide gap between front teeth
  - my mom is in a depression…what can i do

Cluster 37  (n=1134)
 Top terms: cancer, prostate, breast, mole, colon, mammogram, lump, radiation, biopsy, treatment, symptom, surgery
 Top tags: prostate cancer, cancer, colon cancer, breast, mammogram, breast cancer, prostate, mole, surgery, lump
 Examples:
  - how to treat leiomyosarcoma and rectal cancer at the same time
  - i have a basal cell carcinoma what specialist should i consult
  - i have recently had breast surgery removal of calcium deposits with cancer cells
  - i have had rapid breast growth want breast reduction surgery what age should i typically get this surgery
  - can hot showers a day cause cancer  i love to take hot showers and take three 5 10 minute showers a day   morning noon and evening is there a danger of cell damage or cancer from this practice also are three too many i like the water very hot is this bad our water in our city is chlorinated should i be concerned

Cluster 38  (n=922)
 Top terms: kidney, uti, cyst, urine, bladder, stone, infection, pain, urination, antibiotic, blood, tract
 Top tags: pain, cyst, kidney, kidney stone, urination, antibiotic, pregnancy, bladder, vision, ovary
 Examples:
  - how to manage large kidney stones that can not be passed on its own while being 16 weeks pregnant and not harm the baby stones are roughly 5 8 mm in size recurrent episodes of pain over the past 2 3 weeks should i stent how safe are the pain meds percocet and dilaudid for the baby
  - just discovered a 4cm x 4cm cyst inside my left ovary via ultrasound will ovary need removal
  - why does my urine smell
  - is it safe to swim in a lake if you have a uti  i have a bladder infection and began antibiotics two days ago we are going to the lake this week can i swim in the lake or should i avoid that
  - my urine has a bad smell and is cloudy what can be wrong

Cluster 39  (n=2255)
 Top terms: period, pregnancy, test, sex, cycle, symptom, birth, sign, cramp, breast, control, pill
 Top tags: pregnancy, period, pregnancy test, sexual intercourse, spotting, nausea, breast, birth control, ovulation, cramps
 Examples:
  - my period only last 36 48 hours which is my norm is that why i have had 2 yrs of no luck getting pregnant  my husband and i have been trying for two years to have a child i am turning 30 next month and in my family after 30 equals issues my normal period is only 36 48 hours could this be preventing me from getting pregnant
  - spotting on day two of my period could i be pregnant i have had my loop taken out 71015 and have had unprotected sex a few times after that hoping to fall pregnant my period was meant to start 12102015 it now day two on my period and i have only been spotting which is very un usual could i be pregnant
  - could i possibly be pregnant  last period may 16th unprotected sex on june 9th supposed to start june 16th still have not if you think i am pregnant when should i take a test  side notes  feel as if i start but do not   i was throwing up at 2 am on saturday the 15th i was nauseous the rest of the day   light cramps  the guy i had sex with says he only has a 3 chance of getting someone pregnant  i have been tired lately i also have been having light heartburn i think  if anyone can help me it would be greatly appreciated
  - pregnant unprotected sex a week before period period came on time and heavy with bad cramps as usual reg 28 day 4 yrs i had unprotected sex a week before my period started he ejaculated awa from me but im worried a little bit may have got it before he pulled out my period came on the dot when it was supposed to get it and was heavy at first then to moderate with bad cramps like i normally have basically my period came on time and was normal in length flow and cramps my periods have been regular for years i do not know when i ovulate or my latueal phase what are my chances of being pregnant
  - my period has been late by 4 days i am trying to conceive please help

Cluster 40  (n=168)
 Top terms: gallstone, gallbladder, gall, pain, function, bladder, treatment, opinion, surgery, leaking, symptom, april
 Top tags: gallstone, pain, gallbladder, diagnosed with abnormal liver function and gallbladder disease, gallbladder surgery, gallbladder removed i have a small tear, i should remove my gallbladder, diabetes, bladder, surgery
 Examples:
  - how are gallstones diagnosed
  - how do l deal with pain from gall bladder infection or gallstones
  - how do l deal with pain from gall bladder infection or gallstones
  - diagnosed with abnormal liver function and gallbladder disease what should i do next
  - what kind of diet would you say is best for a 62 year old man with gallbladder cancer

Cluster 41  (n=456)
 Top terms: herpe, outbreak, virus, partner, sex, simplex, sore, gum, area, hsv, treatment, type
 Top tags: genital herpes, herpes, virus, sexual intercourse, cold sore, herpes simplex, vagina, mouth, sore, oral herpes
 Examples:
  - how is herpes simplex treated
  - herpes transmission somewhat complicated question i have genital herpes but no outbreaks after the initial one my partner got it from me and has genital outbreaks we are wondering if we are able to spread it to each other in areas where we have not had outbreaks during shedding not active outbreak seems obvious that we could spread it around while there are lesions so we are careful during her outbreaks but the big mystery is  can we spread it to new areas during shedding    thanks a million we have been searching hard for the answer 
  - are there any ointments that can speed up the healing process of genitle herpes already on 3 day valtrex should i be worried if it dosent clear up by then
  - can i get a vasectomy if i have genital herpes
  - i shave my pubic region and noticed a single white bump in my upper pubic region could this be herpes  it is a single bump whereas i have heard herpes is usually a cluster it is also a larger bump more the size of a pimple it was whiteyellow in color and a whiteclear liquid oozed out of it my region is slightly itchy but i wonder if this is herpes or an ingrown hair i have no blisters in my actual vaginal region only one at the top of my pubic region i did feel slightly under the weather today as well the area was painful and itchy but only in that area

Cluster 42  (n=1820)
 Top terms: heart, pressure, chest, blood, pain, rib, side, failure, rate, bp, disease, hypertension
 Top tags: high blood pressure, blood pressure, chest, heart, pain, congestive heart failure, heart disease, atherosclerosis, heart attack, low blood pressure
 Examples:
  - ekg says there was “moderate right axis deviation ” “normal sinus rhythm with marked sinus arrythmia ”
  - is there evidence that statins increase life expectancy for people without heart disease
  - broken collarbone 3 5 cm overlap its been three weeks after break and still feels broken or loose
  - i felt like electric shock like feeling which became sharp on my left chest
  - would lad lesion cause tachycardia

Wrote assignments to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/clustered_short_questions.csv

=== Assessment Metrics (COSINE) ===
 Algo: Spectral
 Features: BERT(all-MiniLM-L6-v2) 384D + Tags→SVD(20D, EV=0.245) = 404D
 Global: silhouette (cosine)=0.0023

--- Cluster Size Stats ---
 k=43 | min=56 max=13250 mean=1104.4 median=315.0 std=2085.9

--- Per-Cluster Averages ---
 cluster | size | mean silhouette (cosine)
       0 | 2297 |         0.0227
       1 | 3610 |        -0.0681
       2 |  256 |         0.4166
       3 | 2262 |         0.0224
       4 | 13250 |        -0.1801
       5 |  254 |         0.5714
       6 |  664 |         0.2746
       7 |  174 |         0.4445
       8 | 1676 |         0.2267
       9 |   91 |         0.4001
      10 |  184 |         0.3572
      11 | 2315 |         0.0209
      12 |   60 |        -0.1214
      13 |  404 |         0.2869
      14 |  126 |         0.5101
      15 | 2425 |        -0.0464
      16 |  164 |         0.3982
      17 |  110 |         0.4564
      18 |  203 |         0.4018
      19 | 2646 |        -0.0483
      20 |  167 |         0.4353
      21 |  315 |         0.3676
      22 |  406 |         0.2108
      23 |  209 |         0.3495
      24 |  837 |         0.1879
      25 |  199 |         0.2372
      26 |  647 |         0.2351
      27 |  156 |         0.3169
      28 |  209 |         0.3565
      29 |  221 |         0.3728
      30 |  726 |         0.1580
      31 |  698 |         0.1793
      32 |  211 |         0.3344
      33 |  151 |         0.3375
      34 |   56 |         0.5157
      35 |   93 |         0.4833
      36 | 2264 |        -0.1228
      37 | 1134 |         0.0509
      38 |  922 |         0.0250
      39 | 2255 |         0.0535
      40 |  168 |         0.4165
      41 |  456 |         0.1497
      42 | 1820 |         0.0322

--- Structure (cosine on scaled space) ---
 Avg intra-cluster cosine similarity: 0.3914 (higher = tighter)
 Mean inter-centroid cosine similarity: -0.0105 (lower = better)

--- Tag Coherence (avg) ---
 Purity=0.503 | Entropy=3.468 | Intra-Jaccard=0.191

Saved model-selection style metrics to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_model_selection.csv

--- Model Selection Rows (head) ---
        scope            algo  min_cluster_size  min_samples  k  n_noise  silhouette  max_cluster_diameter
sample_k_scan MiniBatchKMeans                 0            0 10        0    0.061311              1.332653
sample_k_scan MiniBatchKMeans                 0            0 11        0    0.066409              1.332653
sample_k_scan MiniBatchKMeans                 0            0 12        0    0.068432              1.332653
sample_k_scan MiniBatchKMeans                 0            0 13        0    0.071887              1.291242
sample_k_scan MiniBatchKMeans                 0            0 14        0    0.072313              1.291242
sample_k_scan MiniBatchKMeans                 0            0 15        0    0.072753              1.286252
sample_k_scan MiniBatchKMeans                 0            0 16        0    0.076705              1.263103
sample_k_scan MiniBatchKMeans                 0            0 17        0    0.077494              1.276472
sample_k_scan MiniBatchKMeans                 0            0 18        0    0.076598              1.282558
sample_k_scan MiniBatchKMeans                 0            0 19        0    0.075843              1.283190

Saved k vs max_cluster_diameter plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v1.4.2_spectral_k_vs_max_cluster_diameter.png

Saved k vs silhouette plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v1.4.2_spectral_k_vs_silhouette.png

[UMAP] Saved 2D scatter to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_scatter.png
[UMAP] Saved embedding CSV to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_embedding.csv

[HClust] Saved dendrogram to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/hierarchical_dendrogram.png

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_timings.json

=== Runtime (seconds) ===
               load_csv_s:   0.2616
         nlp_preprocess_s: 114.5308
    vectorize_text_tags_s:  98.9635
         scale_features_s:   0.1711
             clustering_s:  75.0509
     interpret_clusters_s:   0.4345
           save_outputs_s:   0.2512
        metrics_compute_s:  37.7621
                   umap_s:  32.6160
           hierarchical_s:   0.2567
          total_runtime_s: 396.1979
```
In v1.4.3, I'm essentially running a semi-supervised topic discovery pipeline over about 47,491 short medical questions with same techniques of data preprocessing and Natural Language Processing as I did in v1.1.3.

For the clustering itself, I first perform model selection on a subset of up to ten thousand points using MiniBatchKMeans, cosine distances, and two diagnostics: cosine silhouette and cosine max cluster diameter. As I increase k, the silhouette gradually rises from roughly 0.06 into the mid-0.08 range, while max diameter steadily decreases before flattening out. There is no sharp elbow; the curves show the classical behavior of a high-dimensional, messy dataset where more clusters gradually carve the space into tighter regions. The configuration around k in the low forties looks best, and I ultimately pick k = 43 as a balance between granularity and stability, knowing that I'm favoring a more fine-grained taxonomy instead of a small set of coarse buckets.

Because the full dataset has nearly fifty thousand points, I run spectral clustering with a nearest-neighbors affinity on a subset and then propagate labels to all points using a k-nearest-neighbor classifier with k = 5 under cosine similarity. This produces forty-three clusters over the entire corpus. When I inspect the cluster summaries, many groups line up very neatly with specific diseases or topics. There are clear clusters for hepatitis and liver disease, tuberculosis, epilepsy and seizures, Parkinson's disease, anemia and thalassemia, lupus, osteoporosis and scoliosis, hernia, gallbladder and gallstones, herpes, shingles, pneumonia, eye and vision problems, ears and tinnitus, hair loss, skin and acne, vaccines and flu shots, and an insurance and Medicare cluster. Other clusters represent symptom-heavy or lifestyle scenarios such as gastrointestinal complaints, musculoskeletal pain, weight and exercise and diet, fever and respiratory infections, anxiety and depression, or mixed pediatric and sleep-related issues.

At the same time, the model produces at least one huge, noisy cluster. Cluster 4, with over thirteen thousand points, has a clearly negative silhouette and extremely broad top tags that mix pregnancy, drugs, pain, sexual terms, colds, smoking, drinking, injections, and more. This is effectively a catch-all for mixed or ambiguous questions. A few other large clusters, such as those dominated by general disorders or family and life context, also show weaker separation and negative silhouettes. This bimodal quality—some clusters very clean and disease-specific, others acting as topic soup—is the main reason the global metrics look underwhelming.

The global cosine silhouette across all points is approximately 0.0023, essentially zero. On its face that looks bad, but it is heavily influenced by a handful of giant, low-quality clusters. When I look at per-cluster silhouettes, the picture changes. Many small or medium clusters have silhouettes in the 0.3 to 0.57 range, which is strong for noisy medical text; some disease-specific clusters like Parkinson's, lupus, and a few others go above 0.5. Others sit in the 0.35 to 0.45 band, which is still quite respectable. The large, mixed clusters such as C4 and C36 drag the global average down with silhouettes around -0.18 or -0.12. So, from a metric perspective, the model is very good on disease-focused islands and poor on the multi-topic continents.

Additional structure metrics support this interpretation. The average intra-cluster cosine similarity is around 0.39, suggesting that, on average, points within a cluster are moderately tight. The mean inter-centroid cosine similarity sits just below zero, indicating that centroids are almost orthogonal on average and distributed rather evenly on the hypersphere. UMAP visualizations show precisely this pattern: several distinct, colorful clumps corresponding to monolithic topics, and a dense central blob representing those broad, mixed clusters. The dendrogram built on cluster centroids further reveals meaningful higher-level groupings, such as STDs and reproductive clusters hanging together, respiratory diseases clustering on another branch, autoimmune and rheumatologic conditions aligning, and cancer-related centroids forming their own subtree. This gives me a taxonomy over the clusters themselves and hints at how to roll forty-plus topics up into ten or fifteen super-topics if needed.

Tag coherence metrics add one more layer of insight. Mean tag purity is about 0.5, so the most frequent tag in a typical cluster accounts for roughly half of all tag assignments in that cluster. Mean tag entropy sits in the mid-threes, which means clusters are not single-label but have a moderate variety of tags, consistent with multi-label medical questions. Mean intra-cluster tag Jaccard around 0.19 shows that tag sets within a cluster overlap more than by chance but are far from identical. Disease-specific clusters tend to have very high purity and clear dominant tags, whereas symptom clusters have higher entropy and more label diversity. This is exactly what I'd expect: diagnoses are sharp, symptoms and life situations are diffuse.

On the computational side, the pipeline runs in a few minutes for the full dataset. The main cost centers are spaCy POS tagging, SentenceTransformer BERT encoding, spectral clustering with k-nearest-neighbor propagation, UMAP embedding, and the more expensive metrics like silhouettes and diameters. The total runtime is roughly six to seven minutes, which is perfectly acceptable for offline analysis at this scale. If I needed to iterate faster, I could cache POS and BERT features, subsample for spectral clustering and metrics, or skip UMAP during early experiments.

From a practical standpoint, v1.4.3 gives me a mid-granularity medical taxonomy with about forty topical clusters that are mostly interpretable and medically coherent. Dozens of clusters are strong enough to serve directly as routing topics, navigation labels, or seeds for weak supervision. A few very large and noisy clusters highlight where further work is needed—either second-stage clustering inside those buckets or light supervision to pull borderline points into cleaner topics. For retrieval and reranking, the cluster assignments and centroid hierarchy provide a natural way to constrain search or build topic-aware features. As a snapshot, v1.4.3 shows that the joint BERT-plus-tag representation is working well and that the major challenge is not discovering disease-specific structure but handling the ambiguous, cross-cutting questions that naturally form broad “soup” clusters.

## v2.1.3

```
Rows after POS cleanup: 47491
[Features] BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245) = 100D
[Shapes] Combined feature matrix: (47491, 100)
Searching for a good k (KMeans, COSINE diagnostics)...
Model selection summary (first rows):
 k  silhouette_cosine  max_cluster_diameter_cosine
10           0.043268                     1.450144
11           0.048399                     1.497016
12           0.051724                     1.479859
13           0.052638                     1.457310
14           0.057137                     1.521825
15           0.062397                     1.479859
16           0.066035                     1.501172
17           0.071343                     1.455607
18           0.073385                     1.501172
19           0.076826                     1.452153
Selected k = 45 with cosine-silhouette = 0.123
Saved model-selection table to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_model_selection.csv

[Plot] Saved cosine-silhouette vs k plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v2.1.2_silhouette_vs_k_kmeans_cosine.png

[Plot] Saved max cluster diameter (cosine) vs k plot to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v2.1.2_max_cluster_diameter_vs_k_kmeans_cosine.png

=== Cluster Summaries ===

Cluster 0  (n=1208)
 Top terms: leg, arthritis, knee, osteoarthritis, pain, hip, thigh, bone, opinion, doctor, groin, arm
 Top tags: arthritis, osteoarthritis, leg, knee, psoriatic arthritis, swelling, thigh, cramps, walking, rheumatoid arthritis
 Examples (original):
  - okay so i am 16 and i want to grow about 3 more inches if i smoke hookah once or twice will i grow to my goal height
  - had total knee replacements i am not feeling good no energy depressed no appetite have lost weight
  - i have lump in the soft tissue of my left leg about 1 inch above my ankle i have had an ultrasound and a xray and i have been told it is not life threatening but the doctor will not tell une but it is hurting and swelling and i am taking pain killers every day me what it is until he sees me again in june
  - i have had 2 knee operations knee feels like loose
  - my mom has one hip is a little higher up than the other and it hurts

Cluster 1  (n=489)
 Top terms: medicare, insurance, medicaid, health, care, income, plan, act, obamacare, coverage, tax, state
 Top tags: medicare, insurance, medicaid, health insurance, obamacare, aca, asthma, vision, supplement, affordable care act
 Examples (original):
  - i am a disabled veteran on medicare am i affected by the affordable care act
  - i am 50 years old and am currently on medicaressdi i have part a b and d  i also have aarp hospital indemnity by united healthcare i also have an aarp rx drug plan is this enough to cover what is needed under the affordable care act
  - i am eligible for medicaid can i drop the insurance offered by my employer and just have medicaid  i currently have insurance for my family through my employer under the new income guidelines in my state we will be eligible for medicaid under the health care reform act will i have to pay a fine for not keeping my employers insurance i am not sure how all this works thanks mark
  - what if you are a veteran and have veteran health care do you need to get health insurance through the marketplaces
  - i need affordable dental care how can i get decent care at an affordable price or on a sliding scale

Cluster 2  (n=780)
 Top terms: heart, failure, heartburn, disease, rate, cholesterol, tachycardia, risk, palpitation, pain, chest, attack
 Top tags: heart disease, heart, congestive heart failure, heartburn, cholesterol, heart attack, high blood pressure, artery, tachycardia, exercise stress test
 Examples (original):
  - ekg says there was “moderate right axis deviation ” “normal sinus rhythm with marked sinus arrythmia ”
  - is there evidence that statins increase life expectancy for people without heart disease
  - would lad lesion cause tachycardia
  - my mother is in extreme pain for a detached rotator cuff but suffers from heart problems and taking medication
  - what factors place women at high risk for heart disease

Cluster 3  (n=833)
 Top terms: eye, vision, cataract, pain, eyelid, circle, nose, doctor, headache, pinkeye, stye, problem
 Top tags: eyes, vision, cataract, pink eye, swelling, burn, eyelid, antibiotic, sty, high blood pressure
 Examples (original):
  - seems i have got chemical in eye from eye cream anything i can do to get relief from burning i have tryed to rinse
  - i have been suffering from pressure and pain behind eyes for almost three years
  - my 2 friends also underwent through lasik procedures but i am afraid about any side effects can anyone guide me about the possible risks thanks
  - i was checking my husband is testicles and i felt a peanut size lump above his left testicle not sure if we should worry  we are both without insurance right now it would be nice to know if i should just keep an eye on it or if it will just go away
  - my eyelids sag and i have heard that botox can lift my eyelids how does this work will not it look strange

Cluster 4  (n=701)
 Top terms: headache, nausea, period, pain, symptom, migraine, head, diarrhea, cramp, doctor, pregnancy, blood
 Top tags: nausea, headache, pregnancy, period, vomit, migraine, dizziness, head, pain, diarrhea
 Examples (original):
  - how long should i wait before bringing my 11 yr old with flu symptoms to our family dr it has been 8 days initial symptoms were nausea high fever severe headache loss of appetite and fatigue those lasted about 2 days now she is very tired little appetite sore bellynausea and has a sore throat and cough
  - my husband has had a fever and a cough for over 10days now not pneumonia had xray 4 days ago what could it be  his fever ranges between 100 103 he is coughing up lots of mucous but does not feel any sinus congestion no runny nose he has been on an antibiotic for 5days now he says it is helping a little but it is not taking care of the fever or cough just making the mucous less green he had a chest x ray 5days ago but it did not show pnumonia should he be x rayed again he is also very pale in color as well as is experiencing dizziness headache and nausea had a cbc blood work came back normal
  - can u get pregnant on the pill  i have been taking the pill for 2 months now and my last peroid wasnt really a period i bled for maybe 5 minutes a day for 3 days could i be pregnant i just started feeling nausea and im hungry but when i start eating i feel like i cant eat
  - could clonazepam be causing nausea and dizziness or is it bupropion
  - stabbing pain under rib cage stabbing pains right side under my ribs for several weeks started in july ct scan normal i eat healthy now it is accompanied with awful nausea and vomiting the nausea is constant i have vomited 20 times in 3 weeks not pregnant i have been to so many doctors and they are really not sure what is wrong all of the tests have came back normal ct scan with contrast blood work xrays ultrasound of gallbladder and hida scan for the hida scan gallbladder and small bowels normal and ef 68

Cluster 5  (n=1639)
 Top terms: pregnancy, sex, pill, woman, chance, birth, trimester, control, test, symptom, condom, sign
 Top tags: pregnancy, ejaculation, birth control, condom, pregnancy test, miscarriage, spotting, birth control pill, sperm, vision
 Examples (original):
  - how effective are foam and male condoms in preventing pregnancy
  - what are some warning signs for pregnant women when they are exercising
  - i am pregant does everything i wear have to have cotton in it  my boyfriend is sure he read that now that i am pregnant everything i wear has to have at least a percentage of cotton in it i know that my panties should be cotton but i can not find answers about the rest of my wardrobe this is making it difficult to find suitable pants for my work uniform
  - sexual health how can i increase my semen due to much masturbation i m starting masturbation in the age of 14 yr regularly many time in a day now i am 28 yr i have following major problems 1 erectile dysfunction 2 premature ejaculation 3 low sperm count after masturbation semen has comes only 4 5 drops and very slander can i pregnant my wife in this situation please sir help me
  - how to confirm my pregnancy

Cluster 6  (n=634)
 Top terms: ear, infection, hearing, pain, doctor, fluid, head, eardrum, wax, sound, pressure, ache
 Top tags: ears, ear infection, pain, tinnitus, head, vision, antibiotic, pressure, neck, swelling
 Examples (original):
  - why am i hearing my heartbeat in my right ear  just recently i have started hearing my heartbeat in my right ear this came on suddenly i am a 66 year old female with no particular health issues what could be the cause of this anything to worry about
  - i have had white noise with corresponding hearing loss in my left ear for 2 months what causes it and can it be fixed i have had an mri    negative for tumor ms or anything else that might be causing it
  - does lipo flavonoid reduce or cure tinnitus
  - i am having ruptured eardrum one doctor suggested to have surgery and other suggested to wait for 3 months
  - my brother has been losing his hearing abilities since he was 6 7 years old

Cluster 7  (n=938)
 Top terms: foot, hand, finger, toe, ankle, athlete, pain, nail, leg, thumb, heel, surgery
 Top tags: foot, finger, hand, swelling, toe, ankle, "athletes foot", leg, walking, burn
 Examples (original):
  - can i sit in a sauna and steam room with a broken ankle
  - i have got a wrist and palm injury in my right hand
  - is hand foot and mouth the same as rubella is hand foot nd mouth the same as rubella
  - i am beginning to notice that i am losing strength in my hands things fall out of them what is wrong with me  if i am holding on to something and apply medium low pressure it slips out my hands or i just can not maintain grip when i become upset or angry my hands swell up noticeably and the condition worsens i am 31 years old 32 in january hispanic 61 255 lbs heart disease and diabetes run in my family please tell me what could be wrong with me  
  - what kind of doctor treats athlete is foot

Cluster 8  (n=1900)
 Top terms: allergy, cough, throat, hive, vaccine, asthma, mouth, reflux, inhaler, lip, tongue, benadryl
 Top tags: smoking, cough, allergy, hives, heartburn, inhaler, throat, asthma, mouth, food allergy
 Examples (original):
  - my son had dtap polio chicken pox and mmr vaccines now can barely move
  - what reactions are likely after an immunization
  - what causes hives
  - for hand and mouth disease can the sores be on tongue
  - gave boyfriend oral then ate an hour later got home and felt mucus in throat and spit it went down my body pregnant  gave my boyfriend oral and swallowed some then ate a small burger and some coke an hour later i took a shower while in the shower i felt then need to spit i felt like mucus so i did spit worried that spit contained sperm and traveled down my body and got me pregnant i did not insert anything in not even with fingers and i was obviously standing while in shower but spit went through middle of my stomach so i am sure it passed by my vagina i am only 19 help can i be pregnant

Cluster 9  (n=685)
 Top terms: stomach, pain, period, diarrhea, test, cramp, blood, stool, doctor, side, nausea, symptom
 Top tags: stomach, pain, pregnancy, nausea, diarrhea, period, vomit, burn, cramps, diet
 Examples (original):
  - how long can flucold causing bacteria live outside the human body  pretty much what i asked my mom had a nasty stomach bug which i am pretty sure was the flu with all the nasty symptoms that go along with it not thinking she used my computer for something to do i have been avoiding it using a friend is computer now but how long do i need to i have stuff i need to do on it so how long do i need to wait before i do not need to worry about contracting it myself
  - pain in upper right quadrant of abdomen may indicate what 18 year old grand daughter had gallbladder stmptoms but no stones doctor said bile was crystalizing and bladder was removed still having problems food test showed food slow at passing from stomach to intestines smaller portions have not helped couple crackers may induce vomiting what are possible causes
  - does nystatin 100 000 also kill stomach yeast  checking to see if it is all over benefit or only oral
  - opting out of chemotherapy after having total stomach removal due to stomach cancer  my mother had the earliest stage of stomach cancer and had her total stomach removed and no signs of cancer anywhere else she is 75 years old and opted out of further treatment will she do ok with keeping up with the 3 month ct scans and lab work she still has the feeding tube attached in case in 3 months chemotherapy is necessary
  - i am lost on ideas what is wrong with me pain in stomach gets worst when i eat i have no gallbladder or apendix it feels like hunger pains dull to sharp at times someone please help no nausea

Cluster 10  (n=330)
 Top terms: shingle, pox, vaccine, chicken, scalp, area, breakout, husband, treatment, chickenpox, child, person
 Top tags: shingles, chickenpox, vaccination, blister, pain, virus, scalp, rash, infant, pregnancy
 Examples (original):
  - can shingles occur in the scalp
  - i am 33 years old i need shingles vaccine but am i too young to have it
  - can i get shingles after having scarlet fever as a child  my mother is 80 and has been iin a lot of pain the last few weeks with her hip now she has a red rash very painful just appeared tonight it has a burning sensation is is possible she could have shingles should i take her to the er for treatment  thank you
  - is shingles contagious
  - do shingles make you pee blue

Cluster 11  (n=956)
 Top terms: test, blood, result, drug, pregnancy, testicle, urine, exam, doctor, hiv, disease, testing
 Top tags: drug test, blood test, pregnancy, pregnancy test, testicle, vision, smoking, hair, marijuana, ultrasound
 Examples (original):
  - pain in testicles lower abdomen rectum is it fatty liver
  - my left side testicle is hurting and pissing blood
  - drug test where to find
  - if i remove my testicles will my testosterone production cease
  - my hiv test came back negative a positive test person was in the room before me could i have been exposed to hiv  i touched the countertop doorknob of the room etc

Cluster 12  (n=4496)
 Top terms: overdose, child, problem, sleep, daughter, son, sex, husband, baby, way, weight, ringworm
 Top tags: drug overdose, virus, burn, bedbug, ringworm, ibuprofen, chickenpox, fertility, cats, infertility
 Examples (original):
  - what is black measles when i was young i had them now in my fifties i have a lot of health problems could it be because of them and what damage do they do to your body i know they have not been heard of in god know how long is there any way to know after all of these years after having them to get information on them
  - i am experiencing a problem keeping an erection are there natural remedies that can be taken for this
  - is it better for a type ii diabetic to eat corn or bread stuffing
  - can you use egg whites on a burn  i read an article that said you can use egg whites to sooth and help heal burns like if you burn yourself with fire but not real bad is this true
  - why does coffee give me such an energy boost

Cluster 13  (n=1083)
 Top terms: insurance, health, plan, coverage, income, exchange, marketplace, company, employer, care, state, aca
 Top tags: insurance, health insurance, affordable care act, health insurance exchange, health insurance marketplace, medicaid, medicare, dental, obamacare, vision
 Examples (original):
  - is liposuction covered by insurance
  - i manage a medical office with 3 employees  rather than offer a health insurance plan we pay 50 of the employees premium so if they purchase their insurance through the marketplace will we no longer be able to do that
  - will they accept obamacare at any hospital
  - if you already have a non  employer based health plan can you still go to the exchange  can you see if you can get a better plan at a lower rate
  - what coverage is available to native americans under the affordable care act

Cluster 14  (n=623)
 Top terms: surgery, option, pain, hernia, bypass, cancer, exercise, knee, doctor, marijuana, leg, area
 Top tags: surgery, pain, hernia, exercise, leg, smoking, knee, pregnancy, walking, marijuana
 Examples (original):
  - how do you know what the best exercise routine is  i have had bariatric bypass surgery in 2010 i went from 340 to 232 and have a lot of access skin i also have fibromyalgia and arthritis that is not able to be controlled at the present time it is my desire to run a mini marathon but i do not even know where to begin on setting myself on the proper program i do not have the money to go to a trainer so need some direction please thank you  vicki
  - can my partner and i have sex with hpv  i am a 21 year old homosexual male my partner and i both have hpv however recently i have had a breakout with anal warts i am scheduled to have surgery over the next 6 months to have the warts removed but will my partner and i be unable to have safe sex during this time or is there a dangerous risk of me infecting him with the warts regardless if he already had the virus
  - the heel of one foot is very sore when i walk on it could this be related to my back surgery  i had back surgery and some times i have shooting pains down the opposite sore foot leg is the sore foot something i need to be concerned about
  - triple bypass completed 7 months ago need my brother be concerned about exercise increasing my heart rate above 110  quit smoking before surgery and have lost weight he is 58 and walking every night for 40 minutes and climbing 50 stairs and a hill up and down he feels fine and does not feel it necessary to stop but someone told him he should not even be mowing a lawn because he has heart disease
  - sack of fluid in scrotum previously had surgery for strangled testicle early twenties surgery 5 yrs ago foggy memory dealt with testicle being strangled my understanding was i had a hernia that released fluid into scrotum belive the strangled testicle was seperate problem solved in same surgery  1 now have a sack of fluid near the same testicle size of an acorn i dont belive it is attached to the actual testicle  2normal for previously strangled testicle to appear almost in pieces   3 that testicle also feels to have extra veins around it stitches

Cluster 15  (n=1837)
 Top terms: blood, diabete, anemia, hiv, vaginosis, sugar, type, diabetes, level, doctor, aid, disease
 Top tags: diabetes, anemia, hiv, bacterial vaginosis, aids, type 1 diabetes, blood sugar, infertility, atherosclerosis, vision
 Examples (original):
  - i was diagnosed with mononucleosis 7 months ago since then i have not been able to recover
  - i was curious about anal used mothers sex toy didnt clean it at risk for stds do not think it was used in a while i was curious and i found a vibrator and i used it i put a condom on it but condom broke i got tested for chlamidia and ghonorea both negative do you think i am at risk for hiv or anything else  also i used other sorta home made toys over a year ago and i just got worried i could have done damage to my body have not had any negative symptoms and havnt used them since last year should i be worried everything is normal and during use nothing negative happened like bleeding of anything
  - i had sex in the 5th week of pregnancy and saw vaginal bleeding why
  - i was diagnosed with mononucleosis 7 months ago since then i have not been able to recover
  - 3 yr old son has small specks of blood on face after napping

Cluster 16  (n=1127)
 Top terms: cancer, prostate, wart, tumor, treatment, colon, symptom, radiation, brain, lymphoma, doctor, hpv
 Top tags: prostate cancer, cancer, wart, colon cancer, benign prostatic hyperplasia, brain tumor, human papillomavirus, prostate, chemotherapy, lung cancer
 Examples (original):
  - can i transmit genital warts seventeen years after having them removed
  - how to treat leiomyosarcoma and rectal cancer at the same time
  - i had ovarian cancer and reflux surgery i still deal with constant nausea i can barely eat and i am unable to live life and go anywhere
  - i have a basal cell carcinoma what specialist should i consult
  - can hot showers a day cause cancer  i love to take hot showers and take three 5 10 minute showers a day   morning noon and evening is there a danger of cell damage or cancer from this practice also are three too many i like the water very hot is this bad our water in our city is chlorinated should i be concerned

Cluster 17  (n=2581)
 Top terms: pain, hair, doctor, neck, shoulder, loss, surgery, muscle, arm, head, scalp, back
 Top tags: hair, hair loss, neck, head, scalp, muscle, vision, shoulder, arm, lupus
 Examples (original):
  - i have heavy pain in both side of my head that causes dizziness sometimes in my back and neck
  - broken collarbone 3 5 cm overlap its been three weeks after break and still feels broken or loose
  - i need relief from chronic epididymitis
  - i have an acute dextroscoliosis i feel pain when i skip meals
  - my body has not been feeling good at all what can be wrong

Cluster 18  (n=629)
 Top terms: anxiety, depression, attack, panic, disorder, symptom, medication, xanax, pain, heart, help, feeling
 Top tags: anxiety, depression, panic attack, pain, stress, fear, heart, chest, drug, vision
 Examples (original):
  - pain when urinating inconsistent urination painfull ejaculation painfull mastrubation etc oh gosh im in all sort of trouble here and its given me anxiety over the past 1 5years ive been having this condition it all began when my urine penis started smelling cheesy after urination then later on when i was laying in bed and rising up i could feel from pelvic area like something is almost pushing my urine out it happened all the time then came premature ejaculation painfull urination painfull mastrubation painfull ejaculation also when i drag back my foreskin pain help please
  - should medical marijuana be used to treat anxiety disorder
  - my primary doctor diagnosed me with anxiety disorder and prescribed xanax and zoloft why do i still have panic attacks
  - can anxiety a side affect to depression
  - social anxiety i want friends where do i go i have such bad social anxiety when i first get to know someone but then it lessens

Cluster 19  (n=841)
 Top terms: chest, pneumonia, pain, rib, side, tuberculosis, ray, doctor, lung, tube, tb, opinion
 Top tags: pneumonia, chest, tuberculosis, walking, lung, i experienced a nstemi ha i had two stents fitted, chronic obstructive pulmonary disease, x-ray, intra-vaginal ultrasound, ultrasound abdominal vessels procedure
 Examples (original):
  - i have had a pneumonia shot can i get either a sinus infection or walking pneumonia from my 6 year old grand daughter  and can i be a carrier to others in my age group
  - how soon should my 14 year old wait before returning to school having been diagnosed with pneumonia
  - i felt like electric shock like feeling which became sharp on my left chest
  - what are the symptoms of pneumonia
  - heart attack i am a parapalegic and the other night when i was going to bed i had a burning sensation that started in the chest area and moved to the back to the point my upper torso was burnig all the way around i was in total discomfort i felt a heaviness in my chest this lasted for about 3 hours are these symptoms of a heart attack or are these symptoms that have to do with my spinal cord injury i have had these symptoms before but never to the degree i had the other night

Cluster 20  (n=760)
 Top terms: baby, period, pregnancy, sex, husband, birth, daughter, pain, food, tube, test, bottle
 Top tags: baby, pregnancy, period, sexual intercourse, pain, feeding, breastfeed, movement, newborn, smoking
 Examples (original):
  - can newborn babies be born addicted to prednisone if the mom took it for asthma in the last trimester
  - how much nap time does my baby need
  - is a baby more likely to be colicky with a baby bottle
  - how to manage large kidney stones that can not be passed on its own while being 16 weeks pregnant and not harm the baby stones are roughly 5 8 mm in size recurrent episodes of pain over the past 2 3 weeks should i stent how safe are the pain meds percocet and dilaudid for the baby
  - is it possible that a baby could be conceived on august 29 2011 and deliver june 27 2012

Cluster 21  (n=1077)
 Top terms: sex, period, condom, pill, pregnancy, intercourse, boyfriend, birth, chance, control, test, husband
 Top tags: sexual intercourse, pregnancy, period, condom, ejaculation, ovulation, birth control, pregnancy test, vagina, spotting
 Examples (original):
  - could i possibly be pregnant  last period may 16th unprotected sex on june 9th supposed to start june 16th still have not if you think i am pregnant when should i take a test  side notes  feel as if i start but do not   i was throwing up at 2 am on saturday the 15th i was nauseous the rest of the day   light cramps  the guy i had sex with says he only has a 3 chance of getting someone pregnant  i have been tired lately i also have been having light heartburn i think  if anyone can help me it would be greatly appreciated
  - i am on the pill and a condom was used pregnant  i have been on the pill for over 5 years i am pretty good with taking it on time but occasionally i will forget a day but immediately take it when i realize ive missed it last weekend saturday i was about an hour late taking my bc the next day i had sex he was wearing a condom a week later getting cramps and what not like a period i usually take the pills continuously but i am scared so i am going to let myself have a period hopefully i am about to start the sugar pills today
  - my boyfriend and i had sex using both the pill and a condom the condom broke should i worry about rubber inside me  we think most stayed on his shaft and i found a small piece about 34 inch x 12 inch like it ripped and a portion tore off do i have to worry about this causing tss 
  - had sex a week later instead of my normal period had alittle blood when wiped now im sick to stomach and nipples itch could i be pregnant
  - what can cause a period to be late  i am 25 have always had regular periods and i am now 3 days late i had protected sex 4 days ago i have been under a lot of stress and not sleeping well could i be pregnant or what else could cause my period to be late

Cluster 22  (n=302)
 Top terms: xarelto, xdr, xolair, xray, yard, yasmin, yaz, yeast, wreck, wrestling, wrinkle, wrist
 Top tags: pregnancy, exercise, coldness, ovulation, cold, epilepsy, fertility, body rash, malaria, i have been feeling very depressed
 Examples (original):
  - is leucoderma curable
  - i am hardly eating anything and when i do i feel nauseous
  - could i be pregnant please help
  - what is arthropathy
  - can you get pregnant if you take adderall

Cluster 23  (n=611)
 Top terms: herpe, hernia, outbreak, partner, simplex, sex, virus, sore, area, treatment, bump, symptom
 Top tags: genital herpes, herpes, hernia, virus, cold sore, herpes simplex, acne, lip, vagina, sore
 Examples (original):
  - can herpes be spread by bed bugs if a person infected with herpes is bitten by a bed bug can another person bitten by the same bug get infected with herpes
  - i take steroid prednisolone for ivfpregnancy due to auto immune issue but i have herpes 2 will this affect baby
  - how is herpes simplex treated
  - herpes transmission somewhat complicated question i have genital herpes but no outbreaks after the initial one my partner got it from me and has genital outbreaks we are wondering if we are able to spread it to each other in areas where we have not had outbreaks during shedding not active outbreak seems obvious that we could spread it around while there are lesions so we are careful during her outbreaks but the big mystery is  can we spread it to new areas during shedding    thanks a million we have been searching hard for the answer 
  - what are the treatments for a hernia

Cluster 24  (n=2134)
 Top terms: pill, acne, migraine, face, skin, headache, birth, nose, sinus, control, spot, condom
 Top tags: acne, migraine, sinus infection, nose, birth control, birth control pill, condom, over the counter, scar, atopic dermatitis
 Examples (original):
  - i have been taking propranolol for the chest pains now have headaches and pain on left side of head and body
  - how effective are male condoms at birth control
  - are diet pills safe for teenagers if so which ones are
  - i swallowed 20 tablets of 40mg citalopram whay should i do
  - can diabetes cause you to have chronic migraines

Cluster 25  (n=833)
 Top terms: tooth, hepatitis, liver, denture, dentist, gum, brace, mouth, toothpaste, root, wisdom, canal
 Top tags: tooth, hepatitis c, hepatitis, virus, mouth, liver, wisdom tooth, hepatitis b, infection, dental
 Examples (original):
  - can you test positive from having the hep b vaccine
  - would braces close wide gap between front teeth
  - how do you get hepatitis c
  - i brush my teeth regularly but i am not a big time flosser what are some tips for making daily flossing simple & fast
  - is cirrhosis a form of liver cancer

Cluster 26  (n=839)
 Top terms: cold, cough, throat, symptom, nose, flu, water, weather, doctor, infection, chest, sinus
 Top tags: coldness, cold, cough, throat, chest, drinking, pain, flu, nose, head
 Examples (original):
  - my baby ate her on poop my baby ate poop 4 days later she is sick weezing coughing and high fever for 4days straight i took her to the doctor and they said shes fine just a normal cold i told them what happen and they just said she should be fine but if she still has a fever next week come back what should i do and is her symptoms related to her eating her poop
  - i smoked cigs for 1 month averaging about 3 a day just wondering if any irreversible was done i did quit since then i started smoking for a month after a period of depression a couple of months ago 2 3 cigs most days with a couple more on bad days i would estimate i probably had 4 packs total over the period i went cold turkey as i started to get my life together and hated the ill feeling from them i exercise regularly and eat healthy and i am still young i would just like to clear my head and hear that i did no damage permanent to my lungs i know it takes a bit to recover hopefully to 100
  - what is the treatment for the common cold
  - going on a long flight with a cold need to sleep sudafed with xanax or sudafed with nyquil  i have a cold and i am going on a very long flight i need to sleep would i be better off taking sudafed with xanax or sudafed with nyquil
  - after being sick i feel like i have hair stuck in my throat and i have a tiny knot under my chin what could this be  ii was sick 4 weeks ago with coldsinus issues but it cleared up i feel like i constantly have to clear my throat and brush my teeth nothing makes the feeling in the back of my throat go away

Cluster 27  (n=483)
 Top terms: hypothyroidism, osteoporosis, thyroid, hyperthyroidism, synthroid, nodule, level, risk, medicine, tsh, hormone, weight
 Top tags: hypothyroidism, osteoporosis, thyroid, hyperthyroidism, hormone, drug overdose, pregnancy, tsh levels and pregnancy, surgery or nuclear medicine better, vision
 Examples (original):
  - what should i do if i suspect an overdose of thyroid
  - what causes peyronie is disease
  - i have all the classic signs of hypothyroidism but lab results normal where to go from here
  - im 27 and ive had almost all of the symptoms since i was about 6 years old could it be possible i have hypothyroidism
  - what would make my tsh levels suddenly jump to critically high level when i feel the same  i take synthroid 1 25 daily and yet my tsh is now 28 3 increased from 23 on friday why would it jump so quickly my dr is on vacation and i just want to make sure this is ok to wait until thursday when he comes back 

Cluster 28  (n=724)
 Top terms: fever, pain, headache, throat, body, cough, symptom, son, chill, nose, grade, skin
 Top tags: fever, pain, running, cough, headache, antibiotic, swelling, skin, sweating, ears
 Examples (original):
  - i have persistent headache and i feel like i have lowgrade fever help  hi so i am a headache everyday it is not too bad though it is completely bearable but a little distracting and i have noticed that i have lowgrade fever most of the time or mostly everyday but just like the headache it is bearable i can not just shrug this feeling off this have been occurring for two or three months now i am hopefully going to the doctor in a few days and get myself checked
  - why do i feel lightheaded fatigued and sweat during sleeping no fever  i am an almost 37 yr old female with a lot of stress right now dr put me on effexor and i started not being able to sleep having bad headaches feeling lightheaded and constipated i took it for 2 weeks and he told me to stop when i called him he called me in something else but i am afraid to get it i have been on paxil prozac and celexa and never felt this horrible i have not taken anything in almost a week but feel lightheaded many times thoughout the day any ideas what could be wrong
  - can i have strep without fever  my 4 year old son was diagnosed with strep throat 3 days ago last night i was fine one minute and suddenly felt like i would been hit by a ton of bricks body aches headache  sore throat and general feeling crappy but no fever is there any point in dragging myself out to doctor when i feel so miserable is it possible to have strep without fever i do not have any runny nose stuffy nose or cough not a cold 
  - my son is 7 and can swallow pills he is running a fever can i give him one 200 mg ibuprofen pill he weighs 53 pounds
  - should i go to the er for severe right earache or wait until monday is appointment  i am 20 on thursday i woke with dull throbbing it has gotten worse i have a 101 f fever i think the throbbing is still there but now my ear canal is so blocked i can not feel anything in the canal but there is still sharp stinging pain in what feels like the back my outer ear is swollen the pain extended to my throat chewing or hiccups feels like my ear is tearing i have almost zero hearing in this ear er or wait

Cluster 29  (n=1143)
 Top terms: exercise, weight, muscle, diet, pain, minute, type, blood, workout, program, heart, pound
 Top tags: exercise, weight, diet, muscle, walking, pain, running, workout, vision, calorie
 Examples (original):
  - do doctors immediately prescribe medication to control blood sugar or do they wait to see if diet and exercise help
  - what exercise does not aggravate achilles tendonitis  i am trying to rest my achilles so i can return to playing tennis in the meantime i want to exercise to stay in shape what types of exercise can i do
  - what is resistance exercise
  - i am 20 and i have missed a period is that normal  my period is usually irregular  once in 9th grade i missed it for 3 months i will be 20 next week and i have missed my period this month the first week of april i did a lot of workexercise that i do not usually do but i have not done it since then also about three weeks ago i completely cut caffeine from my diet cold turkey also i am a virgin i have never done anything sexual
  - what is a weight loss exercise for people with arthritis in feet ankles and disc herniation in neck and low back  swimming irritates the neck and pain shoots down my arms walking in good sneakers pains my feet and ankles which have arthritis i can stretch but that is not enough to lose weight

Cluster 30  (n=1209)
 Top terms: food, weight, diet, water, pound, drink, blood, milk, loss, soda, sugar, alcohol
 Top tags: food, drinking, diet, weight, smoking, exercise, weight loss, meal, corpulence, vision
 Examples (original):
  - what weighs more muscle or fat  im just wondering about weight gain due to muscle growth my wife has been working out for some time with weights and cardio training but she is finding that her weight has been fluctuating and at times gains weight a little bit
  - can you be allergic to mold in your food
  - i have started a low sugar and low wheat diet and i keep going to the toilet more than normal is my diet the reason
  - what can i do to gain back my missing pounds and feel healthy again  i have been sick and lost 17 pounds i am fatigued all the time and look poorly i want to gain my wieght back and feel good again as quickly as possible
  - can the paleo diet raise my cholesterol my wife is on the paleo diet and i have joined her to some degree i eat two eggs every morning along with 2 pieces of bacon i use coconut oil along with real butter to cook with could these factors be raising my cholesterol

Cluster 31  (n=836)
 Top terms: skin, bump, spot, acne, product, area, face, patch, ringworm, wart, doctor, rash
 Top tags: skin, acne, vision, burn, penis, rash, itch, hand, lump, head
 Examples (original):
  - guest in my home has scabies do i have house sterilized or will a good cleaning do it we have not had skin contact he is getting treated and i am having a general cleaner come in this afternoon to change linens etc is this sufficient i have never had anything like this in my home somewhat disturbed
  - what is the best moisturizer for older skin
  - what medicines can cause your skin to turn blue
  - the most effective skin care  i am for normal skincare but do not take my word for it try some out yourself study customer opinions and make your brain up next the most effective businesses may have a no risk assure you can send total or applied containers back and obtain a complete refund if you are not 100 pleased skin care>> <link>
  - my skin is irritated after i treated a flea infestation in my hair with kerosene how can i reduce skin irritation

Cluster 32  (n=476)
 Top terms: breast, milk, nipple, mammogram, lump, cancer, doctor, biopsy, pain, surgery, reduction, discharge
 Top tags: breast, nipple, pregnancy, mammogram, milk, breast cancer, lump, tenderness, pain, ultrasound
 Examples (original):
  - i have recently had breast surgery removal of calcium deposits with cancer cells
  - i have had rapid breast growth want breast reduction surgery what age should i typically get this surgery
  - why do my breasts hurt and feel and look like they are getting bigger  for almsot 2 weeks now my breasts have been painful and feel like and look like they are getting bigger i instantly thought of being pregnant but i am on the mirena birth control and although its not 100 effective i still did not believe i were pregnant i took a home pregnancy test from my local drugstore and it was negative why are my breasts feeling this way people i know have been commenting on them looking huge or bigger than normal
  - will health insurance through the marketplace cover genetic testing for breast cancer
  - msra on the breast i got the mrsa infection from cleaning motel rooms when should i be able to go back to work do i need to worry about losing it

Cluster 33  (n=530)
 Top terms: rash, arm, hand, bump, penis, skin, body, ivy, thigh, cream, area, leg
 Top tags: rash, arm, skin, hand, penis, thigh, leg, itch, reaction, wrist
 Examples (original):
  - can an antibiotic through an iv give you a rash a couple days later
  - i have a rash under my armpit what kind of treatment should i use a group of 7 red bumps slightly raised and itchy just really trying to figure out how i can start the healing process and get them to go away thanks
  - can poison ivy rash return same place after 2 months  rash returned same places on arms after 2 months since had first rash  have not been any where near any plants no pets to come in contact with poison ivy any one ever hear of this happening
  - is there such a thing as a weather allergy  for the past three years in the autumn i will start getting itchy red scaly rash from wrist to elbow it will last from september to usually april may it remains consistent through these months i remain flare free in the summer months i have asked my doctor countless times and he sent me home with a very strong hydrocortisone cream i have resorted to over the counter hydrocortisone which burns i can not use uv therapy as i am allergic to the sun and i am currently on an antihistamine for that
  - treating a groin area rash with daily apps of triamcinolene cream for months i stop for 1 day and rash returns ideas  identified by dermatologist as related to jock itch prescription is used up so i am looking for otc alternatives or   best case   permanent solution

Cluster 34  (n=886)
 Top terms: penis, vagina, sex, discharge, condom, sperm, area, infection, ejaculation, yeast, pain, bump
 Top tags: penis, vagina, pregnancy, burn, ejaculation, pain, masturbation, condom, finger, yeast infection
 Examples (original):
  - will my glans burn recover  accidentally i got a very hot water on my penis the water hit a small area of glans and the area under it now rubbing it either by hand or cloths has some sort of annoying feeling i would like to know whether recovery is possible or not if yes please tell me how
  - what other than a yeast infection could it be if the medicine does not work and more symptoms start occuring  for a few months now i have had constant vaginal itching and burning and after using yeast infection medicine multiple times it still has not gone away and now blisters have begun forming after looking up genital herpes i have noticed that i have nearly all the symptoms for it but i have yet to have sexual activity could it still be possible or is it something else
  - why will not my penis stay hard when in pregame  i get hard quite easily when around my gf but then all of a sudden when it comes to me taking my jeans off it goes down why  when it does decide to work i really do love sex with her so what is causing this also i can not cum when she tries to give bj
  - inserting finger into girls vagina leads to pregnancy
  - can a yeast infection cause pain in the urethra as well as the head of the penis  originially thought to have prostatitis a month ago have been experiencing penis soreness urethra pain some redness gonnohrea and chlamydia negative and 3 weeks of cipro provided no relief now thinking it may be fungal could this be a yeast infection and if not what could it be

Cluster 35  (n=544)
 Top terms: pressure, blood, hypertension, bp, medication, heart, doctor, problem, med, rate, lisinopril, cause
 Top tags: high blood pressure, blood pressure, low blood pressure, heart, pressure, dizziness, headache, pulse, pregnancy, sweating
 Examples (original):
  - why see a renal specialists for high blood pressure
  - what does blood pressure of 123 over 101 with pul 75 mean
  - i have low blood pressure low heart rate tired and lack energy is it normal
  - why would a healthy 83 year old woman have high blood pressue
  - i am a indian and 21 years old i am a post kidney transplantation patient of six week from now i have blood pressure of 165 to 150 can i do bodybuilding is any any affect of increasing blood pressure by doing exercises

Cluster 36  (n=1453)
 Top terms: period, pregnancy, test, sex, birth, pill, control, cycle, symptom, chance, breast, condom
 Top tags: period, pregnancy, pregnancy test, birth control, spotting, breast, condom, ovulation, ejaculation, cramps
 Examples (original):
  - my period only last 36 48 hours which is my norm is that why i have had 2 yrs of no luck getting pregnant  my husband and i have been trying for two years to have a child i am turning 30 next month and in my family after 30 equals issues my normal period is only 36 48 hours could this be preventing me from getting pregnant
  - spotting on day two of my period could i be pregnant i have had my loop taken out 71015 and have had unprotected sex a few times after that hoping to fall pregnant my period was meant to start 12102015 it now day two on my period and i have only been spotting which is very un usual could i be pregnant
  - pregnant unprotected sex a week before period period came on time and heavy with bad cramps as usual reg 28 day 4 yrs i had unprotected sex a week before my period started he ejaculated awa from me but im worried a little bit may have got it before he pulled out my period came on the dot when it was supposed to get it and was heavy at first then to moderate with bad cramps like i normally have basically my period came on time and was normal in length flow and cramps my periods have been regular for years i do not know when i ovulate or my latueal phase what are my chances of being pregnant
  - my period has been late by 4 days i am trying to conceive please help
  - i have been off depo shot for over a year and have no period i recently began having pregnancy symptoms am i pregnant  i was on the depo shot for 2 years and have not gotten the shot since september 2011 i have still not had a period and have been having pregnancy symptoms my fiance and i use the withdraw method i realize that you can still get pregnant this way after over a year of not receiving the shot i do not think the symptoms could be attributed to the shot i had a pap smear 9 12 12 and it came back fine so could i be pregnant and if not what could be causing the pregnancy symptoms

Cluster 37  (n=885)
 Top terms: flu, shot, injection, depo, vaccine, steroid, influenza, period, child, insulin, symptom, testosterone
 Top tags: injection, flu, pregnancy, period, steroid, antibiotic, sexual intercourse, pain, testosterone, depo-provera
 Examples (original):
  - why would a rn choose not to get her kids a flu shot as the grandparent is there anything i can do
  - was skiing and fell on my knee cap now 3 days later i just heard a pop and its throbbing and its severe pain  they did an xray but they did not have an mri machine available i will not be able to access a doctor for another 3 9 days a half hour the pop happened and it really hurts on a scale of 1 10 10 being like you have just been shot its more of an 8 7
  - can allergy shots be used to treat asthma
  - treatment for household cleaning product allergies symptoms are fatigue shakiness nasalchest burning weakness and asthmatic reaction or shortness of breath already getting immunology shots for seasonal dust mites cat dander and mold  does not appear to help the cleaning product symptoms
  - what happens to someone when they get influenza

Cluster 38  (n=872)
 Top terms: medication, drug, medicine, treatment, counter, med, prescription, effect, blood, pressure, xdr, ibuprofen
 Top tags: drug, over the counter, pregnancy, ibuprofen, coldness, pain, drug test, tuberculosis, vision, smoking
 Examples (original):
  - i have been thinking about buying drugs online to save money is that a good idea
  - i am extremely agoraphobic despite help from my psychologist psychiatrist and medication why else can i do
  - is avinza a narcotic medication
  - does soma show up on a drug urinalysis
  - what medications and treatments are available to treat pagets disease of bone

Cluster 39  (n=424)
 Top terms: vitamin, supplement, multivitamin, deficiency, folic, acid, energy, calcium, level, pill, pregnancy, woman
 Top tags: vitamin, vitamin d, multivitamin, pregnancy, prenatal vitamin, supplement, folic acid, calcium, vitamin c, diet
 Examples (original):
  - is all vitamin d the same
  - how to correct vitamin b12 deficiency
  - when is the best time to take vitamins in the morning or at bedtime
  - is it true that calcium makes garcinia cambogia ineffective
  - hi i am 15 years old take nature made b complex omega369 100mg ubiquinol daily is this okay  i am using nature made super b complex omega 3 6 9 and qunol 100mg ubuiquinol as alternative adhd medicine i am just concerned that these pills maybe detrimental i have read the label on the nature made super b comples and it contains 6 667 daily value of thiamin 1 176 riboflavin and 250 b12 also the qunol has 100mg of pure coq10 i have heard the reccomended amount was around 50mg is this too much for my body to handle should i cut the pills into smaller portions please help  thanks

Cluster 40  (n=1614)
 Top terms: disease, disorder, depression, parkinson, brain, epilepsy, seizure, dementia, mri, schizophrenia, memory, medication
 Top tags: bipolar disorder, "parkinsons disease", depression, epilepsy, dementia, schizophrenia, brain, seizure, postpartum depression, celiac disease
 Examples (original):
  - what are the dietary restrictions for celiac disease gluten
  - had a stroke on the brain in 2012 its 2016 i cant get no more than 5 hours of sleep a day
  - where can i go for help for bipolar disorder
  - does prozac cause weight gain what about zoloft
  - my husband is taking 40 mg of prozac and is really depressed and has thought of suicide what do we do

Cluster 41  (n=872)
 Top terms: kidney, urine, uti, gallstone, stone, bladder, infection, pain, gallbladder, urination, blood, problem
 Top tags: gallstone, kidney, kidney stone, urination, burn, kidney infection, bladder, vision, urinalysis, overactive bladder
 Examples (original):
  - i have been suffering from low grade fever excessive saliva at night dark and less urine
  - how are gallstones diagnosed
  - why does my urine smell
  - is it safe to swim in a lake if you have a uti  i have a bladder infection and began antibiotics two days ago we are going to the lake this week can i swim in the lake or should i avoid that
  - my urine has a bad smell and is cloudy what can be wrong

Cluster 42  (n=1296)
 Top terms: infection, antibiotic, chlamydia, yeast, strep, throat, penicillin, amoxicillin, cyst, bacteria, std, doctor
 Top tags: antibiotic, chlamydia, bacterium, yeast infection, penicillin, amoxicillin, virus, throat, strep throat, cyst
 Examples (original):
  - can taking multiple antibiotics cause redness and dryness of vagina
  - is clindamycin effective in treating syphilis
  - why it is necessary to take 2 antibiotics for a diverticuilitis infection will taking just the cipro work
  - how long does it take antibiotics to flush from your system after you stop taking when taking cefdinir 300 mg for a sinus infection i developed diarrhea and can not seem to stop it nurse line said to stop taking and it would take some time to get it out of my system how long is that time
  - can you have a stiff neck with pneumonia  i was treated for pneumonia with an antibiotic and prednisone for my chronic asthma i have recently noticed a stiff neck that seems to worsen by the end of the day can this be a side effect from the medicine or pneumonia or should i be concerned with a completely different illness

Cluster 43  (n=1154)
 Top terms: pain, side, doctor, chest, back, arthritis, area, migraine, neck, leg, knee, sensitivity
 Top tags: pain, burn, chest, arthritis, swelling, vision, neck, leg, pregnancy, arm
 Examples (original):
  - pain ring finger to the middle of arm before elbow for 45 days started after i held on to stop falling worse on lifting its not bad3 on 10 earlier it was more painful but now it does not hurt as much unless i type or write i have barely used my right hand for anything for the past month else the pain increases it starts hurting at one point about five fingers from my wrist but pain goes away completely if i press down on it it also hurts in the area below middle to pinkie i got an xray done already so nothing there shd i get an mri if so only for wrist or forearm also or shd anti inflammatories be enough thanks
  - what are the ingredients inibuprofen  i take a 600mg ibuprofen only as needed for nerve pain my question is what ingredients are in this medication i have a legal prescription for it
  - i have been having very sharp stabbing pains down through the top rtrear of my head the pain almost knocks me down i have been having these pains for 6 7 weeks i have had no previous head injuries they just started out of the blue they are not headaches they are in  a dime sized spot on top of my head right side just off center back portion top of head does that make any sense these pains happen wether  i am standing or laying down thank you for your time
  - does lidocaine cure canker sores on your throat  i found a small white sore on my throat and i was prescribed lidocaine and it numbs the pain but i was wondering if it cures it at the same time before it gets worse
  - can age appropriate arthritis be exaserbated by barometric pressure change or other weather conditions  im 55 very fit and athletic and often suffering with joint pain and stiffness like never before and all over my body not sure where it came from or how to get rid of

Cluster 44  (n=1224)
 Top terms: diarrhea, stomach, stool, colitis, bowel, pain, constipation, movement, disease, abdomen, blood, crohn
 Top tags: diarrhea, bowel movement, ulcerative colitis, constipation, "crohns disease", gastroenteritis, colonoscopy, stomach pain, "travelers diarrhea", ibuprofen
 Examples (original):
  - my butthole hurts and there is a bump and it hurts whenever i sit or move i assume it happened a few days ago when i was trying to hold in my bowel movement but i could not so i went to the bathroom but i do not remember anything happening to me for the rest of the day or the next but on sun today8th 9th its been bothering me severely and it hurts whenever in doing anything that requires me to move my lower body
  - my fart hurt the back of my head on the left side  this is a stupid question but i farted and the left side of the back of my head started hurting for a few seconds did this fart damage my brain or anything
  - after a bowel movement abdominal pain puts me on my knees severe nausea to the point i do vomit
  - can you take ibuprofen when you are breastfeeding
  - i began to feel a sudden twitchspasm feeling in my stomach went to er twice no solution

Wrote assignments to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/clustered_short_questions.csv

=== Assessment Metrics (COSINE) ===
 Features: BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245) = 100D
 Global: silhouette_cosine=0.1229

--- Cluster Size Stats ---
 k=45 | min=302 max=4496 mean=1055.4 median=872.0 std=704.0

--- Per-Cluster Averages ---
 cluster | size | mean silhouette (cosine)
       0 | 1208 |                 0.0603
       1 |  489 |                 0.1351
       2 |  780 |                 0.1521
       3 |  833 |                 0.2306
       4 |  701 |                 0.2267
       5 | 1639 |                 0.1024
       6 |  634 |                 0.2971
       7 |  938 |                 0.0992
       8 | 1900 |                -0.0270
       9 |  685 |                 0.2597
      10 |  330 |                 0.4705
      11 |  956 |                 0.1026
      12 | 4496 |                -0.0754
      13 | 1083 |                 0.1092
      14 |  623 |                 0.3287
      15 | 1837 |                -0.0169
      16 | 1127 |                 0.1103
      17 | 2581 |                -0.0385
      18 |  629 |                 0.4494
      19 |  841 |                 0.0672
      20 |  760 |                 0.3647
      21 | 1077 |                 0.2342
      22 |  302 |                 0.9137
      23 |  611 |                 0.2521
      24 | 2134 |                -0.0519
      25 |  833 |                 0.1352
      26 |  839 |                 0.2433
      27 |  483 |                 0.2423
      28 |  724 |                 0.3380
      29 | 1143 |                 0.2398
      30 | 1209 |                 0.1016
      31 |  836 |                 0.2851
      32 |  476 |                 0.1765
      33 |  530 |                 0.2954
      34 |  886 |                 0.1800
      35 |  544 |                 0.2555
      36 | 1453 |                 0.2577
      37 |  885 |                 0.2628
      38 |  872 |                 0.3418
      39 |  424 |                 0.2865
      40 | 1614 |                 0.0130
      41 |  872 |                 0.0977
      42 | 1296 |                 0.0312
      43 | 1154 |                 0.1278
      44 | 1224 |                 0.0488

--- Structure (cosine on scaled space) ---
 Avg intra-cluster cosine similarity: 0.3074 (higher = tighter)
 Mean inter-centroid cosine similarity: -0.0204 (lower = better)

--- Tag Coherence (avg) ---
 Purity=0.487 | Entropy=4.313 | Intra-Jaccard=0.117

--- Model Selection Top Rows (sorted by cosine silhouette) ---
 k  silhouette_cosine  max_cluster_diameter_cosine
45           0.122891                     1.432196
44           0.121567                     1.452643
43           0.119432                     1.432757
42           0.118345                     1.450955
37           0.117396                     1.490952
41           0.115392                     1.420857
39           0.115335                     1.482598
36           0.114750                     1.450955
40           0.114228                     1.425299
35           0.114138                     1.457310

[UMAP] Saved 2D scatter to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_scatter.png
[UMAP] Saved embedding CSV to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_embedding.csv

[HClust] Saved dendrogram to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/hierarchical_dendrogram.png

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_timings.json

=== Runtime (seconds) ===
               load_csv_s:   0.2580
         nlp_preprocess_s: 115.3173
    vectorize_text_tags_s: 101.4866
         scale_features_s:   0.0623
        model_selection_s: 419.3490
     interpret_clusters_s:   0.4339
           save_outputs_s:   0.2585
        metrics_compute_s:  24.5368
                   umap_s:  31.9565
           hierarchical_s:   0.2596
          total_runtime_s: 693.9252
```

In v2.1.3 I'm clustering about 47,491 short medical questions after cleanup, which is essentially the entire corpus with same techniques of data preprocessing and NLP as in v1.1.3.

Before final clustering I standardize the 100D features and then L2-normalize rows, so the geometry is essentially cosine-based. For model selection I run KMeans across k from 10 to 45. For each k I fit KMeans on the scaled features and compute cosine silhouette and a sampled estimate of max cluster diameter in cosine distance. The silhouette rises in a nearly monotonic way from about 0.043 at k=10 to about 0.123 at k=45, with no clean elbow. Max diameter floats around 1.43 to 1.52 and trends slightly downward as k increases. The behavior is typical of large, noisy semantic spaces: more clusters gradually improve local purity, but global overlap never disappears. I pick k=45 as the best configuration by cosine silhouette, accepting that I'm in a regime where extra clusters continue to help modestly rather than collapsing into an obvious optimum.

The global structure metrics fit that story. The overall cosine silhouette is about 0.1229—modest in absolute terms, but not surprising for unsupervised clustering over 47k real-world questions with overlapping symptom space. Average intra-cluster cosine similarity is roughly 0.31, while mean inter-centroid cosine similarity is around -0.02. That combination tells me that members of a given cluster are clearly more similar to one another than cluster centroids are to each other, and that the centroids themselves are roughly orthogonal in this 100D space. In other words, the clusters cover diverse semantics without collapsing into a single blob.

The cluster size distribution is quite imbalanced: with k=45, the smallest cluster has just over 300 points and the largest has around 4,500, with a mean of about 1,055, a median under 900, and a sizable standard deviation. That means a few clusters behave as grab-bag "super bins," while some are small but very tight. When I look at per-cluster silhouettes, that imbalance appears again. One tiny cluster achieves a silhouette near 0.91 and is essentially an ultra-narrow intent. Clusters around shingles and varicella, or anxiety and panic, sit in the 0.45 range. Others, such as baby and pregnancy care, fever and flu, or medication-related topics, land around 0.34 to 0.37. Many more clusters occupy the 0.22-0.33 band, covering eyes, skin, thyroid, gastrointestinal issues, blood pressure, and other coherent topics. The problematic areas are a handful of negative-silhouette clusters—cluster IDs like 8, 12, 15, 17, or 24—which represent broad, overlapping symptom or family-problem regions and bleed heavily into neighboring topics. Those are natural "miscellaneous" or boundary clusters rather than crisp intents.

Tag coherence metrics add another layer. Mean tag purity is about 0.487, so just under half of each cluster's points share the dominant tag on average. Entropy is high, around 4.3, which reflects the multi-label nature of the questions: even in a coherent topic, different users mention different combinations of conditions and symptoms. Mean intra-cluster tag Jaccard is about 0.117, indicating that tag sets within a cluster overlap more than random but are far from identical. When I inspect specific clusters, the tag story aligns with the text. Pregnancy, sex, and periods tend to appear in a consistent set of clusters, cardiovascular topics cluster heart-related tags together, and kidneys and UTIs form another tag-coherent group. Insurance and ACA questions split into a couple of distinct clusters that clearly reflect subpopulations such as veterans and Medicare versus marketplace or employer coverage. Ear, hand and foot, dental versus liver, and bowel versus IBD clusters also show consistent tag patterns.

The visual diagnostics back up the numbers. Silhouette versus k shows an almost linear rise from about 0.04 to 0.12, with no saturation by k=45, suggesting that if I pushed k to something like 60 or 70 I might squeeze out a bit more purity at the cost of more small clusters. The max-diameter curve meanders in the 1.45-1.52 band with a mild downward drift, which matches a "fuzzy" structure where each cluster still spans a wide cosine distance range but gradually tightens. UMAP in two dimensions shows many small, tight blobs and some streaky arms where a single cluster dominates, along with a central overlap region where the big generic clusters live. A dendrogram over centroids using cosine distance reveals that most merges happen at high distances, around 0.9-1.0, with only a few very close pairs indicating near-duplicate clusters. For taxonomy-building purposes, that dendrogram can be used to roll related centroids into super-clusters like "Pregnancy and Sex," "Respiratory Infections," "GI and Bowel," "Mental Health," or "Dermatology."

The runtime profile is dominated by model selection. The overall pipeline takes roughly 694 seconds. POS tagging and spaCy consume around 115 seconds, BERT plus tag embeddings another 100 seconds, and KMeans sweeps with silhouette and diameter calculations nearly 420 seconds. UMAP takes about half a minute, while metrics and tag coherence add another 25 seconds or so. Everything else is minor. If I need to speed things up, the clearest levers are to reduce the k grid, introduce early stopping once silhouette gains flatten, subsample for the model-selection sweep and then re-fit the chosen k on the full dataset, or change the clustering method for the scan phase.

From a business or routing perspective, v2.1.3 looks like a 45-topic taxonomy that distinguishes many clinically intuitive domains: multiple pregnancy and sex clusters, GI and bowel groups, respiratory and infection clusters, dermatology and skin clusters, mental health and neurologic conditions, musculoskeletal and orthopedic issues, chronic diseases such as heart disease, thyroid disorders, kidney problems, liver and hepatitis, an oncology cluster, multiple insurance and ACA clusters, and several drug, medication, and vitamin clusters. There are also a few necessary "other" or noisy clusters that absorb cross-topic or multi-issue questions. Overall I'd summarize v2.1.3 as a pipeline that successfully organizes about 47,000 medical chatbot questions into semantically coherent groups using POS-filtered BERT embeddings plus tag-based supervision. Many clusters behave like stable anchors for taxonomy and intent design, with moderate intra-cluster similarity and consistent tags, while the modest global silhouette simply reflects the presence of a handful of large, heterogeneous clusters and the unavoidable overlap in symptoms across conditions.

## v2.3.3

```
Rows after POS cleanup: 47491
modules.json: 100%
 349/349 [00:00<00:00, 77.3kB/s]
config_sentence_transformers.json: 100%
 116/116 [00:00<00:00, 27.6kB/s]
README.md: 
 10.5k/? [00:00<00:00, 2.18MB/s]
sentence_bert_config.json: 100%
 53.0/53.0 [00:00<00:00, 13.1kB/s]
config.json: 100%
 612/612 [00:00<00:00, 145kB/s]
model.safetensors: 100%
 90.9M/90.9M [00:01<00:00, 109MB/s]
tokenizer_config.json: 100%
 350/350 [00:00<00:00, 86.5kB/s]
vocab.txt: 
 232k/? [00:00<00:00, 22.4MB/s]
tokenizer.json: 
 466k/? [00:00<00:00, 62.9MB/s]
special_tokens_map.json: 100%
 112/112 [00:00<00:00, 28.6kB/s]
config.json: 100%
 190/190 [00:00<00:00, 45.2kB/s]
[Features] BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245) = 100D
[Shapes] Combined feature matrix: (47491, 100)
[OPTICS] Using min_cluster_size=189 (frac=0.004, abs_floor=60)
OPTICS (cosine) found 4 clusters; noise points: 46650
[Clustering] Using OPTICS labels (n_clusters=4).
Saved cluster count summary to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_counts_summary.csv
[Final Clustering] Algorithm: OPTICS | labeled clusters: 4 | noise: 46650

=== Cluster Summaries (OPTICS) ===

Cluster 0  (n=221)
 Top terms: shingle, pox, vaccine, chicken, breakout, husband, treatment, area, chickenpox, child, herpe, vaccination
 Top tags: shingles, chickenpox, vaccination, infant, pain, short stature, herpes, i need shingles vaccine, pregnancy, i am suffering from shingles
 Examples:
  - i am 33 years old i need shingles vaccine but am i too young to have it
  - is shingles contagious
  - do shingles make you pee blue
  - twenty years old with shingles  i am only twenty years old and i have been diagnosed with shingles i feel like its spreading fast i am really worried and im not really sure what to do the doctor was really short with me and said theres nothing to do but let it take its course if anyone could help me out that would be great   thanks
  - i had chicken pox at the age of 21 and then at the age of 60 i had shingles can i get shingles again once i had it

Cluster 1  (n=191)
 Top terms: hepatitis, liver, virus, infection, symptom, prognosis, risk, vaccine, cirrhosis, mess, transmission, september
 Top tags: hepatitis c, hepatitis, virus, hepatitis b, infection, liver, hepatitis a, concerned about atrophic liver, just found out i have hepatitis c?, i have had symptoms of hepatitis
 Examples:
  - can you test positive from having the hep b vaccine
  - how do you get hepatitis c
  - who should receive antiviral therapy for hepatitis c virus
  - is it true that the hepatitis c virus cannot survive for more than two hours outside the human body
  - how many kinds of viral hepatitis are there

Cluster 2  (n=234)
 Top terms: hypothyroidism, osteoporosis, hyperthyroidism, risk, symptom, diet, treatment, tsh, pregnancy, hypothyroid, sign, thyroid
 Top tags: osteoporosis, hypothyroidism, hyperthyroidism, diet, pregnancy, tsh gone up to 6.63, classic signs of hypothyroidism, pregnancy and hypothyroidism, thyroid, food
 Examples:
  - i have all the classic signs of hypothyroidism but lab results normal where to go from here
  - im 27 and ive had almost all of the symptoms since i was about 6 years old could it be possible i have hypothyroidism
  - how is hypothyroidism treated
  - hypothyroidism does this make your hair fall out in unusual amounts
  - what can be done to avoid the consequences of hypothyroidism in pregnancy

Cluster 3  (n=195)
 Top terms: xarelto, xdr, xolair, xray, yard, yasmin, yaz, yeast, wreck, wrestling, wrinkle, wrist
 Top tags: epilepsy, body rash, malaria, i have been feeling very depressed, i do i feel nauseous when i eat, hepatitis c, heartburn, lymphedema, lupus, urination
 Examples:
  - is leucoderma curable
  - i am hardly eating anything and when i do i feel nauseous
  - what is arthropathy
  - how can ringworm be prevented
  - once i am stressed how can i calm myself

Wrote assignments to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/clustered_short_questions.csv

=== Assessment Metrics (OPTICS; labeled clusters only, COSINE) ===
 Features: BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245) = 100D
 Labeled clusters: 4 | Noise: 46650
 Global: silhouette_cosine=0.6733

--- Cluster Size Stats (labeled only) ---
 min=191 max=234 mean=210.2 median=208.0 std=17.9

--- Per-Cluster Averages (silhouette, cosine) ---
       0 | size= 221 | mean silhouette=0.6200
       1 | size= 191 | mean silhouette=0.7127
       2 | size= 234 | mean silhouette=0.4227
       3 | size= 195 | mean silhouette=0.9957

--- Structure (cosine on cluster space) ---
 Avg intra-cluster cosine similarity: 0.7011 (higher = tighter)
 Mean inter-centroid cosine similarity: -0.0025 (lower = better)

--- Tag Coherence (avg) ---
 Purity=0.450 | Entropy=2.630 | Intra-Jaccard=0.235

Saved model-selection metrics to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_model_selection.csv

--- Model Selection Rows ---
 min_cluster_size  min_samples  k  n_noise  silhouette_cosine  max_cluster_diameter_cosine
              189           20  4    46650           0.673258                     1.140345

[K-scan] MiniBatchKMeans for K-vs-Silhouette and K-vs-MaxDiameter (cosine):
  k=10 | silhouette(cosine)=0.0539 | max_diameter(cosine)=1.4334
  k=11 | silhouette(cosine)=0.0556 | max_diameter(cosine)=1.4514
  k=12 | silhouette(cosine)=0.0585 | max_diameter(cosine)=1.4573
  k=13 | silhouette(cosine)=0.0594 | max_diameter(cosine)=1.4645
  k=14 | silhouette(cosine)=0.0603 | max_diameter(cosine)=1.4910
  k=15 | silhouette(cosine)=0.0652 | max_diameter(cosine)=1.5108
  k=16 | silhouette(cosine)=0.0691 | max_diameter(cosine)=1.4463
  k=17 | silhouette(cosine)=0.0702 | max_diameter(cosine)=1.4463
  k=18 | silhouette(cosine)=0.0716 | max_diameter(cosine)=1.4526
  k=19 | silhouette(cosine)=0.0755 | max_diameter(cosine)=1.4380
  k=20 | silhouette(cosine)=0.0792 | max_diameter(cosine)=1.4261
  k=21 | silhouette(cosine)=0.0828 | max_diameter(cosine)=1.4744
  k=22 | silhouette(cosine)=0.0859 | max_diameter(cosine)=1.4744
  k=23 | silhouette(cosine)=0.0886 | max_diameter(cosine)=1.4522
  k=24 | silhouette(cosine)=0.0904 | max_diameter(cosine)=1.4632
  k=25 | silhouette(cosine)=0.0941 | max_diameter(cosine)=1.4446
  k=26 | silhouette(cosine)=0.0966 | max_diameter(cosine)=1.4340
  k=27 | silhouette(cosine)=0.0990 | max_diameter(cosine)=1.4893
  k=28 | silhouette(cosine)=0.1018 | max_diameter(cosine)=1.4242
  k=29 | silhouette(cosine)=0.1040 | max_diameter(cosine)=1.4826
  k=30 | silhouette(cosine)=0.1055 | max_diameter(cosine)=1.4826
  k=31 | silhouette(cosine)=0.1067 | max_diameter(cosine)=1.4826
  k=32 | silhouette(cosine)=0.1084 | max_diameter(cosine)=1.4826
  k=33 | silhouette(cosine)=0.1107 | max_diameter(cosine)=1.4893
  k=34 | silhouette(cosine)=0.1097 | max_diameter(cosine)=1.4820
  k=35 | silhouette(cosine)=0.1108 | max_diameter(cosine)=1.4893
  k=36 | silhouette(cosine)=0.1119 | max_diameter(cosine)=1.4893
  k=37 | silhouette(cosine)=0.1145 | max_diameter(cosine)=1.4280
  k=38 | silhouette(cosine)=0.1154 | max_diameter(cosine)=1.4893
  k=39 | silhouette(cosine)=0.1182 | max_diameter(cosine)=1.4522
  k=40 | silhouette(cosine)=0.1191 | max_diameter(cosine)=1.4312
  k=41 | silhouette(cosine)=0.1211 | max_diameter(cosine)=1.4522
  k=42 | silhouette(cosine)=0.1220 | max_diameter(cosine)=1.4522
  k=43 | silhouette(cosine)=0.1232 | max_diameter(cosine)=1.4436
  k=44 | silhouette(cosine)=0.1257 | max_diameter(cosine)=1.4744
  k=45 | silhouette(cosine)=0.1281 | max_diameter(cosine)=1.4115

[Plot] Saved k vs silhouette (cosine) to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v2.3.3_k_vs_silhouette_cosine.png

[Plot] Saved k vs max cluster diameter (cosine) to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v2.3.3_k_vs_max_cluster_diameter_cosine.png

[UMAP] Saved 2D scatter to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_scatter.png
[UMAP] Saved embedding CSV to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_embedding.csv

[HClust] Saved dendrogram to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/hierarchical_dendrogram.png

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_timings.json

=== Runtime (seconds) ===
               load_csv_s:   6.1872
         nlp_preprocess_s: 119.2112
    vectorize_text_tags_s: 114.8873
         scale_features_s:   0.0673
             clustering_s: 1370.7516
     interpret_clusters_s:   0.2030
           save_outputs_s:   0.9507
        metrics_compute_s:   0.5271
                   umap_s:  46.8953
           hierarchical_s:   0.5684
          total_runtime_s: 2043.3094
```

For v2.3.3 I use the same techniques of data preprocessing and NLP as in v1.1.3.

---

**Feature representation**

The feature construction is the same joint representation I used in v2.1.3. On the text side, I embed `short_question_pos` with Sentence-BERT (`all-MiniLM-L6-v2`), yielding 384-dimensional sentence embeddings. I then apply PCA to reduce this to 80 dimensions while still explaining about 72.1% of the BERT variance. On the tag side, I transform `tag_list` with a MultiLabelBinarizer into a sparse multi-hot matrix over tags, and then use TruncatedSVD to get a 20-dimensional tag embedding that explains about 24.5% of the variance in the tag space.

These two blocks are concatenated into a 100-dimensional dense vector per question (80D BERT-PCA plus 20D tag-SVD). I apply StandardScaler and then row-wise L2 normalization to produce `X_unit`, which is the main feature matrix used for clustering and cosine-based diagnostics. Intuitively, the BERT part captures paraphrase-level semantic similarity between the content words, while the tag part adds a soft supervision signal so that questions with similar labels are pulled closer in the latent space. Because the representation is identical to v2.1.3, any differences in behavior are driven entirely by the change in clustering algorithm, not by the embedding itself.

---

**Clustering model**

The main clustering model in v2.3.3 is OPTICS, used as a density-based method to find only the most coherent regions of the space. With approximately 47,491 points, I set `min_cluster_size` to the maximum of 189 and 0.004 times the corpus size, which ends up being 189. I use `min_samples = 20`, `xi = 0.03`, `metric = "cosine"`, and `cluster_method = "xi"`. Under this configuration OPTICS returns four clusters (labels 0 to 3) and marks 46,650 points as noise, meaning roughly 98.4% of the dataset is considered background. In other words, OPTICS in this setup is deliberately acting as a dense-core detector: only very tight, high-density islands are labeled as clusters, and everything else is left unassigned.

In addition to the OPTICS model, I also run a KMeans k-scan on the same `X_unit` purely for diagnostics. Using MiniBatchKMeans for k between 10 and 45, I observe that the cosine silhouette increases from about 0.054 at k = 10 to around 0.128 at k = 45, while the maximum cluster diameter (cosine distance) fluctuates between roughly 1.42 and 1.51 with a mild improvement at higher k. That picture is almost identical to what I saw in v2.1.3: if I force a full partition of the space, the 40-45 clusters regime still looks like the most structurally reasonable region. Conceptually, the OPTICS solution is saying "label only the densest disease islands," while the KMeans scan says "if a complete taxonomy is needed, something like k ≈ 45 is still the right order of magnitude."

---

**What OPTICS actually finds**

In practice, OPTICS discovers four extremely tight, disease-focused clusters and treats the rest of the corpus as noise. The first cluster corresponds to shingles, chickenpox, and varicella-zoster style topics: its top terms include "shingle", "pox", "vaccine", "chicken", "breakout", and "chickenpox", and its dominant tags revolve around shingles, chickenpox, and vaccination. The second cluster is clearly a hepatitis and liver disease island, with terms like "hepatitis", "liver", "virus", "infection", and "cirrhosis" and tags centered on hepatitis B, hepatitis C, and liver-related conditions.

A third cluster represents an endocrine and bone health theme, mixing thyroid disorders and osteoporosis risk. Top terms here include "hypothyroidism", "hyperthyroidism", "osteoporosis", "tsh", and "pregnancy", and the tag distribution reflects osteoporosis, thyroid conditions, and pregnancy-related contexts. The fourth cluster is more idiosyncratic: its top terms include "xarelto", "xdr", "xolair", "xray", "yaz", "yeast", "wrist", and "wrinkle". This group looks like a very tight but somewhat semantic-odd bucket of questions that share "x-words" and rare medical terms; internally it is extremely consistent in TF-IDF space even if its human label is less obvious.

The UMAP visualization reflects this pattern perfectly. Almost all points belong to a single, large background cloud colored as noise (label -1), while a few small, distinct islands appear in different colors for the four OPTICS clusters. A hierarchical dendrogram on the four cluster means in cosine space shows that the centroids are all roughly equally distant from one another. With only four centroids there is no deep tree structure; they form four well-separated islands with no obvious sub-hierarchy.

---

**Metrics and measurement in cosine space**

All metrics in v2.3.3 are computed on labeled points only, excluding the noise. Using cosine distances on `X_unit`, the global silhouette score for the four OPTICS clusters is about 0.6733, which is very high compared to the KMeans runs. The cluster size statistics show that labeled clusters are quite balanced: the minimum size is 191 points, the maximum is 234, and the mean is around 210 with a small standard deviation. This is what I would expect if OPTICS is carving out several similarly dense cores of comparable size.

Per-cluster mean silhouettes further confirm the density of these islands. The shingles cluster has a mean silhouette around 0.62, the hepatitis cluster around 0.71, the thyroid/osteoporosis cluster about 0.42, and the "X-bucket" cluster an almost extreme 0.996. That last value indicates that points in cluster 3 are almost indistinguishable from each other in this feature space and extremely far from the other clusters; silhouette close to 1.0 is a textbook signature of a very tight, isolated group. On average, the intra-cluster cosine similarity is about 0.7011, which is significantly higher than in the v2.1.3 KMeans solution, while the mean inter-centroid cosine similarity is approximately -0.0025, showing that cluster centers are nearly orthogonal on average.

Tag-coherence metrics also improve markedly compared to v2.1.3. Mean purity is about 0.450, so roughly 45% of the points in a typical cluster share the most common tag. Mean tag entropy drops to around 2.63, indicating tighter label distributions, and mean intra-cluster Jaccard similarity for tags rises to about 0.239. This tells me that within each OPTICS cluster, not only are the texts highly similar, but the tag sets overlap much more strongly than they do in the broader KMeans clusters. In other words, these clusters are both semantically and label-wise very coherent, which is exactly what I want from dense-core discovery.

---

**Runtime and performance**

From a runtime perspective, v2.3.3 is substantially heavier than v2.1.3. The preprocessing steps are comparable: spaCy-based POS filtering takes around 115 seconds, and BERT embeddings plus PCA/SVD take about 99 seconds. The big difference lies in clustering. The OPTICS clustering stage consumes roughly 1,332 seconds, whereas the KMeans model-selection sweep in v2.1.3 took about 419 seconds. UMAP adds another ~31 seconds, and the total runtime of v2.3.3 ends up near 1,957 seconds, or about 32.6 minutes.

That means this pipeline is around three times slower than the v2.1.3 KMeans version. The cost comes from two sides: the OPTICS algorithm itself has near-quadratic behavior in practice on 47k points with cosine, and I still run a full KMeans k-scan for diagnostic plots and comparisons. In practice, this makes v2.3.3 more of a heavy exploratory model rather than something I would rerun constantly in a production loop.

---

** How I'd summarize v2.3.3's behavior**

Stepping back, the representation in v2.3.3 is identical to v2.1.3: BERT-PCA plus tag-SVD on POS-filtered noun phrases, scaled and normalized for cosine geometry. The major change is in the clustering objective. The v2.1.3 KMeans solution with k = 45 tries to build a complete taxonomy over the entire corpus, forcing every question into one of many moderately coherent topics. The v2.3.3 OPTICS solution, by contrast, is explicitly designed to label only the most unambiguous, high-density disease islands and to leave almost everything else as unclustered background.

The four clusters that emerge are exactly what this design suggests: a shingles/chickenpox vaccination island, a hepatitis and liver disease island, a thyroid/osteoporosis endocrine-risk island, and a very tight idiosyncratic "X-token" bucket. They have excellent silhouette, high intra-cluster similarity, and strong tag coherence. From a downstream perspective, that makes v2.3.3 very attractive if I want high-confidence anchor topics—places where I can safely assign clear medical labels and maybe use them as seeds or distant supervision for later supervised models.

If I instead need a full routing taxonomy where every question must be assigned to some topic, v2.1.3 remains more appropriate: its 45-cluster KMeans partition covers the full space, even though its global silhouette is lower and some clusters are fuzzy. In short, v2.1.3 gives me a broad, somewhat noisy map of the entire corpus, while v2.3.3 acts like a microscope that highlights only a handful of very dense, very clean disease-specific groups and treats all remaining questions as background.

## v2.4.3

```
Rows after POS cleanup: 47491
[Features] BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245) = 100D
[Shapes] Combined feature matrix: (N=47491, D=100)
[Spectral-Nystrom] N=47491, m=2000, r=40, TOP_T=64 → Z shape (47491, 40), time 3.68s
Saved k-scan table to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_model_selection_kscan.csv

[Plot] Saved k vs silhouette (cosine) to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v2.4.2_k_vs_silhouette_cosine.png

[Plot] Saved k vs max cluster diameter (cosine) to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/v2.4.2_k_vs_max_cluster_diameter_cosine.png
[Model-Select] Chosen k=43 (sampled silhouette_cosine=0.312)

=== Cluster Summaries (Spectral, Nyström) ===

Cluster 0  (n=892)
 Top terms: medication, drug, medicine, treatment, counter, med, prescription, effect, blood, allergy, pressure, xdr
 Top tags: drug, over the counter, pregnancy, coldness, ibuprofen, pain, drug test, tuberculosis, vision, smoking
 Examples:
  - i have been thinking about buying drugs online to save money is that a good idea
  - i am extremely agoraphobic despite help from my psychologist psychiatrist and medication why else can i do
  - is avinza a narcotic medication
  - does soma show up on a drug urinalysis
  - what medications and treatments are available to treat pagets disease of bone

Cluster 1  (n=1548)
 Top terms: weight, food, diet, sugar, pound, water, loss, blood, drink, calorie, meal, level
 Top tags: food, diet, drinking, weight, smoking, exercise, weight loss, diabetes, blood sugar, meal
 Examples:
  - what weighs more muscle or fat  im just wondering about weight gain due to muscle growth my wife has been working out for some time with weights and cardio training but she is finding that her weight has been fluctuating and at times gains weight a little bit
  - can you be allergic to mold in your food
  - i have started a low sugar and low wheat diet and i keep going to the toilet more than normal is my diet the reason
  - had total knee replacements i am not feeling good no energy depressed no appetite have lost weight
  - what can i do to gain back my missing pounds and feel healthy again  i have been sick and lost 17 pounds i am fatigued all the time and look poorly i want to gain my wieght back and feel good again as quickly as possible

Cluster 2  (n=4012)
 Top terms: overdose, sleep, pain, hernia, body, muscle, pill, medication, hair, blood, loss, head
 Top tags: drug overdose, hernia, hair loss, pregnancy, allergy, wart, head, over the counter, asthma, hysterectomy
 Examples:
  - can i transmit genital warts seventeen years after having them removed
  - does prozac cause weight gain what about zoloft
  - what is black measles when i was young i had them now in my fifties i have a lot of health problems could it be because of them and what damage do they do to your body i know they have not been heard of in god know how long is there any way to know after all of these years after having them to get information on them
  - is erythema multiforme an autoimmune disorder
  - i have been taking propranolol for the chest pains now have headaches and pain on left side of head and body

Cluster 3  (n=1937)
 Top terms: stomach, diarrhea, pain, colitis, symptom, disease, diabete, reflux, bowel, constipation, risk, stool
 Top tags: diarrhea, heartburn, stomach, diabetes, bowel movement, ulcerative colitis, constipation, "crohns disease", prostatitis, type 1 diabetes
 Examples:
  - i need relief from chronic epididymitis
  - why does coffee give me such an energy boost
  - i have an acute dextroscoliosis i feel pain when i skip meals
  - my butthole hurts and there is a bump and it hurts whenever i sit or move i assume it happened a few days ago when i was trying to hold in my bowel movement but i could not so i went to the bathroom but i do not remember anything happening to me for the rest of the day or the next but on sun today8th 9th its been bothering me severely and it hurts whenever in doing anything that requires me to move my lower body
  - how can i get rid of dark spots on my face

Cluster 4  (n=3132)
 Top terms: hepatitis, husband, problem, nose, woman, hand, virus, treatment, sinus, liver, person, wart
 Top tags: smoking, virus, pregnancy, hepatitis c, wart, nose, hepatitis, acne, hiv, bedbug
 Examples:
  - can you test positive from having the hep b vaccine
  - okay so i am 16 and i want to grow about 3 more inches if i smoke hookah once or twice will i grow to my goal height
  - can herpes be spread by bed bugs if a person infected with herpes is bitten by a bed bug can another person bitten by the same bug get infected with herpes
  - my husband is taking 40 mg of prozac and is really depressed and has thought of suicide what do we do
  - i am experiencing a problem keeping an erection are there natural remedies that can be taken for this

Cluster 5  (n=1533)
 Top terms: period, sex, pregnancy, test, pill, birth, control, cycle, symptom, breast, blood, chance
 Top tags: period, pregnancy, pregnancy test, birth control, spotting, breast, ovulation, condom, cramps, ejaculation
 Examples:
  - my period only last 36 48 hours which is my norm is that why i have had 2 yrs of no luck getting pregnant  my husband and i have been trying for two years to have a child i am turning 30 next month and in my family after 30 equals issues my normal period is only 36 48 hours could this be preventing me from getting pregnant
  - spotting on day two of my period could i be pregnant i have had my loop taken out 71015 and have had unprotected sex a few times after that hoping to fall pregnant my period was meant to start 12102015 it now day two on my period and i have only been spotting which is very un usual could i be pregnant
  - pregnant unprotected sex a week before period period came on time and heavy with bad cramps as usual reg 28 day 4 yrs i had unprotected sex a week before my period started he ejaculated awa from me but im worried a little bit may have got it before he pulled out my period came on the dot when it was supposed to get it and was heavy at first then to moderate with bad cramps like i normally have basically my period came on time and was normal in length flow and cramps my periods have been regular for years i do not know when i ovulate or my latueal phase what are my chances of being pregnant
  - my period has been late by 4 days i am trying to conceive please help
  - i have been off depo shot for over a year and have no period i recently began having pregnancy symptoms am i pregnant  i was on the depo shot for 2 years and have not gotten the shot since september 2011 i have still not had a period and have been having pregnancy symptoms my fiance and i use the withdraw method i realize that you can still get pregnant this way after over a year of not receiving the shot i do not think the symptoms could be attributed to the shot i had a pap smear 9 12 12 and it came back fine so could i be pregnant and if not what could be causing the pregnancy symptoms

Cluster 6  (n=693)
 Top terms: anxiety, depression, attack, panic, disorder, medication, symptom, pain, xanax, heart, help, feeling
 Top tags: anxiety, depression, panic attack, pain, stress, fear, chest, heart, vision, drug
 Examples:
  - pain when urinating inconsistent urination painfull ejaculation painfull mastrubation etc oh gosh im in all sort of trouble here and its given me anxiety over the past 1 5years ive been having this condition it all began when my urine penis started smelling cheesy after urination then later on when i was laying in bed and rising up i could feel from pelvic area like something is almost pushing my urine out it happened all the time then came premature ejaculation painfull urination painfull mastrubation painfull ejaculation also when i drag back my foreskin pain help please
  - should medical marijuana be used to treat anxiety disorder
  - my primary doctor diagnosed me with anxiety disorder and prescribed xanax and zoloft why do i still have panic attacks
  - what is bipolar disorder and why is it sometimes called manic depression
  - can anxiety a side affect to depression

Cluster 7  (n=1158)
 Top terms: pill, birth, control, sex, woman, fertility, infertility, pregnancy, chance, condom, abortion, miscarriage
 Top tags: pregnancy, birth control, birth control pill, condom, infertility, fertility, ejaculation, sperm, ovary, ovulation
 Examples:
  - what are some warning signs for pregnant women when they are exercising
  - how effective are male condoms at birth control
  - i take steroid prednisolone for ivfpregnancy due to auto immune issue but i have herpes 2 will this affect baby
  - i am pregant does everything i wear have to have cotton in it  my boyfriend is sure he read that now that i am pregnant everything i wear has to have at least a percentage of cotton in it i know that my panties should be cotton but i can not find answers about the rest of my wardrobe this is making it difficult to find suitable pants for my work uniform
  - inserting finger into girls vagina leads to pregnancy

Cluster 8  (n=771)
 Top terms: baby, period, pregnancy, sex, husband, birth, daughter, pain, food, tube, test, bottle
 Top tags: baby, pregnancy, period, sexual intercourse, pain, feeding, breastfeed, movement, newborn, smoking
 Examples:
  - can newborn babies be born addicted to prednisone if the mom took it for asthma in the last trimester
  - how much nap time does my baby need
  - is a baby more likely to be colicky with a baby bottle
  - how to manage large kidney stones that can not be passed on its own while being 16 weeks pregnant and not harm the baby stones are roughly 5 8 mm in size recurrent episodes of pain over the past 2 3 weeks should i stent how safe are the pain meds percocet and dilaudid for the baby
  - is it possible that a baby could be conceived on august 29 2011 and deliver june 27 2012

Cluster 9  (n=1372)
 Top terms: insurance, health, plan, medicare, income, coverage, care, exchange, marketplace, medicaid, employer, state
 Top tags: insurance, health insurance, medicare, affordable care act, medicaid, health insurance exchange, health insurance marketplace, obamacare, family, pre-existing condition
 Examples:
  - i am a disabled veteran on medicare am i affected by the affordable care act
  - i manage a medical office with 3 employees  rather than offer a health insurance plan we pay 50 of the employees premium so if they purchase their insurance through the marketplace will we no longer be able to do that
  - will they accept obamacare at any hospital
  - i am 50 years old and am currently on medicaressdi i have part a b and d  i also have aarp hospital indemnity by united healthcare i also have an aarp rx drug plan is this enough to cover what is needed under the affordable care act
  - if you already have a non  employer based health plan can you still go to the exchange  can you see if you can get a better plan at a lower rate

Cluster 10  (n=1066)
 Top terms: sex, period, condom, pill, intercourse, pregnancy, boyfriend, chance, birth, test, control, partner
 Top tags: sexual intercourse, pregnancy, period, condom, ejaculation, ovulation, birth control, vagina, penis, pregnancy test
 Examples:
  - could i possibly be pregnant  last period may 16th unprotected sex on june 9th supposed to start june 16th still have not if you think i am pregnant when should i take a test  side notes  feel as if i start but do not   i was throwing up at 2 am on saturday the 15th i was nauseous the rest of the day   light cramps  the guy i had sex with says he only has a 3 chance of getting someone pregnant  i have been tired lately i also have been having light heartburn i think  if anyone can help me it would be greatly appreciated
  - i am on the pill and a condom was used pregnant  i have been on the pill for over 5 years i am pretty good with taking it on time but occasionally i will forget a day but immediately take it when i realize ive missed it last weekend saturday i was about an hour late taking my bc the next day i had sex he was wearing a condom a week later getting cramps and what not like a period i usually take the pills continuously but i am scared so i am going to let myself have a period hopefully i am about to start the sugar pills today
  - my boyfriend and i had sex using both the pill and a condom the condom broke should i worry about rubber inside me  we think most stayed on his shaft and i found a small piece about 34 inch x 12 inch like it ripped and a portion tore off do i have to worry about this causing tss 
  - had sex a week later instead of my normal period had alittle blood when wiped now im sick to stomach and nipples itch could i be pregnant
  - my husband and i had sex in a pool yesterday i am 7 weeks pregnant and now i am spotting and cramping is this normal

Cluster 11  (n=1533)
 Top terms: infection, pneumonia, throat, antibiotic, chlamydia, strep, sinus, cough, penicillin, yeast, amoxicillin, bronchitis
 Top tags: antibiotic, pneumonia, chlamydia, sinus infection, throat, penicillin, bacterium, cough, amoxicillin, sore throat
 Examples:
  - can taking multiple antibiotics cause redness and dryness of vagina
  - i have had a pneumonia shot can i get either a sinus infection or walking pneumonia from my 6 year old grand daughter  and can i be a carrier to others in my age group
  - how soon should my 14 year old wait before returning to school having been diagnosed with pneumonia
  - is clindamycin effective in treating syphilis
  - why it is necessary to take 2 antibiotics for a diverticuilitis infection will taking just the cipro work

Cluster 12  (n=18)
 Top terms: vitamin, cold, supplement, fridge, system, seed, food, effectiveness, child, fat, degree, control
 Top tags: coldness, vitamin, cold, immune system, food, flaxseed, flax, birth control pill, fried chicken, spasm
 Examples:
  - do any supplements like cold milled golden flax seed affect the effectiveness of birth control pills
  - is it ok to give children vitamins or supplements when they have a cold
  - is it okay to double your vitamins when you are sick  i have a cold and my doctor said it is okay to double my vitamins at times i want to know if i can and will it help
  - does ms weaken the immune system  when i was initially diagnosed as having ms   40 years ago   i was told to be very careful if i got minor illnesses such as colds and that it would be easier for me to get infections  has the enormous amount of subsequent research shown this to be true  i have relapsing remitting ms which slows me down and makes me tired wobbly i only use vitamin d modafinil to give me more energy and clonazepam to help with the spasms that are difficult at night thank you in advance for your answer
  - is there a way to eliminate or fade lipmustache scars from cold sores  i had a cold sore 5 weeks ago that has since healed but left behind a dark discoloration on the upper lip and lower mustache area is there a way to ge rid of or fade the scars

Cluster 13  (n=805)
 Top terms: pregnancy, trimester, test, symptom, sign, woman, miscarriage, risk, weight, breast, problem, chance
 Top tags: pregnancy, pregnancy test, miscarriage, ejaculation, stress, urination, first trimester, breast, drinking, fear
 Examples:
  - how effective are foam and male condoms in preventing pregnancy
  - why do my breasts hurt and feel and look like they are getting bigger  for almsot 2 weeks now my breasts have been painful and feel like and look like they are getting bigger i instantly thought of being pregnant but i am on the mirena birth control and although its not 100 effective i still did not believe i were pregnant i took a home pregnancy test from my local drugstore and it was negative why are my breasts feeling this way people i know have been commenting on them looking huge or bigger than normal
  - how to confirm my pregnancy
  - pregnancy tests were negative but i passed blood clots what could it be
  - can a uti mask a pregnancy  i am having all the signs of a pregnancy and i took a pregnancy test and it came back negative and i have a uti can a uti mask a pregnancy  \

Cluster 14  (n=726)
 Top terms: rash, hand, arm, bump, body, penis, leg, area, thigh, pain, ivy, cream
 Top tags: rash, arm, hand, penis, skin, thigh, leg, itch, swelling, wrist
 Examples:
  - can an antibiotic through an iv give you a rash a couple days later
  - is hand foot and mouth the same as rubella is hand foot nd mouth the same as rubella
  - i have a rash under my armpit what kind of treatment should i use a group of 7 red bumps slightly raised and itchy just really trying to figure out how i can start the healing process and get them to go away thanks
  - can poison ivy rash return same place after 2 months  rash returned same places on arms after 2 months since had first rash  have not been any where near any plants no pets to come in contact with poison ivy any one ever hear of this happening
  - is there such a thing as a weather allergy  for the past three years in the autumn i will start getting itchy red scaly rash from wrist to elbow it will last from september to usually april may it remains consistent through these months i remain flare free in the summer months i have asked my doctor countless times and he sent me home with a very strong hydrocortisone cream i have resorted to over the counter hydrocortisone which burns i can not use uv therapy as i am allergic to the sun and i am currently on an antihistamine for that

Cluster 15  (n=1109)
 Top terms: exercise, weight, muscle, diet, pain, minute, type, workout, heart, blood, program, fat
 Top tags: exercise, weight, diet, walking, muscle, pain, workout, running, diabetes, knee
 Examples:
  - do doctors immediately prescribe medication to control blood sugar or do they wait to see if diet and exercise help
  - what exercise does not aggravate achilles tendonitis  i am trying to rest my achilles so i can return to playing tennis in the meantime i want to exercise to stay in shape what types of exercise can i do
  - what is resistance exercise
  - what is a weight loss exercise for people with arthritis in feet ankles and disc herniation in neck and low back  swimming irritates the neck and pain shoots down my arms walking in good sneakers pains my feet and ankles which have arthritis i can stretch but that is not enough to lose weight
  - what specialists deals in muscles tendons and ligaments dr says my muscle cant be healed only exercise and meds  i would like to see a pain specialist who would that be i have a stretched muscle upper back shoulder right arm muscle spasms severe at times do i need surgery for this muscle to heal

Cluster 16  (n=976)
 Top terms: stomach, nausea, pain, period, diarrhea, headache, symptom, cramp, side, feeling, vomiting, doctor
 Top tags: nausea, stomach, pain, pregnancy, diarrhea, vomit, headache, period, dizziness, cramps
 Examples:
  - my husband has had a fever and a cough for over 10days now not pneumonia had xray 4 days ago what could it be  his fever ranges between 100 103 he is coughing up lots of mucous but does not feel any sinus congestion no runny nose he has been on an antibiotic for 5days now he says it is helping a little but it is not taking care of the fever or cough just making the mucous less green he had a chest x ray 5days ago but it did not show pnumonia should he be x rayed again he is also very pale in color as well as is experiencing dizziness headache and nausea had a cbc blood work came back normal
  - can u get pregnant on the pill  i have been taking the pill for 2 months now and my last peroid wasnt really a period i bled for maybe 5 minutes a day for 3 days could i be pregnant i just started feeling nausea and im hungry but when i start eating i feel like i cant eat
  - pain in upper right quadrant of abdomen may indicate what 18 year old grand daughter had gallbladder stmptoms but no stones doctor said bile was crystalizing and bladder was removed still having problems food test showed food slow at passing from stomach to intestines smaller portions have not helped couple crackers may induce vomiting what are possible causes
  - could clonazepam be causing nausea and dizziness or is it bupropion
  - stabbing pain under rib cage stabbing pains right side under my ribs for several weeks started in july ct scan normal i eat healthy now it is accompanied with awful nausea and vomiting the nausea is constant i have vomited 20 times in 3 weeks not pregnant i have been to so many doctors and they are really not sure what is wrong all of the tests have came back normal ct scan with contrast blood work xrays ultrasound of gallbladder and hida scan for the hida scan gallbladder and small bowels normal and ef 68

Cluster 17  (n=1067)
 Top terms: skin, acne, face, product, bump, spot, area, rash, patch, problem, pimple, help
 Top tags: skin, acne, rash, burn, vision, itch, penis, arm, hand, lump
 Examples:
  - guest in my home has scabies do i have house sterilized or will a good cleaning do it we have not had skin contact he is getting treated and i am having a general cleaner come in this afternoon to change linens etc is this sufficient i have never had anything like this in my home somewhat disturbed
  - what is the best moisturizer for older skin
  - what medicines can cause your skin to turn blue
  - the most effective skin care  i am for normal skincare but do not take my word for it try some out yourself study customer opinions and make your brain up next the most effective businesses may have a no risk assure you can send total or applied containers back and obtain a complete refund if you are not 100 pleased skin care>> <link>
  - my skin is irritated after i treated a flea infestation in my hair with kerosene how can i reduce skin irritation

Cluster 18  (n=445)
 Top terms: flu, shot, vaccine, influenza, swine, child, fluoride, symptom, tamiflu, adult, type, vaccination
 Top tags: flu, injection, swine flu, vaccination, vaccines, pregnancy, virus, coldness, blood pressure, family
 Examples:
  - why would a rn choose not to get her kids a flu shot as the grandparent is there anything i can do
  - does my water filter take out the fluoride out of the water
  - what happens to someone when they get influenza
  - can getting the flu shot raise blood pressure  i have been monitoring my blood pressure for about 3 weeks now and the readings have been normal 11075 but i noticed ever since i received my flu shot from a few days ago my blood pressure has been reading higher 140 150 range85 is this a side effect of the flu shot
  - can you get a flu shot and the whooping cough vaccine at the same time or does it cause problems  we have had several patients tell us they have heard not to get the 2 shots together   but we have never heard that & our health department says it is fine

Cluster 19  (n=822)
 Top terms: fever, pain, headache, throat, body, symptom, cough, son, grade, chill, skin, nose
 Top tags: fever, pain, running, headache, cough, antibiotic, skin, stomach, swelling, sweating
 Examples:
  - i have persistent headache and i feel like i have lowgrade fever help  hi so i am a headache everyday it is not too bad though it is completely bearable but a little distracting and i have noticed that i have lowgrade fever most of the time or mostly everyday but just like the headache it is bearable i can not just shrug this feeling off this have been occurring for two or three months now i am hopefully going to the doctor in a few days and get myself checked
  - why do i feel lightheaded fatigued and sweat during sleeping no fever  i am an almost 37 yr old female with a lot of stress right now dr put me on effexor and i started not being able to sleep having bad headaches feeling lightheaded and constipated i took it for 2 weeks and he told me to stop when i called him he called me in something else but i am afraid to get it i have been on paxil prozac and celexa and never felt this horrible i have not taken anything in almost a week but feel lightheaded many times thoughout the day any ideas what could be wrong
  - can i have strep without fever  my 4 year old son was diagnosed with strep throat 3 days ago last night i was fine one minute and suddenly felt like i would been hit by a ton of bricks body aches headache  sore throat and general feeling crappy but no fever is there any point in dragging myself out to doctor when i feel so miserable is it possible to have strep without fever i do not have any runny nose stuffy nose or cough not a cold 
  - how long should i wait before bringing my 11 yr old with flu symptoms to our family dr it has been 8 days initial symptoms were nausea high fever severe headache loss of appetite and fatigue those lasted about 2 days now she is very tired little appetite sore bellynausea and has a sore throat and cough
  - my son is 7 and can swallow pills he is running a fever can i give him one 200 mg ibuprofen pill he weighs 53 pounds

Cluster 20  (n=715)
 Top terms: shot, injection, vaccine, depo, steroid, provera, chickenpox, tetanus, insulin, period, flu, testosterone
 Top tags: injection, pregnancy, vaccines, period, flu, steroid, antibiotic, depo-provera, pain, sexual intercourse
 Examples:
  - was skiing and fell on my knee cap now 3 days later i just heard a pop and its throbbing and its severe pain  they did an xray but they did not have an mri machine available i will not be able to access a doctor for another 3 9 days a half hour the pop happened and it really hurts on a scale of 1 10 10 being like you have just been shot its more of an 8 7
  - can allergy shots be used to treat asthma
  - treatment for household cleaning product allergies symptoms are fatigue shakiness nasalchest burning weakness and asthmatic reaction or shortness of breath already getting immunology shots for seasonal dust mites cat dander and mold  does not appear to help the cleaning product symptoms
  - i had a dtap in 2011 and i stepped on a nail today do i need to get a tetanus shot  im 16
  - suffering from sesamoid pain podiatrist says no multiple cortisone shots

Cluster 21  (n=567)
 Top terms: tooth, mouth, dentist, denture, gum, brace, toothpaste, wisdom, root, canal, problem, pain
 Top tags: tooth, mouth, dental, wisdom tooth, pain, cavity, jaw, insurance, problems with tooth, tongue
 Examples:
  - would braces close wide gap between front teeth
  - i brush my teeth regularly but i am not a big time flosser what are some tips for making daily flossing simple & fast
  - i heard that fluoride can damage teeth if so what are the alternatives for toothpaste
  - abscessed tooth symptoms and oral cancer symptoms similar
  - i brush my teeth twice daily even then my teeth are yellow what to do

Cluster 22  (n=1089)
 Top terms: breast, lump, side, pain, shoulder, chest, arm, armpit, mammogram, neck, nipple, back
 Top tags: breast, lump, nipple, pregnancy, mammogram, swelling, breast cancer, atopic dermatitis, arm, shoulder
 Examples:
  - broken collarbone 3 5 cm overlap its been three weeks after break and still feels broken or loose
  - i was diagnosed with mononucleosis 7 months ago since then i have not been able to recover
  - i was diagnosed with mononucleosis 7 months ago since then i have not been able to recover
  - i have got a wrist and palm injury in my right hand
  - hi my husband has a lump on his head that has been there for 2 weeks and now has a lump under his armpit hi my husband has a lump on the back of his head for about 2 weeks now and it appeared there without injury we were a little concerned but now i am really concerned because he has another lump under his armpit that has been there for 2days

Cluster 23  (n=601)
 Top terms: pain, discharge, vagina, burn, burning, sensation, itching, infection, penis, sex, yeast, lip
 Top tags: burn, pain, vagina, urination, stomach, penis, vision, yeast infection, itch, swelling
 Examples:
  - will my glans burn recover  accidentally i got a very hot water on my penis the water hit a small area of glans and the area under it now rubbing it either by hand or cloths has some sort of annoying feeling i would like to know whether recovery is possible or not if yes please tell me how
  - what other than a yeast infection could it be if the medicine does not work and more symptoms start occuring  for a few months now i have had constant vaginal itching and burning and after using yeast infection medicine multiple times it still has not gone away and now blisters have begun forming after looking up genital herpes i have noticed that i have nearly all the symptoms for it but i have yet to have sexual activity could it still be possible or is it something else
  - can you use egg whites on a burn  i read an article that said you can use egg whites to sooth and help heal burns like if you burn yourself with fire but not real bad is this true
  - for hand and mouth disease can the sores be on tongue
  - heart attack i am a parapalegic and the other night when i was going to bed i had a burning sensation that started in the chest area and moved to the back to the point my upper torso was burnig all the way around i was in total discomfort i felt a heaviness in my chest this lasted for about 3 hours are these symptoms of a heart attack or are these symptoms that have to do with my spinal cord injury i have had these symptoms before but never to the degree i had the other night

Cluster 24  (n=411)
 Top terms: hypothyroidism, thyroid, osteoporosis, hyperthyroidism, synthroid, nodule, medicine, level, risk, gland, symptom, condition
 Top tags: hypothyroidism, osteoporosis, thyroid, hyperthyroidism, tsh levels and pregnancy, surgery or nuclear medicine better, thyroid nodule benign or malignant, diet, i have had a thyroid nodule since diagnosed, drug overdose
 Examples:
  - what should i do if i suspect an overdose of thyroid
  - what causes peyronie is disease
  - i have all the classic signs of hypothyroidism but lab results normal where to go from here
  - im 27 and ive had almost all of the symptoms since i was about 6 years old could it be possible i have hypothyroidism
  - what would make my tsh levels suddenly jump to critically high level when i feel the same  i take synthroid 1 25 daily and yet my tsh is now 28 3 increased from 23 on friday why would it jump so quickly my dr is on vacation and i just want to make sure this is ok to wait until thursday when he comes back 

Cluster 25  (n=351)
 Top terms: baby, son, boy, tsh, point, help, trouble, stuff, cause, man, count, beat
 Top tags: pregnancy, exercise, coldness, ovulation, cold, epilepsy, fertility, body rash, malaria, i have been feeling very depressed
 Examples:
  - is leucoderma curable
  - i am hardly eating anything and when i do i feel nauseous
  - could i be pregnant please help
  - what is arthropathy
  - can you get pregnant if you take adderall

Cluster 26  (n=834)
 Top terms: cold, cough, throat, symptom, nose, water, weather, doctor, sinus, infection, chest, eye
 Top tags: coldness, cold, cough, throat, chest, drinking, nose, pain, head, sore throat
 Examples:
  - my baby ate her on poop my baby ate poop 4 days later she is sick weezing coughing and high fever for 4days straight i took her to the doctor and they said shes fine just a normal cold i told them what happen and they just said she should be fine but if she still has a fever next week come back what should i do and is her symptoms related to her eating her poop
  - i smoked cigs for 1 month averaging about 3 a day just wondering if any irreversible was done i did quit since then i started smoking for a month after a period of depression a couple of months ago 2 3 cigs most days with a couple more on bad days i would estimate i probably had 4 packs total over the period i went cold turkey as i started to get my life together and hated the ill feeling from them i exercise regularly and eat healthy and i am still young i would just like to clear my head and hear that i did no damage permanent to my lungs i know it takes a bit to recover hopefully to 100
  - what is the treatment for the common cold
  - how long can flucold causing bacteria live outside the human body  pretty much what i asked my mom had a nasty stomach bug which i am pretty sure was the flu with all the nasty symptoms that go along with it not thinking she used my computer for something to do i have been avoiding it using a friend is computer now but how long do i need to i have stuff i need to do on it so how long do i need to wait before i do not need to worry about contracting it myself
  - going on a long flight with a cold need to sleep sudafed with xanax or sudafed with nyquil  i have a cold and i am going on a very long flight i need to sleep would i be better off taking sudafed with xanax or sudafed with nyquil

Cluster 27  (n=1279)
 Top terms: daughter, child, son, kid, age, girl, mother, school, pain, doctor, blood, boy
 Top tags: pregnancy, depression, ibuprofen, bipolar disorder, type 1 diabetes, my daughter cries, corpulence, vision, height, family
 Examples:
  - my mom is in a depression…what can i do
  - 3 yr old son has small specks of blood on face after napping
  - could my son have future problems or would he been okay  my 2 year old was around w|our puppy he slide and the stick jabbed him in his privates w|stick  he fell to the floor screamed and gasping for air but he took it very hard i am a little worried because this morning i noticed a blackred ring on his private and pulled it down& it was bigger kinda looked like a bloodbruise he does not complain about it but i just want to know if he brokeripped something&if it will effect him in the future needing 2nd opinion ty  worried toddler mother *
  - my son is 8 months old and coughs all the time and has a hard tummy
  - can children take sudafed and motrin at the same time

Cluster 28  (n=711)
 Top terms: heart, chest, failure, pain, disease, heartburn, rate, risk, attack, tachycardia, palpitation, problem
 Top tags: heart disease, congestive heart failure, heart, heartburn, heart attack, high blood pressure, artery, chest, cholesterol, exercise stress test
 Examples:
  - ekg says there was “moderate right axis deviation ” “normal sinus rhythm with marked sinus arrythmia ”
  - is there evidence that statins increase life expectancy for people without heart disease
  - would lad lesion cause tachycardia
  - my mother is in extreme pain for a detached rotator cuff but suffers from heart problems and taking medication
  - what factors place women at high risk for heart disease

Cluster 29  (n=591)
 Top terms: ear, infection, hearing, doctor, pain, fluid, head, eardrum, pressure, wax, sound, throat
 Top tags: ears, ear infection, tinnitus, pain, head, pressure, antibiotic, vision, neck, swelling
 Examples:
  - why am i hearing my heartbeat in my right ear  just recently i have started hearing my heartbeat in my right ear this came on suddenly i am a 66 year old female with no particular health issues what could be the cause of this anything to worry about
  - i have had white noise with corresponding hearing loss in my left ear for 2 months what causes it and can it be fixed i have had an mri    negative for tumor ms or anything else that might be causing it
  - i am having ruptured eardrum one doctor suggested to have surgery and other suggested to wait for 3 months
  - my brother has been losing his hearing abilities since he was 6 7 years old
  - what is wrong with my ear very concerned would just really love some direction for as long as i can remember i have these sort of episodes where my hearing goes out in my right ear and a horrible loud ringing ensues for a few minutes now i only occasionally get it and it is more of an almost constant pulse  yesterday after a violent cry session i lost my aunt i had a sudden headache and felt and heard the pulse not only in my ear but also in the top back of my head it lasted maybe 10 seconds after it stopped i felt something fall out it was a slightly bloody scab

Cluster 30  (n=669)
 Top terms: arthritis, osteoarthritis, knee, pain, hip, lupus, arm, bone, symptom, treatment, disease, problem
 Top tags: arthritis, osteoarthritis, psoriatic arthritis, knee, lupus, rheumatoid arthritis, do i have lupus, knee pain, leg, exercise
 Examples:
  - i have had 2 knee operations knee feels like loose
  - my mom has one hip is a little higher up than the other and it hurts
  - after a bowel movement abdominal pain puts me on my knees severe nausea to the point i do vomit
  - pain in knee hip right leg need key hole surgery
  - bad asthma flare up no signs of mononucleosis white blood cell count was low any ideas

Cluster 31  (n=1107)
 Top terms: pain, side, chest, back, doctor, arthritis, area, leg, knee, neck, shoulder, hip
 Top tags: pain, chest, arthritis, neck, pregnancy, leg, shingles, swelling, shoulder, movement
 Examples:
  - pain ring finger to the middle of arm before elbow for 45 days started after i held on to stop falling worse on lifting its not bad3 on 10 earlier it was more painful but now it does not hurt as much unless i type or write i have barely used my right hand for anything for the past month else the pain increases it starts hurting at one point about five fingers from my wrist but pain goes away completely if i press down on it it also hurts in the area below middle to pinkie i got an xray done already so nothing there shd i get an mri if so only for wrist or forearm also or shd anti inflammatories be enough thanks
  - what are the ingredients inibuprofen  i take a 600mg ibuprofen only as needed for nerve pain my question is what ingredients are in this medication i have a legal prescription for it
  - i have been having very sharp stabbing pains down through the top rtrear of my head the pain almost knocks me down i have been having these pains for 6 7 weeks i have had no previous head injuries they just started out of the blue they are not headaches they are in  a dime sized spot on top of my head right side just off center back portion top of head does that make any sense these pains happen wether  i am standing or laying down thank you for your time
  - does lidocaine cure canker sores on your throat  i found a small white sore on my throat and i was prescribed lidocaine and it numbs the pain but i was wondering if it cures it at the same time before it gets worse
  - can age appropriate arthritis be exaserbated by barometric pressure change or other weather conditions  im 55 very fit and athletic and often suffering with joint pain and stiffness like never before and all over my body not sure where it came from or how to get rid of

Cluster 32  (n=3906)
 Top terms: shingle, hair, vitamin, lip, gallstone, problem, anemia, blood, scalp, mouth, face, oil
 Top tags: shingles, hair, pregnancy, gallstone, vitamin, hair loss, lip, chickenpox, anemia, blister
 Examples:
  - is all vitamin d the same
  - is it better for a type ii diabetic to eat corn or bread stuffing
  - my son had dtap polio chicken pox and mmr vaccines now can barely move
  - is liposuction covered by insurance
  - i have been feeling extremely exhausted and unable to do basic tasks need advice

Cluster 33  (n=634)
 Top terms: surgery, option, pain, hernia, bypass, cancer, knee, doctor, marijuana, leg, exercise, surgeon
 Top tags: surgery, pain, hernia, exercise, leg, smoking, knee, pregnancy, marijuana, walking
 Examples:
  - how do you know what the best exercise routine is  i have had bariatric bypass surgery in 2010 i went from 340 to 232 and have a lot of access skin i also have fibromyalgia and arthritis that is not able to be controlled at the present time it is my desire to run a mini marathon but i do not even know where to begin on setting myself on the proper program i do not have the money to go to a trainer so need some direction please thank you  vicki
  - can my partner and i have sex with hpv  i am a 21 year old homosexual male my partner and i both have hpv however recently i have had a breakout with anal warts i am scheduled to have surgery over the next 6 months to have the warts removed but will my partner and i be unable to have safe sex during this time or is there a dangerous risk of me infecting him with the warts regardless if he already had the virus
  - the heel of one foot is very sore when i walk on it could this be related to my back surgery  i had back surgery and some times i have shooting pains down the opposite sore foot leg is the sore foot something i need to be concerned about
  - i now am going for robotic surgery had hysterectomy almost a year ago
  - triple bypass completed 7 months ago need my brother be concerned about exercise increasing my heart rate above 110  quit smoking before surgery and have lost weight he is 58 and walking every night for 40 minutes and climbing 50 stairs and a hill up and down he feels fine and does not feel it necessary to stop but someone told him he should not even be mowing a lawn because he has heart disease

Cluster 34  (n=805)
 Top terms: disease, disorder, parkinson, epilepsy, brain, dementia, seizure, schizophrenia, mri, memory, stroke, people
 Top tags: "parkinsons disease", bipolar disorder, epilepsy, dementia, schizophrenia, brain, seizure, celiac disease, obsessive-compulsive disorder, "alzheimers disease"
 Examples:
  - what are the dietary restrictions for celiac disease gluten
  - had a stroke on the brain in 2012 its 2016 i cant get no more than 5 hours of sleep a day
  - where can i go for help for bipolar disorder
  - what is prolopa for parkinson is disease
  - what is a seizure and what is epilepsy

Cluster 35  (n=1020)
 Top terms: doctor, opinion, pain, neck, kind, type, blood, test, solution, dermatologist, diabetes, symptom
 Top tags: vision, pregnancy, pain, insurance, family, coldness, ultrasound, blood test, leg, wart
 Examples:
  - my 12th week scan showed everything is o k but radiologist suggested there is caudal regression syndrome
  - i have or think i have parkinsons disease when should i contact my doctor
  - what kind of doctor do i need to see if my clavicle did not heal properly
  - do you find patients are embarrassed to ask a doctor about sexual concerns
  - had surgery on my tail bone will not heel doctor said was good but it is not

Cluster 36  (n=456)
 Top terms: pressure, blood, hypertension, bp, heart, medication, doctor, rate, med, lisinopril, problem, blocker
 Top tags: high blood pressure, blood pressure, low blood pressure, heart, pressure, pulse, dizziness, corpulence, sweating, family has a history of stroke and heart disease
 Examples:
  - why see a renal specialists for high blood pressure
  - what does blood pressure of 123 over 101 with pul 75 mean
  - i have low blood pressure low heart rate tired and lack energy is it normal
  - why would a healthy 83 year old woman have high blood pressue
  - i am a indian and 21 years old i am a post kidney transplantation patient of six week from now i have blood pressure of 165 to 150 can i do bodybuilding is any any affect of increasing blood pressure by doing exercises

Cluster 37  (n=598)
 Top terms: migraine, headache, pain, head, sensitivity, aura, eye, dizziness, brain, pressure, headachesmigraine, food
 Top tags: migraine, headache, pain, photosensitivity, head, eyes, headaches, neck, ibuprofen, blood pressure
 Examples:
  - can diabetes cause you to have chronic migraines
  - i suffer with headachesmigraines frequently
  - unexplained headaches … does mri show problem
  - is lipitor used to treat migraines
  - how can i take feverfew without worrying about the side effects its going to have on my elavil  my migraines tend to come when i am asleep i tried the petadolax herb butterbur for 5 months only change it made was no migraines before and after my menstrual cycle other than that i still get the typical moderate mild 4 5 migraines a month im on 2 types of medications elavil taken before bedtime and topamax 2x a day 1xam 2xpm when my migraine hits my nuerologist has reccomended i take imitrex along w 4 aleve or maxalt w 4 aleve pls help point me in the rt direction

Cluster 38  (n=1189)
 Top terms: foot, leg, ankle, pain, toe, hand, surgery, athlete, finger, area, numbness, thigh
 Top tags: foot, leg, swelling, ankle, toe, "athletes foot", walking, knee, arthritis, finger
 Examples:
  - can i sit in a sauna and steam room with a broken ankle
  - i have lump in the soft tissue of my left leg about 1 inch above my ankle i have had an ultrasound and a xray and i have been told it is not life threatening but the doctor will not tell une but it is hurting and swelling and i am taking pain killers every day me what it is until he sees me again in june
  - i just can not gain weight have chest pains get sharp nerve pain in arms and legs
  - what kind of doctor treats athlete is foot
  - i started having a burning uncomfortable pain in my right flank area

Cluster 39  (n=1420)
 Top terms: herpe, penis, sex, condom, vagina, vaginosis, partner, ejaculation, sperm, intercourse, std, area
 Top tags: penis, vagina, pregnancy, ejaculation, condom, genital herpes, herpes, bacterial vaginosis, sperm, anus
 Examples:
  - i was curious about anal used mothers sex toy didnt clean it at risk for stds do not think it was used in a while i was curious and i found a vibrator and i used it i put a condom on it but condom broke i got tested for chlamidia and ghonorea both negative do you think i am at risk for hiv or anything else  also i used other sorta home made toys over a year ago and i just got worried i could have done damage to my body have not had any negative symptoms and havnt used them since last year should i be worried everything is normal and during use nothing negative happened like bleeding of anything
  - why will not my penis stay hard when in pregame  i get hard quite easily when around my gf but then all of a sudden when it comes to me taking my jeans off it goes down why  when it does decide to work i really do love sex with her so what is causing this also i can not cum when she tries to give bj
  - can a yeast infection cause pain in the urethra as well as the head of the penis  originially thought to have prostatitis a month ago have been experiencing penis soreness urethra pain some redness gonnohrea and chlamydia negative and 3 weeks of cipro provided no relief now thinking it may be fungal could this be a yeast infection and if not what could it be
  - can battery acid from a vibrator burn my genitals i thought i had a vaginal yeast infection but it turns out my vibrator was leaking battery acid and has burned my vaginal area included and up to my rectum what can i do to ease my pain
  - what is a condom

Cluster 40  (n=819)
 Top terms: cancer, prostate, tumor, colon, treatment, radiation, brain, breast, symptom, chemotherapy, therapy, lung
 Top tags: prostate cancer, cancer, colon cancer, breast cancer, brain tumor, chemotherapy, lung cancer, radiation surgery, prostate, liver
 Examples:
  - how to treat leiomyosarcoma and rectal cancer at the same time
  - i had ovarian cancer and reflux surgery i still deal with constant nausea i can barely eat and i am unable to live life and go anywhere
  - i have a basal cell carcinoma what specialist should i consult
  - i have recently had breast surgery removal of calcium deposits with cancer cells
  - is cirrhosis a form of liver cancer

Cluster 41  (n=1328)
 Top terms: test, urine, blood, kidney, uti, bladder, result, drug, pain, infection, pregnancy, stone
 Top tags: drug test, blood test, pregnancy, pregnancy test, urination, smoking, kidney stone, marijuana, kidney, chlamydia
 Examples:
  - i have been suffering from low grade fever excessive saliva at night dark and less urine
  - my semen was pink in color and the last time it had blood in it
  - pain in testicles lower abdomen rectum is it fatty liver
  - my left side testicle is hurting and pissing blood
  - why does my urine smell

Cluster 42  (n=776)
 Top terms: eye, vision, cataract, head, pain, nose, doctor, eyelid, stye, episode, pinkeye, neck
 Top tags: eyes, vision, cataract, pink eye, swelling, nose, eyelid, sty, antibiotic, burn
 Examples:
  - i have heavy pain in both side of my head that causes dizziness sometimes in my back and neck
  - seems i have got chemical in eye from eye cream anything i can do to get relief from burning i have tryed to rinse
  - i have been suffering from pressure and pain behind eyes for almost three years
  - i was checking my husband is testicles and i felt a peanut size lump above his left testicle not sure if we should worry  we are both without insurance right now it would be nice to know if i should just keep an eye on it or if it will just go away
  - i have hyperpigmentation on my face what sort of concealer or base should i look for

Wrote assignments to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/clustered_short_questions.csv

=== Assessment Metrics (Spectral Nyström, COSINE-only) ===
 Features: BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245) = 100D
 k=43 clusters
 Global: silhouette(sampled, cosine)=0.3123

--- Cluster Size Stats ---
 min=18 max=4012 mean=1104.4 median=834.0 std=804.3

--- Per-Cluster Averages (sampled silhouette, cosine) ---
       0 | size=  892 | mean silhouette=0.9021
       1 | size= 1548 | mean silhouette=0.3244
       2 | size= 4012 | mean silhouette=-0.3237
       3 | size= 1937 | mean silhouette=-0.2286
       4 | size= 3132 | mean silhouette=-0.2605
       5 | size= 1533 | mean silhouette=0.8271
       6 | size=  693 | mean silhouette=0.9091
       7 | size= 1158 | mean silhouette=-0.0982
       8 | size=  771 | mean silhouette=0.9056
       9 | size= 1372 | mean silhouette=0.9004
      10 | size= 1066 | mean silhouette=0.7530
      11 | size= 1533 | mean silhouette=0.1310
      12 | size=   18 | mean silhouette=0.0000
      13 | size=  805 | mean silhouette=0.7448
      14 | size=  726 | mean silhouette=0.6311
      15 | size= 1109 | mean silhouette=0.7983
      16 | size=  976 | mean silhouette=0.2842
      17 | size= 1067 | mean silhouette=0.7810
      18 | size=  445 | mean silhouette=0.6149
      19 | size=  822 | mean silhouette=0.7955
      20 | size=  715 | mean silhouette=0.4872
      21 | size=  567 | mean silhouette=0.7154
      22 | size= 1089 | mean silhouette=0.0491
      23 | size=  601 | mean silhouette=0.2198
      24 | size=  411 | mean silhouette=0.6975
      25 | size=  351 | mean silhouette=0.9371
      26 | size=  834 | mean silhouette=0.7156
      27 | size= 1279 | mean silhouette=0.2774
      28 | size=  711 | mean silhouette=0.4600
      29 | size=  591 | mean silhouette=0.5848
      30 | size=  669 | mean silhouette=0.3613
      31 | size= 1107 | mean silhouette=0.6065
      32 | size= 3906 | mean silhouette=-0.1449
      33 | size=  634 | mean silhouette=0.7380
      34 | size=  805 | mean silhouette=0.3403
      35 | size= 1020 | mean silhouette=0.0284
      36 | size=  456 | mean silhouette=0.5492
      37 | size=  598 | mean silhouette=0.7668
      38 | size= 1189 | mean silhouette=0.2881
      39 | size= 1420 | mean silhouette=0.1444
      40 | size=  819 | mean silhouette=0.6571
      41 | size= 1328 | mean silhouette=0.2762
      42 | size=  776 | mean silhouette=0.5350

--- Structure (cosine on Z) ---
 Avg intra-cluster cosine similarity: 0.6702 (higher = tighter)
 Mean inter-centroid cosine similarity: 0.0202 (lower = better)

--- Tag Coherence (avg) ---
 Purity=0.498 | Entropy=4.316 | Intra-Jaccard≈0.117

Saved model-selection metrics to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_model_selection.csv

--- Model Selection Rows ---
 k  silhouette_cosine  max_cluster_diameter_cosine
43           0.312308                     1.485693

[UMAP] Saved 2D scatter to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_scatter.png
[UMAP] Saved embedding CSV to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/umap_2d_embedding.csv

[HClust] Saved dendrogram to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/hierarchical_dendrogram.png

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/cluster_timings.json

=== Runtime (seconds) ===
               load_csv_s:   0.2568
         nlp_preprocess_s: 115.1775
    vectorize_text_tags_s:  97.1178
         scale_features_s:   0.0696
         spectral_embed_s:   3.6825
                 k_scan_s:  31.0121
     interpret_clusters_s:   0.4376
           save_outputs_s:   0.2705
        metrics_compute_s:   1.8832
                   umap_s:  31.2011
           hierarchical_s:   0.2671
          total_runtime_s: 281.7612
```
<img width="790" height="490" alt="CSCi E-108 v2 4 3 plot 1" src="https://github.com/user-attachments/assets/ba2e66e8-f15e-4c65-a12e-2e399297044e" />

<img width="790" height="490" alt="CSCi E-108 v2 4 3 plot 2" src="https://github.com/user-attachments/assets/8a5b24fb-f2ae-4470-8658-4e7453cdbda0" />

<img width="988" height="790" alt="CSCi E-108 v2 4 3 plot 7" src="https://github.com/user-attachments/assets/bb6dbc9b-60a2-4cfa-ac6d-a98116a3ec93" />

<img width="1189" height="590" alt="CSCi E-108 v2 4 3 plot 8" src="https://github.com/user-attachments/assets/61c2a465-741b-45e4-bb53-0605ba067233" />

In v2.4.3 I am still working with the same corpus: 47,491 short medical questions after POS cleanup, with `short_question` and `tags` as the core fields and same data preprocessing and NLP as in v1.1.3.

The feature pipeline follows the same pattern as in v2.3.3. Sentence-BERT (`all-MiniLM-L6-v2`) creates 384-dimensional embeddings from `short_question_pos`. I apply optional L2 normalization and then reduce these embeddings to 80 dimensions using PCA, retaining about 72.1% of the variance. In parallel, `tag_list` is binarized via MultiLabelBinarizer and then compressed to 20 dimensions with TruncatedSVD, capturing about 24.5% of the variance in tag space. Concatenating these gives a 100-dimensional vector per question: 80 dimensions of BERT-PCA carrying semantic similarity between cleaned texts and 20 dimensions of tag-SVD injecting weak label information so that questions with similar tag patterns move closer together. A StandardScaler then standardizes these 100 dimensions, and a row-wise L2 normalization yields `X_unit`. Cosine distance on `X_unit` is thus equivalent to Euclidean distance, which stabilizes kernels and distance-based methods in subsequent steps. Since the base representation is unchanged from v2.3.3, any behavioural differences come from what happens in the clustering stage, not from the embedding itself.

The major modelling change is the introduction of Nyström spectral embedding before clustering. I randomly sample 2,000 landmark points from `X_unit` and build a cosine kernel among these landmarks, clipping similarities at zero and adding a small ridge on the diagonal for numerical stability. After an eigen-decomposition, I keep the 40 largest eigenvectors. For all 47k points, I compute cosine similarities to the landmarks, then sparsify by keeping only the top 64 similarities per row, building a CSR matrix S. The Nyström approximation then maps all points into a 40-dimensional spectral space `Z = S @ (U_m * Λ^{-1/2})`, which approximates the leading eigenvectors of the full cosine kernel. This whole spectral step runs in only a few seconds and yields a smoother, more globular manifold where K-means behaves better than directly on the original 100D feature space. In contrast, v2.3.3's OPTICS run operated on the 100D space with a density-based objective, which was slower and produced many noise points.

Once I have Z, I run a k-scan with MiniBatchKMeans for k between 10 and 45. For each k I fit K-means on Z, compute a sampled cosine silhouette on up to 8,000 points, and estimate the maximum cluster diameter in cosine distance. The silhouette rises from about 0.095 at k = 10 to about 0.312 at k around 43, without a clear elbow; it is essentially "more clusters yield more local purity" within the design cap. The maximum diameter stays in the range of roughly 1.50-1.56 across k, with no strong downward trend. That suggests that higher k improves separation but does not dramatically shrink the worst-case cluster, which is consistent with a few big "miscellaneous" clusters remaining diffuse. I select k = 43 as the best compromise, since it maximizes the sampled silhouette within the chosen range and yields a reasonably granular taxonomy.

With k fixed at 43, every question is hard-assigned to a cluster; there is no notion of noise here. Cluster sizes range from as small as 18 points (for a tiny, tight vitamin/cold cluster) to as large as 4,012 points (for a catch-all symptom/overdose/pain cluster), with mean size around 1,104, median 834, and a standard deviation around 804. Qualitatively, I can see several categories emerging. A large number of clusters are high-quality, very tight topical groups with silhouettes above about 0.7. These include medication and OTC-related clusters, period and pregnancy and birth-control clusters, anxiety and depression, rashes and dermatology, flu and vaccines, respiratory infections, heart disease and blood pressure, headaches and migraines, STDs and genital symptoms, cancer and tumors, UTI and kidney and lab tests, eye and vision problems, hepatitis and viral disease, and thyroid and osteoporosis. Their top terms and dominant tags line up cleanly, and example questions read as semantically homogeneous. These are the clusters I would trust the most for routing questions, defining interpretable topic labels, or acting as anchor groups for supervised models.

Alongside those, there is a small set of broad "hub" clusters with negative or low mean silhouette, which function more like residual manifolds. The biggest ones, clusters like C2 (around 4,000 points), C3 (about 1,900), C4 (over 3,000), and C32 (nearly 4,000), mix overdose, musculoskeletal pain, hair loss, various GI issues, hepatitis, smoking, chronic conditions and other heterogeneous symptoms. Their negative silhouettes mean many points in those clusters are actually closer (in cosine distance on Z) to another cluster centroid than to their assigned one. A few other clusters show low but positive silhouettes and mixed content, often combining multiple related systems. These broad clusters are natural candidates for second-stage refinement: one could re-cluster within them, introduce tag-based constraints to separate incompatible diseases, or treat them as "other" buckets where human review or additional modelling is needed.

The dendrogram built on the 43 centroids in cosine space provides a higher-level view. Many merges occur at cosine distances around 0.85-0.95, which indicates moderate distinctness rather than complete orthogonality. Branches often align with intuitive families: cardio clusters sit near blood-pressure clusters; skin and rash clusters are grouped; pregnancy and menstrual clusters appear together; some disease and neurology clusters sit next to mood disorder or pediatric clusters. That hierarchical structure is helpful for rolling up the 43 fine-grained topics into coarser macro-topics like "Pregnancy and Reproductive Health", "Cardio-Metabolic", "Respiratory Infections", "Dermatology", "Mental Health", and similar.

The metrics confirm this structural picture. The global sampled cosine silhouette on Z at k = 43 is about 0.312, which is reasonably strong given the size and noisiness of the medical corpus. Many clusters individually have silhouettes in the 0.6-0.9 range, reflecting the tight topical groups already mentioned. A handful sit in the 0-0.3 range, and the four huge mixed clusters have negative silhouettes, which is expected given their heterogeneity. On Z, average intra-cluster cosine similarity is about 0.670, so points within a given cluster are fairly close in the spectral space. The mean inter-centroid cosine similarity is around 0.020, meaning the centroids are nearly orthogonal on average; given that there are 43 of them, that level of separation is quite acceptable.

Tag-based metrics give another angle. Mean tag purity is about 0.498, so roughly half the questions in a typical cluster share the dominant tag. Mean tag entropy is around 4.316, which is fairly high, reflecting multi-tag questions and noisy labeling. Mean intra-cluster tag Jaccard similarity is about 0.117, so questions in the same cluster share tag sets more often than by chance, but overlap is far from complete. In high-silhouette clusters, purity and Jaccard are substantially higher; the mixed clusters drag down the overall averages. The maximum cosine cluster diameter at k = 43 is around 1.486, which corresponds to a worst-case pair in the largest cluster having cosine similarity about -0.486. That again is consistent with big mixed clusters where some points are almost opposite directions in the embedding but still assigned to the same group under K-means.

One of the nicest aspects of v2.4.3 is runtime. Total runtime is around 282 seconds (4.7 minutes), compared to roughly 1,957 seconds (32.6 minutes) for v2.3.3. POS NLP with spaCy still takes about 115 seconds and vectorization (BERT and tag SVD) about 97 seconds, so those parts are unchanged. The Nyström spectral embedding is extremely fast, around 3.7 seconds for 47k points. The K-scan over 36 K-means runs on Z takes about 31 seconds, and UMAP another 31 seconds. In contrast, OPTICS alone in v2.3.3 took around 1,332 seconds. So the Nyström-plus-K-means stack not only gives a better global picture (full partition, higher global silhouette, many tight disease-specific clusters) but also dramatically improves scalability.

Overall, v2.4.3 gives a full hard assignment of all 47k medical questions into 43 clusters, with roughly half of those clusters being high-quality, single-topic medical themes and a handful of very large, low-quality clusters acting as mixed reservoirs. For practical use, I would name and keep the high-silhouette clusters as stable topics (Pregnancy and Birth Control, Vaccines and Flu, Respiratory Infections, Thyroid, Heart Disease, Migraine, STDs, Ear and Eye, Cancer, Hepatitis, UTI, and so on). The broad mixed clusters would then be candidates for either second-stage clustering or domain-specific splitting. For supervised intent modelling, this clustering can supply pseudo-labels for high-confidence groups and cluster IDs as additional features. For taxonomy or analytics, it provides a workable mid-granularity map of the corpus, which is far more complete and efficient than the four-disease-island picture in v2.3.3.

If I wanted to refine the system further, I would probably start by re-clustering within the big negative-silhouette clusters, perhaps allowing a slightly higher k than 45 in that subspace, and experiment with tag-aware constraints so that tag-incompatible points are less likely to share clusters. But even at this stage, v2.4.3 feels like a practical, balanced compromise between interpretability, coverage, clustering quality and runtime.

# Comparison between Clustering Models


# Table 1. Embedding and NLP Representation

| Version      | Text Cleaning               | Embedding Strategy                          | Dimensionality           | Tag Usage | Notes                                             |
| ------------ | --------------------------- | ------------------------------------------- | ------------------------ | --------- | ------------------------------------------------- |
| **v1.1.3**   | Basic normalize + POS nouns | MiniLM-L6-v2 raw embeddings                 | ~384 + tag dims          | SVD-20    | High dimensional; no PCA; early baseline          |
| **v1.2.3**   | Same POS noun filtering     | MiniLM raw embeddings (no PCA)              | ~384 + tag SVD           | SVD-20    | Dense representation; stable but noisy            |
| **v1.4.3**   | Same POS noun filtering     | MiniLM → graph manifold embedding           | Variable (manifold dims) | SVD-20    | Embedding reflects nearest-neighbor structure     |
| **v1.2.1.3** | Same POS filtering          | MiniLM + optional PCA                       | ~100 total               | SVD-20    | Transitional version reducing noise before KMeans |
| **v2.3.3**   | Same POS filtering          | Same 100-D embedding                        | 100-D                    | Yes       | Embedding identical; difference lies in OPTICS    |
| **v2.4.3**   | Same POS filtering          | MiniLM → PCA(80)+SVD(20) → Nyström spectral | 40-D manifold            | Yes       | Most geometry-aware embedding in all versions     |

---

# Table 2. Modelling and Clustering Strategy

| Version      | Main Algorithm                     | Cluster Coverage    | Behavior                            | My Interpretation                      |
| ------------ | ---------------------------------- | ------------------- | ----------------------------------- | -------------------------------------- |
| **v1.1.3**   | KMeans on raw BERT                 | Full assignment     | Broad clusters, overlapping         | Good for a first taxonomy              |
| **v1.2.3**   | KMeans with k-scan                 | Full assignment     | Slightly cleaner than v1.1.3        | Still sensitive to high dimensionality |
| **v1.4.3**   | Spectral clustering on graph       | Full assignment     | Mixed cluster sizes; manifold heavy | Reveals latent topics, uneven grouping |
| **v1.2.1.3** | KMeans on reduced dims             | Full assignment     | More stable boundaries than v1.2.3  | PCA reduces noise nicely               |
| **v2.3.3**   | OPTICS                             | Very small coverage | Extremely pure tiny clusters        | Best for prototype-label discovery     |
| **v2.4.3**   | Nyström spectral + MiniBatchKMeans | Full assignment     | Balanced, globally coherent         | Best blend of coverage + purity        |

---

# Table 3. Visualization Tools

| Version      | UMAP / 2D Plots | Dendrogram | Diagnostic Plots                   | Notes                             |
| ------------ | --------------- | ---------- | ---------------------------------- | --------------------------------- |
| **v1.1.3**   | Optional UMAP   | No         | Silhouette vs k                    |                                   |
| **v1.2.3**   | Yes             | Yes        | Silhouette vs k; diameter vs k     | Good interpretability tools       |
| **v1.4.3**   | Yes             | Yes        | Graph-based spectra                | Strong for hierarchy inspection   |
| **v1.2.1.3** | Yes             | Yes        | Same k-scan diagnostics            | Improved visuals from PCA         |
| **v2.3.3**   | Yes             | No         | Reachability plot (OPTICS)         | Shows a few dense spikes          |
| **v2.4.3**   | Yes             | Yes        | k-scan; spectral plots             | Most complete visualization suite |

---

# Table 4. Performance Outcomes and Behavior

| Version      | Cluster Coherence            | Overlap         | Noise           | Speed                      | Practical Use                      |
| ------------ | ---------------------------- | --------------- | --------------- | -------------------------- | ---------------------------------- |
| **v1.1.3**   | Low-moderate                 | High            | None            | Moderate                   | Rough taxonomy                     |
| **v1.2.3**   | Moderate                     | Medium          | None            | Moderate                   | Improved first-generation clusters |
| **v1.4.3**   | Mixed; some very good        | Medium          | None            | Heavy computation          | Manifold-based grouping            |
| **v1.2.1.3** | Higher than v1.2.3           | Lower           | None            | Fast                       | Useful intermediate model          |
| **v2.3.3**   | Extremely high purity        | Minimal         | Very high noise | Slow                       | Core island extraction             |
| **v2.4.3**   | Consistently strong clusters | Reduced overlap | None            | Fastest of advanced models | My preferred taxonomy model        |

---

# Table 5. Metrics and Quantitative Diagnostics

| Version      | Global Silhouette       | Per-cluster Silhouette | Diameter Trends   | Tag Coherence | Overall Impression              |
| ------------ | ----------------------- | ---------------------- | ----------------- | ------------- | ------------------------------- |
| **v1.1.3**   | ~0.07                   | Many low               | High              | Low           | Sparse internal cohesion        |
| **v1.2.3**   | ~0.09                   | Some medium            | High              | Low-medium    | Slightly better but noisy       |
| **v1.4.3**   | Variable                | Uneven                 | Cluster dependent | Medium        | Interesting but unstable        |
| **v1.2.1.3** | ~0.12                   | Several medium         | Lower than v1.2.3 | Medium        | Good step forward               |
| **v2.3.3**   | High on small clusters  | Very high              | Very small        | Very high     | Excellent purity, poor coverage |
| **v2.4.3**   | ~0.31                   | Many 0.6-0.9           | Stable            | Medium-high   | Best balance across all metrics |

---

# Table 6. Plot Quality and Insights

| Version      | Silhouette Plot       | Diameter Plot    | Reachability / Density | Spectral / Nyström | My View                  |
| ------------ | --------------------- | ---------------- | ---------------------- | ------------------ | ------------------------ |
| **v1.1.3**   | Simple                | Simple           | None                   | None               | Limited insight          |
| **v1.2.3**   | Clear rising trend    | Flat/slow        | None                   | None               | Helps k selection        |
| **v1.4.3**   | Not central           | Not central      | None                   | Yes                | Reveals structure        |
| **v1.2.1.3** | Useful                | Useful           | None                   | No                 | Better shape due to PCA  |
| **v2.3.3**   | N/A                   | N/A              | Strong reachability    | No                 | Shows purity spikes      |
| **v2.4.3**   | Monotonic improvement | Stable diameters | No                     | Yes                | Most informative plots   |

---

# Short Summary

v1.1.3-v1.2.3 represent early, high-dimensional KMeans clustering.
v1.4.3 moves toward manifold thinking but stays uneven.
v2.1.3 stabilizes early pipelines using PCA.
v2.3.3 focus on density-based approaches with high purity but limited coverage.
v2.4.3 becomes the most balanced system: global geometry awareness, cleaner clusters, faster runtime, strong metrics, and a coherent visualization suite.


## Embedding & Representation

Across all versions the core NLP preprocessing remains the same: normalize text, use spaCy to keep only nouns/proper nouns (plus a small whitelist of short medical tokens), drop verbs/adjectives/adverbs/stopwords/numerals — so the representation of each question becomes a compact "bag of medical concepts." That is consistent from v1.1.3 through v2.4.3.

* **v1.1.3**: After cleaning, the pipeline uses the full 384-dimensional MiniLM embeddings on the nounfiltered text, plus tag embeddings (via SVD), yielding a dense ~400-D space.
* **v1.2.3**: Same as v1.1.3: raw 384-D BERT embeddings + 20-D tag SVD → 404-D features (no further dimensionality reduction).
* **v1.4.3**: Embedding is similar to v1.2.3 in concept, but in this version I move away from BERT space clustering: I compute a nearest-neighbors graph over the embeddings, run a spectral decomposition (manifold extraction), and embed questions into a lower-dimensional manifold before clustering. This changes the geometry significantly.
* **v2.1.3**: I modify the representation more deliberately: after BERT embedding I apply PCA to reduce to 80 dimensions, then concatenate 20-D tag SVD. This yields a 100-D feature space. The intent is to denoise, remove redundant variance, and make clustering more robust.
* **v2.3.3**:  Representation remains the 100-D BERT-PCA + tag-SVD space, standardized and L2 normalized. I then treat that as a cosine-friendly latent space for density-based clustering. And I use the clustering method (density-based OPTICS) and define how clusters are selected.
* **v2.4.3**: Embedding remains the 100-D space initially, but before clustering I compute a Nyström spectral embedding (kernel approximation) over the normalized vectors, mapping everything into a 40-D manifold that better captures global cosine geometry and latent structure. Clustering is then done on that 40-D space.

In short: v1.1.3 → v1.2.3 stays dense and high-dimensional; v1.4.3 uses manifold extraction; v2.1.3 reduces dimensionality; v2.3.3 keep that but change clustering method; v2.4.3 moves toward a kernel-based low-dimensional spectral manifold before clustering.

---

## Modelling / Clustering Strategy

Because each version combines embedding with a clustering model, the chosen clustering algorithm fundamentally changes how the groups form.

* **v1.1.3 & v1.2.3** use **KMeans** over the dense embedding space. In v1.2.3 I perform a k-scan from k = 10 to 45 using cosine silhouette and maximum cluster diameter to pick k = 45. That forces a full partition: every question belongs to some cluster.
* **v1.4.3** abandons pure KMeans partitions: after embedding + graph construction, it runs **spectral clustering** (via nearest-neighbors graph) and uses a subset of data plus label propagation to assign all questions. This helps recover latent manifold structure and avoids forcing unnatural partitions based on Euclidean volume in the embedding space.
* **v2.1.3** returns to KMeans (cosine on 100-D normalized features) with a full partition approach, scanning k up to 45, benefiting from a denser but lower-dim embedding space that is less noisy than 400-D.

* **v2.3.3** uses **OPTICS** (another density-based method) on the same 100-D space, with a configuration that yields only a handful of very tight clusters and classifies most questions as noise. This favors high precision at the cost of coverage.
* **v2.4.3** brings in a hybrid strategy: perform a spectral embedding (Nyström) into a lower-dimensional manifold, then run **MiniBatchKMeans** (cosine) on that 40-D space. This combines the benefits of geometry-aware embedding with scalable partitioning.

Thus over versions the clustering strategy evolves from naive volume-based partitions to manifold-aware clustering, then to density-aware clustering (noise + islands), and finally to a hybrid: manifold embedding + partitioning.

---

## Visualization, Diagnostics & Metrics

Visualization and diagnostic tools also evolve.

* **v1.1.3 / v1.2.3** include classic evaluation: silhouette vs k, maximum cluster diameter vs k (cosine distance), and after clustering I inspect top-term summaries (via TF-IDF on noun text), tags per cluster, and sample original questions. There is no noise, and every question receives a cluster. Visual outputs include scatter plots (if dimensionality reduction used), though without manifold tools.
* **v1.4.3** adds a more sophisticated view: since clustering is done on a graph embedding, I use graph-based manifold visualization (e.g. UMAP or similar) and hierarchical dendrogram of centroids to give a sense of topical hierarchy. That helps reveal latent structure beyond flat clusters.
* **v2.1.3** keeps the k-scan diagnostics and cluster summaries, benefiting from lower-dimensional embeddings that make silhouette and diameter calculations more meaningful and stable. UMAP / 2D visualizations remain helpful, but the lower embedding noise makes clusters appear tighter and better separated in 2D projections.

* **v2.3.3** uses OPTICS, which tends to find a very few but extremely tight clusters. Diagnostically, I see very high intra-cluster cohesion and high silhouette on the labeled subset, but nearly all data is unlabeled. UMAP confirms a few isolated "islands" and a large scattered remainder. Metrics are calculated only for clusters; noise is ignored. This version serves as a strong "core-island detector" rather than taxonomy builder.
* **v2.4.3** introduces a Nyström spectral embedding step, which reveals global geometry better, followed by KMeans partitioning: it recovers a full partition but over a manifold-informed 40-D space. Diagnostics include k-scan silhouette vs diameter plots, UMAP on the 40-D manifold, TF-IDF + tag summaries for cluster interpretability, per-cluster size breakdowns, and structure measures (intra-cluster cosine, inter-centroid cosine) that are stronger than in earlier versions. The result is a comprehensive taxonomy with many high-confidence clusters and a diagnostic toolkit that makes cluster evaluation more principled.

---

## Performance, Results & Quality Trade-offs

Putting them all side by side, the versions show different trade-offs between coverage, purity, interpretability, and runtime.

* **v1.1.3 / v1.2.3** deliver full coverage: every question is assigned to a cluster. But because clusters are fairly broad, global silhouette is low (~0.07-0.10), intra-cluster similarity is modest, and many clusters are noisy or overlapping in semantics. Their advantage: simple and fast-ish (though k-scan over high dimensions is expensive), and produce a rough—but usable—first taxonomy.
* **v1.4.3** sacrifices some interpretability by consolidating structure via spectral clustering; large mixed clusters emerge that absorb many ambiguous cases, but a few clean topic islands appear. Global structure is more hierarchical, but coverage remains full (though clusters vary wildly in size), and overall coherence improves for some clusters at the cost of others.
* **v2.1.3** benefits from a cleaner embedding space (100-D) and yields slightly better global silhouette (~0.12), better cluster separation, and somewhat tighter clusters overall compared to v1.2.3. However, because it still uses KMeans, it inherits the weakness that some clusters remain mixed and overlap remains substantial.

* **v2.3.3** pushes density-only clustering further: a handful of very tight clusters remain, and everything else is noise. The clusters are high-purity and semantically coherent, but coverage is extremely limited. So the result is a small set of "anchor topics" rather than a comprehensive taxonomy.
* **v2.4.3** strikes a balance: by embedding into a 40-D spectral manifold, then doing KMeans, I recover a full partition of all questions (coverage), while gaining better separation, manifold-respecting geometry, and a higher global silhouette (~0.31). Many clusters are tight, coherent, and medically meaningful; a few remain broad or mixed, but the overall taxonomy is much higher quality and substantially more usable than v1.x or raw v2.1.3. Runtime is also good — faster than the density-based versions — because spectral embedding compresses geometry and KMeans scales well in lower dimension.

---

## Predicted Use-cases and Strengths / Weaknesses

From what I see:

* If I want a **rough but full coverage taxonomy** quickly, v1.2.3 or v2.1.3 works: easy setup, full assignment, interpretable clusters though somewhat noisy.
* If I want **very clean, high-purity topic anchors** (for say, building a small seed taxonomy or supervision seed), then v2.3.3 (OPTICS) excel: it isolates dense semantic islands where the questions are strongly coherent and tags line up, but it leaves most of the data unassigned — a feature, not a bug, if the goal is precision over recall.
* If I want a **better-balanced taxonomy** — full coverage, reasonable cohesion, more stable cluster geometry, and interpretable clusters — v2.4.3 seems like the best compromise. It uses manifold-aware embedding plus efficient clustering to carve the space into many medically relevant topics, while avoiding some of the noise and overlap overhead of early high-dimensional KMeans or the over-fragmented or over-noisy clusters of density-only methods.
* As a final fallback, **v1.4.3** has historical interest: spectral clustering on the full 400-D space gives interesting insights into latent manifold structure, but its mixed clusters and uneven sizes make it harder to use for a stable, universal taxonomy.

---

# Similarity Search and Reranking

## v1

```
[Preprocess] Rows after POS cleanup: 47491
[Features] BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245)
[Shapes] X_sem_norm: (47491, 80), X_tag_norm: (47491, 20)

=== Example 1: text-only query ("period late cramps") ===
                                                                                                                                                                                                                                                                                                                                                                                              short_question                                                          tags  semantic_score  blended_score
normal period two weeks later bad stomach cramps discharge tender breasts and started bleeding never happened before this has never happened before i normally get my period around the same time and have never im 27 years old gotten two periods in one month or after only two weeks after the first one just started feeling tender breasts and discharge and then started bleeding could i be pregnant        ['period' 'pregnancy' 'breast' 'stomach' 'tenderness']        0.846314       0.592419
normal period two weeks later bad stomach cramps discharge tender breasts and started bleeding never happened before this has never happened before i normally get my period around the same time and have never im 27 years old gotten two periods in one month or after only two weeks after the first one just started feeling tender breasts and discharge and then started bleeding could i be pregnant        ['period' 'pregnancy' 'breast' 'stomach' 'tenderness']        0.846314       0.592419
normal period two weeks later bad stomach cramps discharge tender breasts and started bleeding never happened before this has never happened before i normally get my period around the same time and have never im 27 years old gotten two periods in one month or after only two weeks after the first one just started feeling tender breasts and discharge and then started bleeding could i be pregnant        ['period' 'pregnancy' 'breast' 'stomach' 'tenderness']        0.846314       0.592419
                                                                                                                                                                                                                             i have sore breast for 3 week and bad cramps but no period and its due come already could i be pregnant  i had sex 3 days before ovulation and a day after he pulled out though ['breast' 'cramps' 'period' 'pregnancy' 'sexual intercourse']        0.784227       0.548959
                                                                                                                                                                                                                             i have sore breast for 3 week and bad cramps but no period and its due come already could i be pregnant  i had sex 3 days before ovulation and a day after he pulled out though ['breast' 'cramps' 'period' 'pregnancy' 'sexual intercourse']        0.784227       0.548959

=== Example 2: tags-only query (['pregnancy', 'period']) ===
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     short_question                                                 tags  semantic_score  tag_jaccard  blended_score
i have been off depo shot for over a year and have no period i recently began having pregnancy symptoms am i pregnant  i was on the depo shot for 2 years and have not gotten the shot since september 2011 i have still not had a period and have been having pregnancy symptoms my fiance and i use the withdraw method i realize that you can still get pregnant this way after over a year of not receiving the shot i do not think the symptoms could be attributed to the shot i had a pap smear 9 12 12 and it came back fine so could i be pregnant and if not what could be causing the pregnancy symptoms                               ['period' 'pregnancy']             0.0          1.0           0.30
                                                                                                                                                                                                                                                                                                                                   can u get pregnant on the pill  i have been taking the pill for 2 months now and my last peroid wasnt really a period i bled for maybe 5 minutes a day for 3 days could i be pregnant i just started feeling nausea and im hungry but when i start eating i feel like i cant eat             ['period' 'pregnancy' 'nausea' 'hunger']             0.0          0.5           0.15
                                                                                                                                                                                                                                                                                                                                                      what can cause a period to be late  i am 25 have always had regular periods and i am now 3 days late i had protected sex 4 days ago i have been under a lot of stress and not sleeping well could i be pregnant or what else could cause my period to be late ['sexual intercourse' 'period' 'stress' 'pregnancy']             0.0          0.5           0.15
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i am 44 yrs tubal ligation 18yrs ago last period came for 1 day my breast hurt for 3 wks and vomiting am i pregnant              ['period' 'breast' 'vomit' 'pregnancy']             0.0          0.5           0.15
                                                                                                                                                                i skip my periods with my birth control trinessa i am sexually active and very busy so if it is unnecessary for me to have a period u would rather not have it my boyfriend and i use both condoms and my birth control is there a higher chance of pregnancy if i skip my pills or is there even a chance of pregnancy i take my pill pretty religously but i have had times where i have missed the time to take it and have taken it hours later      ['period' 'birth control' 'condom' 'pregnancy']             0.0          0.5           0.15

=== Example 3: mixed query (text + tags) ===
                                                                                                                                                                                 short_question                                               tags  semantic_score  tag_jaccard  blended_score
                                                                             if i dont start my period til 1017 but my pregnancy test today showed a very barely visible positive am i pregnant            ['period' 'pregnancy' 'pregnancy test']        0.740517         1.00       0.818362
pregnant  unprotected sex the 7th period due the 26th but days later there was brown and red spotting when i wiped i took a pregnancy test the following week but it was negative am i pregnant ['period' 'spotting' 'pregnancy test' 'pregnancy']        0.815224         0.75       0.795657
                                                       the first day of my last period was 110<negative_smiley>12 and today is 121312 no period when can i take a pregnancy test  am i pregnant            ['period' 'pregnancy' 'pregnancy test']        0.693769         1.00       0.785638
                                                       the first day of my last period was 110<negative_smiley>12 and today is 121312 no period when can i take a pregnancy test  am i pregnant            ['period' 'pregnancy' 'pregnancy test']        0.693769         1.00       0.785638
                                                                        im 25 trying to get pregnant for the past 3 years havnt had period since dec 10 2013 test pregnancy test shows negative            ['pregnancy' 'period' 'pregnancy test']        0.654260         1.00       0.757982

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/retrieval_timings.json

=== Runtime (seconds) ===
       preprocess_s: 123.6044
        vectorize_s:  92.6548
    total_runtime_s: 217.8842
```

### 1. About my pipeline generally

In this v1 similarity search and reranking pipeline, I start from `train_data_chatbot.csv`, fill missing `short_question` and `tags` with empty strings, and then normalize the text. The normalization step lowercases everything, normalizes quotes and dashes, strips non-alphanumeric characters except `'` and `?`, and stores the result as `short_question_norm`. For the tag side, I apply a regex pattern `TAG_PATTERN = r"'([^']+)'"` to extract everything that appears inside single quotes, clean and normalize those strings, and store them as `tag_list`. This gives me a clean, machine-friendly pair for each row: normalized question text and a structured tag list.

On top of that, I run a **POS-filtering** step with spaCy via `build_pos_filtered_texts`. I use `en_core_web_sm` with NER and textcat turned off for efficiency. For each token, I keep it if its POS tag is NOUN or PROPN, or if its lemma appears in a small whitelist `MED_KEEP_SHORT` (for medical abbreviations like HPV, UTI, HIV). I drop standard and domain stopwords, numeric-like tokens, and very short tokens (length ≤ 2) unless they are whitelisted medical terms. This produces `short_question_pos`, which deliberately focuses on *medical content nouns*, diseases, body parts, drugs, and similar concepts, while stripping out chatty filler and syntax.

The effect is that I push most semantic weight onto medically meaningful nouns. For clustering and retrieval, this has clear benefits: it makes topical *islands* like shingles, hypothyroidism, or hepatitis much more visible in embedding space, because I am not distracted by auxiliary phrasing. The downside is that I lose tense, negation, and some modifiers, so something like "not pregnant" and "pregnant" can look more similar than they should. From the logs, preprocessing takes around `preprocess_s ≈ 123 s`, and spaCy POS tagging is the main one-time cost.

---

### 1.2 Modeling and representation

After preprocessing, I construct a **joint dense representation** with two separate blocks: one for semantic text, one for tags. On the semantic side, I run Sentence-BERT (`all-MiniLM-L6-v2`) over `short_question_pos`, producing `X_sem` as dense vectors. I optionally L2-normalize these rows, then apply PCA to reduce to 80 dimensions (`N_BERT_DIM`), with an explained variance around 0.72. After that, I apply `StandardScaler` and another row-wise L2 normalization so that I end up with `X_sem_norm` of shape `(47491, 80)` that lives cleanly in cosine space.

On the tag side, I use `MultiLabelBinarizer` to convert `tag_list` into a sparse multi-hot matrix, then apply TruncatedSVD to get a 20-dimensional representation (`N_TAG_DIM`) with an explained variance of about 0.24. I then scale and L2-normalize that as well, resulting in `X_tag_norm` of shape `(47491, 20)`.

A key design choice here is that I **do not** concatenate semantic and tag vectors into one big feature array for retrieval. Instead, I keep them as two separate channels and combine them only at scoring time. Conceptually, the BERT+PCA block encodes semantic similarity over noun phrases, while the tag SVD block captures a kind of **weak supervision** via tag co-occurrence structure. I actually use tags in two forms: a dense latent representation (`X_tag_norm`), which is currently "parked" for future use, and discrete tag sets (`tag_sets`) that I use directly for Jaccard similarity. In the current `search_questions` function, the **actual** score I use is made from:

* `semantic_score = X_sem_norm @ q_sem_norm` (cosine similarity because of L2),
* `tag_jaccard` between the query tag set and each candidate's tag set,

and then I blend them as:

[
\text{blended_score} = 0.7 \cdot \text{semantic_score} + 0.3 \cdot \text{tag_jaccard}.
]

The dense tag vectors `X_tag_norm` are computed but not yet used in the scoring formula; I keep them in reserve in case I want a **dense tag similarity** term later.

---

### 1.3 Retrieval logic

For each query, I run a query tower that mirrors the document tower. First, I normalize the query text and apply the same POS filtering, then encode it with Sentence-BERT, reduce via PCA, scale, and L2-normalize to obtain `q_sem_norm`. If the query includes tags, I pass them through the same multilabel binarizer and SVD to get `q_tag_norm`, and also maintain a discrete `q_tag_set` for Jaccard.

Candidate selection is initially *semantic-only*: I compute

[
\text{sem_scores} = X_{\text{sem_norm}} @ q_{\text{sem_norm}},
]

and select the top `candidate_k` (for example 300) documents by this semantic score. On that reduced candidate pool, I then compute `tag_jaccard` between `q_tag_set` and each candidate's `tag_sets[i]`, and finally recompute the **blended score**

[
\text{score} = 0.7 \cdot \text{semantic_cosine} + 0.3 \cdot \text{tag_jaccard}.
]

I then sort by this blended score and return the top_k results along with their original question text, tags, POS-filtered text, `semantic_score`, `tag_jaccard`, `blended_score`, and overlapping tags for inspection.

This setup behaves nicely in different regimes. When a query is text-only, it effectively behaves like pure dense retrieval in BERT space. If the query is tags-only (as in one of my examples), the semantic vector is essentially zero, so only the tag Jaccard component drives ranking. For mixed queries, BERT pulls in semantically similar questions while the tag Jaccard term promotes those with overlapping intent or diagnosis labels.

---

### 1.4 Measurement and metrics

Right now I am evaluating **informally rather than with a labeled retrieval benchmark**. The scores themselves are interpretable: the semantic component is essentially a cosine similarity in `[-1, 1]` but mostly lives in `[0, 1]` due to the POS filtering and BERT behavior, the tag Jaccard is bounded in `[0, 1]`, and the blended score is a weighted combination of the two. The quality assessment is largely **qualitative auditing**: I print neighbors for representative queries and visually inspect whether the retrieved questions make sense medically and whether the tags and question types align with what I expect. So far, for pregnancy/period and mental-health examples, the neighbours look coherent and medically meaningful. I have not computed explicit retrieval metrics like nDCG or recall@k, mostly because I lack a ground-truth relevance labeling for query-question pairs. For my current use case, medical FAQ reuse and intent discovery, this manual inspection is adequate, but it could be extended later if I build a labeled test set.

---

## 2. What's slow and what can be improved?

From the recorded timings, `preprocess_s ≈ 123.6 s` is dominated by spaCy POS tagging over ~47.5k rows, and `vectorize_s ≈ 92.7 s` covers BERT embedding, PCA, SVD, and scaling. Both of these are **one-off offline costs**, which I can tolerate as long as I cache the outputs. Once the embeddings and tag structures are built, query-time is relatively cheap: a single BERT forward pass, one spaCy+POS run on the query, and a single matrix multiply against `X_sem_norm`.

The main runtime pain point is spaCy at query-time. Running full POS over every user query is heavier than it needs to be. I see two reasonable options here. One is to keep POS filtering strictly at **index time** and do only a lighter normalization at query time, which would speed up interactive use considerably while still keeping the document embeddings denoised. Another is to keep the current POS step but add a configuration flag like `USE_SPACY_FOR_QUERY = False`, so I can switch it off for bulk or latency-sensitive scenarios.

For similarity search itself, I currently do a full matrix multiplication `X_sem_norm @ q_sem_norm` over approximately 47k documents, which is very fast on modern hardware (usually well below 0.01 seconds). At this scale I don't really need an approximate nearest neighbor index such as FAISS or hnswlib. If I scale up to millions of documents later, I would revisit that decision and introduce an ANN index for the semantic tower.

There are also some **under-used pieces**. The dense tag representation `X_tag_norm` is built but not used in the scoring, so I have some slack to incorporate a dense tag cosine term if I want. Also, there is currently no **BM25 or lexical scoring component**, which would be useful in cases where BERT fails to capture rare phrases, or when the query is extremely short and lexical overlap carries more signal than semantics.

---

## 3. HDBSCAN vs BM25 for improving similarity and reranking

I have thought about whether to plug **HDBSCAN** into this pipeline, but HDBSCAN is fundamentally a clustering algorithm, not a retrieval mechanism. It is great for discovering dense "islands" in embedding space, similar to what I do in v2.x with OPTICS/HDBSCAN, but it does not integrate cleanly with query-time search. Many points might be labeled as noise, queries themselves are not pre-clustered, and I would still need to rank neighbours within clusters and across clusters using some distance measure. For online similarity search, that extra complexity does not buy me much, so I would not insert HDBSCAN into this v1 retrieval path. Instead, clustering is something I'd use offline to define topic clusters or for exploration, not for scoring individual queries.

**BM25**, on the other hand, is a lexical ranking function that directly complements my dense BERT tower. It scores documents based on overlapping tokens with IDF and length normalization and tends to shine when exact phrases or rare keywords really matter. In a medical QA context, BM25 can rescue cases where BERT doesn't fully understand a rare disease name or where a very short query relies on a specific token. From a design perspective, BM25 is the right additional component for **similarity search and reranking**, whereas HDBSCAN belongs more to unsupervised topic discovery.

---

## 4. Concrete improvements I'd suggest

Given all of this, the next step I would take is to move toward a **hybrid scoring** scheme: semantic, tags, and BM25. A natural extension of the current formula would be:

[
\text{score} = \alpha \cdot \text{semantic_cosine} + \beta \cdot \text{tag_jaccard} + \gamma \cdot \text{bm25_norm},
]

where, for example, I might start with something like (\alpha = 0.6), (\beta = 0.2), and (\gamma = 0.2), then tune them based on manual inspection or small labeled sets. In this setup, semantic cosine preserves the dense BERT ranking, tag Jaccard enforces intent and diagnosis consistency, and normalized BM25 ensures that rare term matches are properly rewarded.

On the retrieval side, I would adjust candidate selection to use the **union of semantic and BM25 candidates**. That is, I would retrieve the top `candidate_k_sem` documents by semantic similarity and top `candidate_k_bm25` by BM25 score, take their union (typically a few hundred documents), and then compute the full blended score on this reduced set. This pattern is common in hybrid IR systems and keeps query-time latency manageable. If I want, I can also experiment with adding a **dense tag similarity** term like (\delta \cdot \cos(X_{\text{tag_norm}}[i], q_{\text{tag_norm}})), but for now the Jaccard term is simpler and more interpretable.

Finally, I would expose a configuration switch for spaCy at query-time. If `USE_SPACY_FOR_QUERY` is false, I would only run `normalize_text_basic` on the query, let BERT operate on lightly normalized text, and reuse the same index embeddings. This gives me a trade-off between recall and latency that I can choose based on the environment (interactive prototype vs. batch analysis).

---

## 5. Refactored code direction

In practice, the refactored full script I am aiming for takes my existing pipeline and adds a **BM25 reranker** on top. I still build the BERT-based semantic tower, the POS-filtered representations, and the tag structures exactly as before. On top of that, I add a BM25 index over `short_question_norm`, implement hybrid candidate selection via the union of semantic and BM25 top-k lists, and then compute a blended score that combines semantic cosine, tag Jaccard, and BM25. The idea is that I should be able to paste this into a fresh Colab, run it end to end, and see that the retrieval behaviour becomes more robust: semantically similar neighbours show up, tag-aligned neighbours get boosted, and rare, lexically crucial terms are no longer lost.

## v2

```
[Preprocess] Rows after POS cleanup: 47491
modules.json: 100%
 349/349 [00:00<00:00, 74.3kB/s]
config_sentence_transformers.json: 100%
 116/116 [00:00<00:00, 26.7kB/s]
README.md: 
 10.5k/? [00:00<00:00, 2.13MB/s]
sentence_bert_config.json: 100%
 53.0/53.0 [00:00<00:00, 12.2kB/s]
config.json: 100%
 612/612 [00:00<00:00, 142kB/s]
model.safetensors: 100%
 90.9M/90.9M [00:00<00:00, 153MB/s]
tokenizer_config.json: 100%
 350/350 [00:00<00:00, 75.3kB/s]
vocab.txt: 
 232k/? [00:00<00:00, 14.2MB/s]
tokenizer.json: 
 466k/? [00:00<00:00, 49.3MB/s]
special_tokens_map.json: 100%
 112/112 [00:00<00:00, 21.6kB/s]
config.json: 100%
 190/190 [00:00<00:00, 42.0kB/s]
[Features] BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245)
[Shapes] X_sem_norm: (47491, 80), X_tag_norm: (47491, 20)
[BM25] Built BM25 index over short_question_norm

=== Example 1: text-only query ("period late cramps") ===
                                                                                                                                                                                                                                                                                                                                                                                              short_question                                                          tags  semantic_score  bm25_score  blended_score
                                                                                                                                                                                                                             i have sore breast for 3 week and bad cramps but no period and its due come already could i be pregnant  i had sex 3 days before ovulation and a day after he pulled out though ['breast' 'cramps' 'period' 'pregnancy' 'sexual intercourse']        0.784227   19.856291       0.638672
                                                                                                                                                                                                                             i have sore breast for 3 week and bad cramps but no period and its due come already could i be pregnant  i had sex 3 days before ovulation and a day after he pulled out though ['breast' 'cramps' 'period' 'pregnancy' 'sexual intercourse']        0.784227   19.856291       0.638672
                                                                                                                                                                                        pregnant  i have been having unprotected sex for the past month and started getting cramps nausea sore nipples etc my period was 8 days late but i did get it but i have no cramps at all could i be pregnant thanks ['sexual intercourse' 'period' 'pregnancy' 'nipple' 'nausea']        0.772467   18.082153       0.616593
                                                                                                                                                                                        pregnant  i have been having unprotected sex for the past month and started getting cramps nausea sore nipples etc my period was 8 days late but i did get it but i have no cramps at all could i be pregnant thanks ['sexual intercourse' 'period' 'pregnancy' 'nipple' 'nausea']        0.772467   18.082153       0.616593
normal period two weeks later bad stomach cramps discharge tender breasts and started bleeding never happened before this has never happened before i normally get my period around the same time and have never im 27 years old gotten two periods in one month or after only two weeks after the first one just started feeling tender breasts and discharge and then started bleeding could i be pregnant        ['period' 'pregnancy' 'breast' 'stomach' 'tenderness']        0.846314   12.440670       0.613131

=== Example 2: tags-only query (['pregnancy', 'period']) ===
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     short_question                                                 tags  semantic_score  tag_jaccard  bm25_score  blended_score
i have been off depo shot for over a year and have no period i recently began having pregnancy symptoms am i pregnant  i was on the depo shot for 2 years and have not gotten the shot since september 2011 i have still not had a period and have been having pregnancy symptoms my fiance and i use the withdraw method i realize that you can still get pregnant this way after over a year of not receiving the shot i do not think the symptoms could be attributed to the shot i had a pap smear 9 12 12 and it came back fine so could i be pregnant and if not what could be causing the pregnancy symptoms                               ['period' 'pregnancy']             0.0          1.0         0.0            0.2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i am 44 yrs tubal ligation 18yrs ago last period came for 1 day my breast hurt for 3 wks and vomiting am i pregnant              ['period' 'breast' 'vomit' 'pregnancy']             0.0          0.5         0.0            0.1
                                                                                                                                                                                                                                                                                                                                                      what can cause a period to be late  i am 25 have always had regular periods and i am now 3 days late i had protected sex 4 days ago i have been under a lot of stress and not sleeping well could i be pregnant or what else could cause my period to be late ['sexual intercourse' 'period' 'stress' 'pregnancy']             0.0          0.5         0.0            0.1
                                                                                                                                                                                                                                                                                                                                   can u get pregnant on the pill  i have been taking the pill for 2 months now and my last peroid wasnt really a period i bled for maybe 5 minutes a day for 3 days could i be pregnant i just started feeling nausea and im hungry but when i start eating i feel like i cant eat             ['period' 'pregnancy' 'nausea' 'hunger']             0.0          0.5         0.0            0.1
                                                                                                                                                                i skip my periods with my birth control trinessa i am sexually active and very busy so if it is unnecessary for me to have a period u would rather not have it my boyfriend and i use both condoms and my birth control is there a higher chance of pregnancy if i skip my pills or is there even a chance of pregnancy i take my pill pretty religously but i have had times where i have missed the time to take it and have taken it hours later      ['period' 'birth control' 'condom' 'pregnancy']             0.0          0.5         0.0            0.1

=== Example 3: mixed query (text + tags) ===
                                                                                                                                                                                                              short_question                                               tags  semantic_score  tag_jaccard  bm25_score  blended_score
                             pregnant  unprotected sex the 7th period due the 26th but days later there was brown and red spotting when i wiped i took a pregnancy test the following week but it was negative am i pregnant ['period' 'spotting' 'pregnancy test' 'pregnancy']        0.815224         0.75   20.399025       0.808054
can you have your period and miss it the next month we had unprotected sex on my very fertile day we have not been together since then the test said no but i missed my period last month and i have not had a my period yet                             ['period' 'fertility']        0.885700         0.25   20.786913       0.753551
can you have your period and miss it the next month we had unprotected sex on my very fertile day we have not been together since then the test said no but i missed my period last month and i have not had a my period yet                             ['period' 'fertility']        0.885700         0.25   20.786913       0.753551
                                                                                    the first day of my last period was 110<negative_smiley>12 and today is 121312 no period when can i take a pregnancy test  am i pregnant            ['period' 'pregnancy' 'pregnancy test']        0.693769         1.00   16.154121       0.750029
                                                                                    the first day of my last period was 110<negative_smiley>12 and today is 121312 no period when can i take a pregnancy test  am i pregnant            ['period' 'pregnancy' 'pregnancy test']        0.693769         1.00   16.154121       0.750029

[Timing] Wrote timings JSON to: /content/drive/My Drive/Colab Notebooks/Harvard/CSCI E-89B NLP/Data/Medical Chatbot Dataset/retrieval_timings_bm25.json

=== Runtime (seconds) ===
       preprocess_s: 128.5489
        vectorize_s: 109.4065
    total_runtime_s: 246.7312
```

### 1. Data preprocessing

I still use the same techniques of data preprocessing and NLP as in v1.

---

### 2. NLP feature engineering

The log summarises the representation succinctly as:

> `[Features] BERT(all-MiniLM-L6-v2)→PCA(80D, EV=0.721) + Tags→SVD(20D, EV=0.245)`
> `[Shapes] X_sem_norm: (47491, 80), X_tag_norm: (47491, 20)`

On the **text (semantic)** side, I use SentenceTransformers with `all-MiniLM-L6-v2`. Each POS-filtered question is embedded into a 384-dimensional vector, which I then compress with PCA down to **80 dimensions**, capturing about **72.1% explained variance**. That gives me an 80D semantic representation per question, collected in `X_sem_norm` with shape `(47491, 80)`. I treat this as a good trade-off between memory/compute efficiency and preserving enough semantic structure for **cosine similarity** to remain meaningful.

On the **tag** side, I map tags into a bag-of-tags representation and then apply Truncated SVD to **20 dimensions**, with around **24.5% explained variance**. The resulting 20D vectors in `X_tag_norm` are more about compressing a discrete, sparse space than about reconstructing variance perfectly; SVD here gives me a smoothed co-occurrence structure between tags and a compact representation I can reuse for clustering or other downstream models. Even though the EV looks lower than for BERT, the tag space is categorical and sparse, so SVD's role is more about latent structure than exact reconstruction.

In the big picture, each question is now backed by three complementary views: an **80D semantic vector** (from BERT+PCA), a **20D tag vector** (from tag SVD), and the raw cleaned text used for BM25. That gives me lexical, semantic, and categorical channels to combine in the retrieval model.

---

### 3. Retrieval model: hybrid semantic + lexical + tags

The printed examples in the log make the scoring logic clear. For a **text-only query** such as "period late cramps," I see `semantic_score` values around 0.78-0.85, which are cosine similarities between the query embedding and `X_sem_norm`, and `bm25_score` values around 12-20, which are classical sparse lexical scores. The `blended_score` for the top hits sits around 0.61-0.64. That tells me I am computing a semantic similarity (cosine) between the query's BERT embedding and all question embeddings, a BM25 similarity over `short_question_norm`, and then combining them into a **normalized blended score**, conceptually something like
[
\text{score} = \alpha \cdot \text{semantic_norm} + \beta \cdot \text{bm25_norm}.
]
The results in this mode look exactly as I would hope: questions about sore breasts, bad cramps, missing or late periods, and "could I be pregnant?" rise to the top, showing both strong semantic and lexical overlap. BM25 emphasizes exact terms like "period" and "cramps," while BERT helps surface paraphrases and variants.

When I run a **tags-only query**, for example with `['pregnancy', 'period']`, the logs show `semantic_score = 0.0` and `bm25_score = 0.0` for all retrieved rows. In other words, for a pure tag-based query I intentionally turn off the text and semantic components and rely entirely on **set overlap between query tags and document tags**. The `tag_jaccard` values range from 1.0 (perfect tag match) down to 0.5, 0.25, and so on, and the `blended_score` tracks that pattern. The Jaccard similarity is just
[
J(A, B) = \frac{|A \cap B|}{|A \cup B|},
]
and this simple behavior is actually a feature: it is transparent, easy to reason about, and very useful when a doctor or another component has already identified relevant tags and wants to search primarily on that basis.

In the **mixed query** regime, where I have both a text description (for example, about late periods) and tags like `['pregnancy', 'period']`, the scoring combines all three signals: semantic cosine, lexical BM25, and tag Jaccard. In the logs I see `semantic_score` around 0.69-0.89, `tag_jaccard` in the 0.75-1.0 range, `bm25_score` around 16-21, and `blended_score` roughly 0.75-0.81. The top-ranked questions are nuanced "am I pregnant?" scenarios involving late periods, spotting, pregnancy tests, and sex timing, all tightly aligned with the query intent. That's the behavior I want for a medically oriented chatbot retrieval backend: semantic similarity to capture meaning, lexical scoring to anchor exact phrases, and tag overlap to enforce domain-specific intent.

---

### 4. Measurement and metrics

At this v2 stage, I am still evaluating **retrieval quality primarily by inspection** rather than with a fully labeled IR benchmark. For the text-only "period late cramps" query, the neighbors clearly focus on sore breasts, bad cramps, late or missing periods, and pregnancy concerns. For the tags-only `['pregnancy', 'period']` query, the neighbors reliably carry both "pregnancy" and "period" tags with high Jaccard scores. For mixed queries, the system surfaces a consistent "could I be pregnant?" neighborhood, including timing of intercourse, test results, and symptom details. So even in the absence of precision/recall numbers, the **face validity** of the top results is strong.

On the runtime side, I log and store timings. The JSON shows

* `preprocess_s ≈ 128.55` seconds,
* `vectorize_s ≈ 109.41` seconds,
* `total_runtime_s ≈ 246.73` seconds.

That means the full v2 build over roughly 47k questions runs in about four minutes, with around 129 seconds spent on preprocessing (POS, normalization, tag cleanup) and 109 seconds on vectorization (BERT inference, PCA, SVD, BM25 index construction). These timings are written to a JSON file, so I can compare v1 vs v2 vs future versions and quantify how architectural changes (for example, larger BERT, higher PCA dimension, or BM25 tweaks) affect both computation and quality. What I currently do not log is per-query latency and explicit IR metrics like **Precision@k**, **Recall@k**, or **nDCG@k**, which would require a labeled relevance set. That would be a natural next step if I move toward deployment.

---

### 5. Overall evaluation of v2

When I step back, v2 looks like a solid **hybrid retrieval** system with some clear strengths and some obvious directions for refinement. On the preprocessing side, POS cleanup preserves almost all rows while removing noise, and the tags are normalized into a usable structure for both filtering and similarity. On the representation side, BERT+PCA to 80 dimensions keeps about 72% of semantic variance with a much smaller footprint, and tag SVD to 20 dimensions provides a compact latent view of the tag space that I can leverage for clustering or future dense tag similarity. Conceptually I now have three aligned channels: **semantic**, **lexical**, and **tag-based**.

The retrieval model itself combines these signals in a way that works across different query styles. In text-only mode I get dense semantic search augmented by BM25, in tags-only mode I fall back to pure Jaccard over tag sets, and in mixed mode I blend all three. The qualitative behavior across the examples is exactly what I would expect in a medical QA setting. I also like that the pipeline now logs runtime metrics to JSON: the ~247 second end-to-end runtime gives me a clear baseline for future optimizations.

There are, however, some limitations and obvious next ideas. Reducing BERT from 384 to 80 dimensions loses about 28% of semantic variance; if I need finer distinctions, I might push the PCA target up to 100 or 128 dimensions and see whether retrieval neighborhoods become tighter. For tag SVD, keeping only 24.5% of variance may be too conservative for some tasks; I could either increase to 32-50 dimensions or simply rely on the raw tag sets, which I already use via Jaccard, for the retrieval path. Another gap is evaluation: I rely on qualitative inspection instead of a **gold relevance set**, so building a small labeled set of query-relevant pairs would let me compute Precision@5, Precision@10, and nDCG and directly measure the contribution of BERT, BM25, and tags. Finally, I do see duplicate rows occasionally in the output; that hints at duplicates either in the raw dataset or in the candidate merging logic, and I probably want a simple deduplication step to keep the UX cleaner.

Overall, though, v2 feels like a meaningful step up from a pure dense or pure lexical system. It is now a **hybrid semantic-lexical-tag reranker**, with good coverage, interpretable behavior, and a runtime profile that I can iterate on.

# Comparison between Similarity Searcha nd Reranking Models

### 1. Big-picture comparison

At a high level, v1 is a **semantic + tag** system, while v2 evolves into a **semantic + tags + BM25** hybrid. Both versions use the same POS-filtered BERT (MiniLM) representation with PCA for text, and both keep tag SVD plus a Jaccard-based tag overlap in scoring. The real shift is that v2 explicitly injects a **lexical signal** through BM25 built on `short_question_norm`. In v1, candidate generation is driven purely by top-K semantic cosine; in v2, candidates come from the **union** of top-K dense semantic neighbours and top-K BM25 neighbours. Reranking in v1 uses semantic cosine blended with tag Jaccard; v2 expands that to a three-way blend: semantic cosine, tag Jaccard, and a normalized BM25 score.

The query-side NLP also diverges. In v1, I always run POS-filtering on the query text before sending it to BERT, which keeps query and corpus strictly symmetric but makes each query relatively heavy. In v2, I keep the same POS-filtered backbone for the corpus but make query POS optional via a `USE_SPACY_FOR_QUERY` flag, so I can choose between fidelity and latency. Offline runtime reflects the added complexity: v1 takes roughly 218 seconds for preprocessing plus vectorization, while v2 sits around 247 seconds because of the extra BM25 index build and slightly heavier vectorization. Conceptually, I think of v1 as a clean **semantic + intent** engine and v2 as a more pragmatic **hybrid** stack that brings semantic, lexical, and tag-based signals together.

---

### 2. NLP and preprocessing logic

Under the hood, both v1 and v2 share the same NLP backbone for the corpus. I normalize raw text into `short_question_norm`, parse tags into `tag_list`, and build a POS-filtered view `short_question_pos` using spaCy by keeping NOUN and PROPN plus a small list of medically important short tokens like "hpv" or "uti." Rows that have neither POS content nor tags get dropped; in practice that leaves **47,491** questions out of about 47,603, so the preprocessing is aggressive at the token level but conservative about row-level coverage. This means both versions operate on a medically focused, noun-heavy representation centered on conditions, body parts, drugs, and symptoms.

The real change starts on the **query side**. In v1, every query goes through the full stack: normalized text, POS filter, BERT, PCA, scaling, and L2 normalization. That gives strict symmetry and focuses the query embedding on content nouns, which is useful when the raw query is chatty. The downside is cost: spaCy is not cheap if I want to handle many queries. In v2, the corpus remains the same, but I introduce a control knob: if `USE_SPACY_FOR_QUERY` is true, everything behaves exactly like v1; if it is false, the query is just normalized and sent directly into BERT without POS filtering. That gives me a spectrum from "maximal NLP" to "lighter, faster queries," while keeping the document representations unchanged.

---

### 3. Modeling logic and retrieval behavior

In v1, the retrieval flow is built around **dense semantics plus tags**. On the index side, I create a semantic tower by feeding `short_question_pos` to BERT, applying PCA to 80 dimensions, scaling and L2-normalizing to get `X_sem_norm`. In parallel, I build a tag tower by mapping `tag_list` with MultiLabelBinarizer, applying SVD to 20 dimensions, scaling and normalizing to get `X_tag_norm`. I also keep raw `tag_sets` as Python sets for each question. At query time, I encode the text with the same POS→BERT→PCA→scale→L2 pipeline, and, if tags are provided, I encode them through the tag SVD and keep a query tag set. Candidate generation is purely dense: I compute cosine similarities `X_sem_norm @ q_sem_norm` and take the top `candidate_k` indices. Reranking within that semantic neighbourhood uses tag Jaccard, and I form a blended score as a weighted sum of semantic cosine and Jaccard overlap. Text-only queries behave as pure dense retrieval; tag-only queries effectively reduce to tag Jaccard ranking; mixed queries balance both signals. In my mind, v1 is a **two-signal reranker**: meaning via BERT and intent via tags.

v2 keeps most of this structure but adds BM25 as a third leg. In addition to `X_sem_norm`, `X_tag_norm`, and `tag_sets`, I build a BM25 index over tokenized `short_question_norm`. On the query side, I still compute `q_sem_norm` and optional `q_tag_set`, but I also tokenize the normalized query into `bm25_query_tokens`. Candidate selection becomes a two-source process: I get top-K candidates by semantic cosine and separately top-K by BM25 score, then take the **union** of those indices as the candidate pool. For this union, I compute semantic scores, BM25 scores (which I normalize to [0,1]), and tag Jaccard. The blended score is now a three-term combination: `α * semantic + β * tag_jaccard + γ * bm25_norm`, with default weights like 0.6, 0.2, 0.2.

This changes the behavior in subtle but important ways. Text-only queries no longer rely solely on dense embeddings; they also benefit from exact phrase and token matches through BM25, which helps when phrasing matters. Tag-only queries still work because both semantic and BM25 stay effectively at zero, leaving tag Jaccard to dominate. Mixed text+tag queries become richer: semantic similarity brings in paraphrases, BM25 rewards exact wording and rare terms, and tags enforce high-level intent. To me, v1 is "dense semantic retrieval with tag-aware reranking," whereas v2 feels like a more realistic **hybrid semantic-lexical-tag** retrieval stack, closer to what I would expect in a production RAG system.

---

### 4. Performance, metrics, and qualitative behavior

From the runtime logs, v1 and v2 are in the same ballpark, but v2 is slightly heavier. For v1, preprocessing takes about 123.6 seconds and vectorization about 92.7 seconds, for a total of ~218 seconds. For v2, preprocessing is around 128.5 seconds and vectorization about 109.4 seconds, with a total near 246.7 seconds. The extra ~30 seconds in v2 come mainly from building the BM25 index and the additional tag/BM25 handling. Given that everything runs over ~47k documents, I treat both as cheap offline jobs, and the per-query latency remains dominated by the BERT forward pass; BM25 scoring itself is relatively fast.

Both versions track some structural metrics like PCA explained variance (around 0.721 for the 80D BERT representation) and tag SVD explained variance (around 0.245 for 20D). That reassures me that dimensionality reduction hasn't become so aggressive that it destroys semantic structure. Qualitatively, the behaviors diverge in predictable ways. For a pure text query, v1's neighbors are already clinically coherent; v2 tends to keep the same "type" of results but adjusts the ordering, sometimes promoting examples with stronger phrase-level overlap. For tags-only queries, both versions behave similarly, since the blended score is essentially proportional to tag Jaccard. For mixed queries, v2's ranking often becomes more nuanced: documents with strong semantic relevance and good BM25 scores sit at the top, and tags resolve ties and prioritize intent-consistent examples. Without an annotated ground truth, I mainly rely on this visual and textual inspection; in that sense, both v1 and v2 look solid, but v2 gives me more control through the added lexical axis.

---

### 5. Conceptual insights about v1 vs v2

When I compare the two versions conceptually, the **NLP design philosophy** is where I notice the biggest difference. In v1, the mindset is essentially: "I will build the cleanest possible semantic representation, tightly focused on medical nouns, and let that plus tags drive everything." The representation is carefully crafted, and retrieval lives almost entirely in embedding space, with tags as a light supervisory overlay. In v2, I keep that careful semantic backbone but acknowledge that real systems benefit from a separate lexical channel. I still lean hard on BERT and tags for semantic structure, but I add BM25 over lightly normalized text (not POS-filtered text), and I give myself the option to relax POS processing on queries. In short, v1 is **semantic-first**, while v2 is deliberately **hybrid-first**.

From a modeling perspective, v1 treats the dense embedding space as the single place where similarity is defined. Tags can reorder neighbors, but they don't change which candidates are seen. In v2, similarity becomes a three-way compromise among semantic alignment, lexical exactness, and tag intent overlap, and either semantic or lexical signals can bring candidates into the pool before tags refine the ranking. That feels closer to how I would design a robust RAG pipeline: allow both semantics and exact keywords to "trigger" candidates and then layer domain knowledge on top.

Operationally, the trade-off is straightforward. v1 is slightly lighter and simpler, which is attractive for pure embedding-based exploration, clustering, or label propagation. v2 is slightly more complex and a bit slower offline, but better suited to end-user retrieval scenarios where exact phrases and rare terms matter, where I might plug this into a chatbot, or where I want more resilience across different query styles. In many ways, v1 is a clean research baseline; v2 is what I would actually roll into a more realistic system.

---

### 6. One-paragraph summary

If I had to summarize the difference in one paragraph, I would say: v1 is my **dense BERT + tags engine**, designed as a clean, mathematically elegant similarity model operating in a POS-filtered medical semantic space, with tags acting as a secondary reranking signal. v2 takes that same backbone and turns it into a more pragmatic **hybrid retrieval stack** by adding BM25 and a configurable query NLP path. The corpus representation stays almost identical, but v2 acknowledges that lexical matching matters for real queries and that query-time POS is a performance trade-off, not an absolute requirement. The cost is about 10-15% extra offline runtime and a few more hyperparameters (weights and candidate sizes), but the gain is an additional, orthogonal lexical signal that lets me control how strongly exact terms versus global semantics drive the final neighborhoods and tag-aware reranking.

---

# **Reference List for Clustering, NLP, EDA, Topic Modelling, and Retrieval**

#### **Core NLP Preprocessing and POS Filtering**

1. Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. *spaCy 3: Industrial-Strength Natural Language Processing in Python* (2021).
   [https://spacy.io/](https://spacy.io/)
2. spaCy Documentation - Tokenization, Lemmatization, POS Tagging.
   [https://spacy.io/usage/linguistic-features](https://spacy.io/usage/linguistic-features)
3. Bird, S., Klein, E., & Loper, E. *Natural Language Processing with Python*. O'Reilly Media (2009).
   [https://www.nltk.org/book/](https://www.nltk.org/book/)

---

#### **TF-IDF, Dimensionality Reduction (LSA/SVD/PCA), and Classical Clustering**

4. Manning, C., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*. Cambridge University Press (2008). (TF-IDF, cosine similarity, sparse vectors)
   [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)
5. Deerwester, S. et al. *Indexing by Latent Semantic Analysis*. JASIS (1990).
6. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR (2011). (KMeans, MiniBatchKMeans, PCA, TruncatedSVD)
   [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

---

#### **UMAP & HDBSCAN (used heavily in v2.x and EDA visualization)**

7. McInnes, L., Healy, J., & Melville, J. *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426 (2018).
   [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
8. Campello, R. J. G. B., Moulavi, D., Sander, J. *Advances in Density-Based Clustering: HDBSCAN*. J. Intell. Inf. Syst. (2015).
   [https://hdbscan.readthedocs.io/](https://hdbscan.readthedocs.io/)
9. McInnes, L. *UMAP + HDBSCAN Examples*.
   [https://umap-learn.readthedocs.io/en/latest/clustering.html](https://umap-learn.readthedocs.io/en/latest/clustering.html)

---

#### **OPTICS (Versions v2.1.3-v2.3.3)**

10. Ankerst, M. et al. *OPTICS: Ordering Points To Identify the Clustering Structure*. SIGMOD (1999).
11. Scikit-learn OPTICS Documentation.
    [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)

---

#### **Spectral Clustering + Nyström Approximation (v2.4.3)**

12. Ng, A. Y., Jordan, M. I., Weiss, Y. *On Spectral Clustering: Analysis and an Algorithm*. NIPS (2001).
13. Drineas, P., Mahoney, M. *Nyström Methods and Their Use in Large-Scale Machine Learning* (2016).
14. Scikit-Learn SpectralEmbedding.
    [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html)

---

#### **Sentence Embeddings & Transformer Models (v2.x feature stack)**

15. Reimers, N., & Gurevych, I. *Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks*. EMNLP (2019).
    [https://www.sbert.net/](https://www.sbert.net/) ([SentenceTransformers][1])
16. HuggingFace Transformers Documentation.
    [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)

---

#### **BERTopic (v5 in my proposal; also inspires UMAP/HDBSCAN setup)**

17. Grootendorst, M. *BERTopic: Neural Topic Modeling with Transformers*.
    [https://maartengr.github.io/BERTopic/](https://maartengr.github.io/BERTopic/)
18. BERTopic GitHub Repository.
    [https://github.com/MaartenGr/BERTopic](https://github.com/MaartenGr/BERTopic)

---

#### **Vector Search, Similarity Search, Retrieval, and Reranking (v1 & v2-style hybrid stack)**

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

#### **Data Cleaning, Heavy-Tailed Distributions, and Tag Noise**

29. Sculley, D. *Web-Scale High-Dimensional Text Clustering* (2007).
30. Mikolov, T. et al. *Linguistic Regularities in Continuous Space Word Representations*.

---

#### **General Machine Learning & Clustering Theory (writing support)**

31. Bishop, C. *Pattern Recognition and Machine Learning*. Springer (2006).
32. Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning*. Springer (2009).
33. Manning, C., & Schütze, H. *Foundations of Statistical Natural Language Processing*. MIT Press (1999).

---

#### **v1.1.3-v1.4.3 (LSA + KMeans family)**

References: 4, 5, 6, 12, 25, 31, 32

#### **v2.1.3-v2.3.3 (BERT-Hybrid + OPTICS / HDBSCAN)**

References: 7, 8, 9, 10, 11, 15, 16

#### **v2.4.3 (BERT-Hybrid + Nyström Spectral + KMeans)**

References: 12, 13, 14, 15, 16, 19, 20, 21

#### **POS filtering + NLP normalization**

References: 1, 2, 3

#### **TF-IDF, tags SVD, sparse matrices**

References: 4, 5, 6

#### **UMAP & EDA visualization**

References: 7, 8, 9

#### **Similarity search, hybrid retrieval, and reranking (v1 & v2)**

References: 15, 16, 19-28













