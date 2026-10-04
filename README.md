# OWL-GPS

### Adapting Actively on the Fly: Relevance-Guided Online Meta-Learning with Latent Concepts for Geospatial Discovery

**NeurIPS 2026**

[Jowaria Khan](https://jowariak.github.io/) · Anindya Sarkar · Yevgeniy Vorobeychik · Elizabeth Bondi-Kelly

[🌐 Project Page](https://jowariak.github.io/OWL-GPS/) · [📄 Paper](https://jowariak.github.io/OWL-GPS/paper.pdf)

---

## Overview

Environmental discovery often requires learning from sparse observations while deciding where to collect new measurements under a limited acquisition budget.

**OWL-GPS** is a relevance-guided online learning framework that jointly performs sequential acquisition and online adaptation. It learns a region-specific relevance representation over environmental concepts and uses this shared representation both to decide which regions to query next and to guide online model updates.

The resulting loop is:

**represent context → estimate relevance → acquire → adapt → recompute relevance → acquire again**

We evaluate OWL-GPS primarily on real-world PFAS contamination discovery and additionally study its portability on a sparsified land-cover task.

---

See the [project page](https://jowariak.github.io/OWL-GPS/) and paper for full results.

