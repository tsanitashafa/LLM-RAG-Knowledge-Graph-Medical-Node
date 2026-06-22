# LLM-RAG Knowledge Graph Medical Node

This project implements a medical knowledge graph using Neo4j, Neo4j Graph Data Science, Python, Streamlit, and LLM integration through OpenRouter. The graph contains medical entities such as diseases, symptoms, drugs, and categories. The system supports Text-to-Cypher, Text-to-Graph Builder, Graph RAG, and graph analytics.

---

## 1. Features

* Build a medical knowledge graph from Wikidata and DBpedia data.
* Store disease, symptom, drug, and category relationships in Neo4j.
* Run graph analytics using Neo4j Graph Data Science.
* Translate natural language questions into Cypher queries using LLM.
* Extract entities and relationships from text using LLM Graph Builder.
* Generate Graph RAG answers based on Neo4j retrieval results.
* Provide a Streamlit interface for demo and execution.

---

## 2. Technology Stack

| Component       | Tool                     |
| --------------- | ------------------------ |
| Database        | Neo4j Desktop v5.x       |
| Graph Analytics | Neo4j Graph Data Science |
| Language        | Python                   |
| Query Language  | Cypher                   |
| LLM Provider    | OpenRouter               |
| UI              | Streamlit                |
| Dataset         | Wikidata and DBpedia     |

---

## 3. Dataset and Graph Schema

Dataset files are stored in the `data/` folder.

Main entities:

* Disease / Penyakit
* Symptom / Gejala
* Drug / Obat
* Category / Kategori

Main relationships:

```text
(Penyakit)-[:MEMILIKI_GEJALA]->(Gejala)
(Penyakit)-[:DIOBATI_DENGAN]->(Obat)
(Penyakit)-[:TERMASUK_KATEGORI]->(Kategori)
```

Additional properties and relationships from graph analytics:

```text
clusterId        : result of Louvain community detection
betweenness      : result of Betweenness Centrality
MIRIP_DENGAN     : result of Node Similarity
similarity       : similarity score
```

---

## 4. Project Structure

```text
LLM-RAG-Knowledge-Graph-Medical-Node/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── important_cypher_queries.csv
│
├── data/
│   ├── dbpedia.csv
│   ├── df_final.csv
│   └── wikidata.csv
│
└── scripts/
    ├── text_to_cypher_openrouter.py
    ├── graph_builder_openrouter.py
    ├── graph_rag_openrouter.py
    └── test_connection.py
```

---

## 5. Installation and Configuration

Clone the repository:

```bash
git clone https://github.com/tsanitashafa/LLM-RAG-Knowledge-Graph-Medical-Node.git
cd LLM-RAG-Knowledge-Graph-Medical-Node
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

The `.env` file is not uploaded to GitHub because it contains private credentials.

---

## 6. How to Run

Make sure Neo4j Desktop is running, then run:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

## 7. Main Functions

### 7.1 Text-to-Cypher

The LLM translates a natural language question into a Cypher query, then the query is executed in Neo4j.

Example question:

```text
What diseases are similar to malaria based on the graph?
```

Example generated Cypher:

```cypher
MATCH (p1:Penyakit)-[s:MIRIP_DENGAN]-(p2:Penyakit)
WHERE toLower(p1.nama) CONTAINS "malaria" AND p1 <> p2
RETURN p1.nama AS Disease_A,
       p2.nama AS Similar_Disease,
       s.similarity AS Similarity
ORDER BY Similarity DESC
LIMIT 20
```

---

### 7.2 Text-to-Graph Builder

The LLM extracts entities and relationships from unstructured text, then inserts the result into Neo4j.

Example input:

```text
Malaria is a disease that has symptoms such as fever, chills, and headache.
Malaria can be treated with chloroquine.
```

Expected output:

```text
Malaria - MEMILIKI_GEJALA - fever
Malaria - MEMILIKI_GEJALA - chills
Malaria - MEMILIKI_GEJALA - headache
Malaria - DIOBATI_DENGAN - chloroquine
```

---

### 7.3 Graph RAG

Graph RAG retrieves data from Neo4j and sends the retrieval result back to the LLM. The final answer is generated based on graph data, not only from the model’s general knowledge.

Pipeline:

```text
User Question
→ LLM generates Cypher
→ Neo4j retrieves graph data
→ Retrieved graph result is sent to LLM
→ LLM generates final answer
```

---

## 8. Graph Analytics

This project uses Neo4j Graph Data Science for graph analysis.

Implemented methods:

1. Louvain Community Detection
   Used to generate `clusterId`.

2. Betweenness Centrality
   Used to identify important connector nodes in the graph.

3. Node Similarity
   Used to create `MIRIP_DENGAN` relationships with `similarity` score.

Important Cypher queries are documented in:

```text
important_cypher_queries.csv
```

---

## 9. AI Usage Documentation

AI was used to assist with:

* Python code generation
* Streamlit UI development
* Text-to-Cypher prompt design
* Text-to-Graph Builder prompt design
* Graph RAG prompt design
* Cypher query examples
* README and documentation formatting

The generated code was manually modified, tested, and adjusted based on the Neo4j schema and assignment requirements.

Model used:

```text
OpenRouter model: openai/gpt-4o-mini
```

---

## 10. Screenshot Documentation

The required screenshots include:

1. Neo4j database connection
2. Query result or graph builder output
3. Graph analytics / ML output
4. LLM demo: Text-to-Cypher, Graph RAG, or Cypher translation

Screenshots can be stored in:

```text
screenshots/
```

---

## 11. Repository URL

```text
https://github.com/tsanitashafa/LLM-RAG-Knowledge-Graph-Medical-Node
```

---

## 12. Video Demonstration

YouTube video URL:

```text
[Insert YouTube video URL here]
```

Video content:

```text
1 minute  : Background and architecture
2 minutes : Code execution and result demo
1 minute  : Graph analysis / insight
1 minute  : Conclusion and lessons learned
```

---

## 13. Conclusion

This project implements a medical knowledge graph with Neo4j and LLM integration. The system supports graph construction, graph analytics, natural language query translation, graph building from text, and Graph RAG through a Streamlit interface.
