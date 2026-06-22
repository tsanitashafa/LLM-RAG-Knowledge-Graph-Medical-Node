import os
import re
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# =========================
# CONFIG
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SCHEMA = """
Node labels:
- Penyakit {qid, nama, clusterId, betweenness}
- Gejala {nama, clusterId, betweenness}
- Obat {nama, clusterId, betweenness}
- Kategori {nama, clusterId, betweenness}

Relationships:
- (Penyakit)-[:MEMILIKI_GEJALA]->(Gejala)
- (Penyakit)-[:DIOBATI_DENGAN]->(Obat)
- (Penyakit)-[:TERMASUK_KATEGORI]->(Kategori)

Important:
- clusterId is the result of GDS Louvain community detection.
- betweenness is the result of GDS Betweenness Centrality.
- If the user asks about cluster/community/kelompok, use clusterId.
- If the user asks about node paling berpengaruh, paling penting, penghubung, atau centrality, use betweenness.
- Cluster is NOT the same as Kategori.
"""

FORBIDDEN_KEYWORDS = [
    "CREATE",
    "DELETE",
    "DETACH",
    "SET",
    "MERGE",
    "DROP",
    "REMOVE",
    "LOAD CSV",
    "CALL",
    "APOC",
    "GDS",
]

ALLOWED_ENTITY_TYPES = {"Penyakit", "Gejala", "Obat", "Kategori"}

ALLOWED_RELATION_TYPES = {
    "MEMILIKI_GEJALA",
    "DIOBATI_DENGAN",
    "TERMASUK_KATEGORI",
}


# =========================
# STYLE
# =========================
st.set_page_config(
    page_title="Medical Knowledge Graph LLM",
    page_icon="🧬",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# OPENROUTER
# =========================
def call_openrouter(messages, temperature=0, max_tokens=1000):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY belum diisi di file .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}:\n{response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


# =========================
# NEO4J
# =========================
@st.cache_resource
def get_driver():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    return driver


def run_cypher(cypher):
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher)
        return [record.data() for record in result]


def run_write_cypher(cypher, params=None):
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(cypher, params or {})


def get_graph_summary():
    node_query = """
    MATCH (n)
    RETURN labels(n)[0] AS label, count(n) AS jumlah
    ORDER BY jumlah DESC
    """

    rel_query = """
    MATCH ()-[r]->()
    RETURN type(r) AS relasi, count(r) AS jumlah
    ORDER BY jumlah DESC
    """

    nodes = run_cypher(node_query)
    rels = run_cypher(rel_query)

    return nodes, rels


# =========================
# COMMON UTILITIES
# =========================
def clean_cypher(text):
    text = text.strip()

    text = re.sub(r"^```cypher", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    if text.endswith(";"):
        text = text[:-1]

    return text.strip()


def is_safe_cypher(cypher):
    upper_query = cypher.upper().strip()

    if not upper_query.startswith("MATCH"):
        return False, "Query harus diawali MATCH."

    if "RETURN" not in upper_query:
        return False, "Query harus memiliki RETURN."

    if "LIMIT" not in upper_query:
        return False, "Query harus memiliki LIMIT."

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in upper_query:
            return False, f"Query mengandung keyword yang tidak diizinkan: {keyword}"

    return True, "Aman"


def display_results(results):
    if not results:
        st.warning("Tidak ada hasil dari graph.")
        return

    df = pd.DataFrame(results)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
        )

    st.dataframe(df, use_container_width=True)


# =========================
# TEXT TO CYPHER
# =========================
def build_cypher_prompt(question):
    return f"""
You are a Neo4j Cypher generator.

Convert the user's natural language question into a valid read-only Cypher query.

Graph schema:
{SCHEMA}

Rules:
- Return Cypher query only.
- Use MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, collect, count, DISTINCT only.
- Do not use CREATE, DELETE, SET, MERGE, DROP, REMOVE, LOAD CSV, CALL, APOC, or GDS.
- Use toLower() and CONTAINS for flexible text search.
- If the question asks about cluster/community/kelompok, use the clusterId property.
- Cluster means GDS Louvain result stored in clusterId, not Kategori.
- Always add LIMIT 20 unless the query already uses LIMIT 1 for top result.
- Do not explain the query.
- Return only the Cypher query, without markdown code block.

Examples:

Question: penyakit dengan gejala fever
Cypher:
MATCH (p:Penyakit)-[:MEMILIKI_GEJALA]->(g:Gejala)
WHERE toLower(g.nama) CONTAINS "fever"
RETURN p.nama AS Penyakit, g.nama AS Gejala
LIMIT 20

Question: satu cluster dengan penyakit terbanyak dan sebutkan nama penyakitnya
Cypher:
MATCH (p:Penyakit)
WHERE p.clusterId IS NOT NULL
WITH p.clusterId AS cluster, collect(DISTINCT p.nama) AS daftarPenyakit, count(DISTINCT p) AS jumlahPenyakit
ORDER BY jumlahPenyakit DESC
LIMIT 1
RETURN cluster, jumlahPenyakit, daftarPenyakit[0..20] AS namaPenyakit

Question: penyakit apa yang mirip dengan malaria berdasarkan graph?
Cypher:
MATCH (p1:Penyakit)-[s:MIRIP_DENGAN]-(p2:Penyakit)
WHERE toLower(p1.nama) CONTAINS "malaria" AND p1 <> p2
RETURN p1.nama AS Penyakit_A,
       p2.nama AS Penyakit_Mirip,
       s.similarity AS Similarity
ORDER BY Similarity DESC
LIMIT 20

User question:
{question}
""".strip()


def generate_cypher(question):
    messages = [
        {
            "role": "system",
            "content": "You generate safe read-only Neo4j Cypher queries only."
        },
        {
            "role": "user",
            "content": build_cypher_prompt(question)
        }
    ]

    raw = call_openrouter(messages, temperature=0, max_tokens=700)
    return clean_cypher(raw)


# =========================
# GRAPH BUILDER
# =========================
def build_graph_extraction_prompt(text):
    return f"""
You are an information extraction system for building a Neo4j medical knowledge graph.

Extract entities and relationships from the text.

Allowed entity types:
- Penyakit
- Gejala
- Obat
- Kategori

Allowed relationship types:
- MEMILIKI_GEJALA: Penyakit -> Gejala
- DIOBATI_DENGAN: Penyakit -> Obat
- TERMASUK_KATEGORI: Penyakit -> Kategori

Rules:
- Return valid JSON only.
- Do not use markdown.
- Do not explain.
- Use English labels if the text is English.
- If an entity is not found, return an empty array.
- Do not invent information outside the text.

JSON format:
{{
  "entities": [
    {{"type": "Penyakit", "name": "example disease"}},
    {{"type": "Gejala", "name": "example symptom"}},
    {{"type": "Obat", "name": "example drug"}},
    {{"type": "Kategori", "name": "example category"}}
  ],
  "relationships": [
    {{"source": "example disease", "source_type": "Penyakit", "relation": "MEMILIKI_GEJALA", "target": "example symptom", "target_type": "Gejala"}},
    {{"source": "example disease", "source_type": "Penyakit", "relation": "DIOBATI_DENGAN", "target": "example drug", "target_type": "Obat"}},
    {{"source": "example disease", "source_type": "Penyakit", "relation": "TERMASUK_KATEGORI", "target": "example category", "target_type": "Kategori"}}
  ]
}}

Text:
{text}
""".strip()


def clean_json_response(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def normalize_name(name):
    return str(name).strip()


def validate_graph_data(graph_data):
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])

    clean_entities = []
    seen_entities = set()

    for entity in entities:
        entity_type = entity.get("type")
        name = normalize_name(entity.get("name", ""))

        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue
        if not name:
            continue

        key = (entity_type, name.lower())
        if key not in seen_entities:
            clean_entities.append({
                "type": entity_type,
                "name": name
            })
            seen_entities.add(key)

    clean_relationships = []
    seen_relationships = set()

    for rel in relationships:
        source = normalize_name(rel.get("source", ""))
        target = normalize_name(rel.get("target", ""))
        source_type = rel.get("source_type")
        target_type = rel.get("target_type")
        relation = rel.get("relation")

        if not source or not target:
            continue
        if source_type not in ALLOWED_ENTITY_TYPES:
            continue
        if target_type not in ALLOWED_ENTITY_TYPES:
            continue
        if relation not in ALLOWED_RELATION_TYPES:
            continue

        if relation == "MEMILIKI_GEJALA":
            if not (source_type == "Penyakit" and target_type == "Gejala"):
                continue

        if relation == "DIOBATI_DENGAN":
            if not (source_type == "Penyakit" and target_type == "Obat"):
                continue

        if relation == "TERMASUK_KATEGORI":
            if not (source_type == "Penyakit" and target_type == "Kategori"):
                continue

        key = (source.lower(), relation, target.lower())
        if key not in seen_relationships:
            clean_relationships.append({
                "source": source,
                "source_type": source_type,
                "relation": relation,
                "target": target,
                "target_type": target_type
            })
            seen_relationships.add(key)

    return {
        "entities": clean_entities,
        "relationships": clean_relationships
    }


def extract_graph_from_text(text):
    messages = [
        {
            "role": "system",
            "content": "You extract structured graph data as valid JSON only."
        },
        {
            "role": "user",
            "content": build_graph_extraction_prompt(text)
        }
    ]

    raw = call_openrouter(messages, temperature=0, max_tokens=1200)
    cleaned = clean_json_response(raw)

    graph_data = json.loads(cleaned)
    return validate_graph_data(graph_data)


def create_entity(entity_type, name):
    if entity_type == "Penyakit":
        run_write_cypher(
            """
            MERGE (n:Penyakit {nama: $name})
            SET n.source = coalesce(n.source, 'LLM Graph Builder')
            """,
            {"name": name}
        )

    elif entity_type == "Gejala":
        run_write_cypher(
            """
            MERGE (n:Gejala {nama: $name})
            SET n.source = coalesce(n.source, 'LLM Graph Builder')
            """,
            {"name": name}
        )

    elif entity_type == "Obat":
        run_write_cypher(
            """
            MERGE (n:Obat {nama: $name})
            SET n.source = coalesce(n.source, 'LLM Graph Builder')
            """,
            {"name": name}
        )

    elif entity_type == "Kategori":
        run_write_cypher(
            """
            MERGE (n:Kategori {nama: $name})
            SET n.source = coalesce(n.source, 'LLM Graph Builder')
            """,
            {"name": name}
        )


def create_relationship(rel):
    relation = rel["relation"]
    source = rel["source"]
    target = rel["target"]

    if relation == "MEMILIKI_GEJALA":
        run_write_cypher(
            """
            MERGE (p:Penyakit {nama: $source})
            MERGE (g:Gejala {nama: $target})
            MERGE (p)-[:MEMILIKI_GEJALA]->(g)
            """,
            {"source": source, "target": target}
        )

    elif relation == "DIOBATI_DENGAN":
        run_write_cypher(
            """
            MERGE (p:Penyakit {nama: $source})
            MERGE (o:Obat {nama: $target})
            MERGE (p)-[:DIOBATI_DENGAN]->(o)
            """,
            {"source": source, "target": target}
        )

    elif relation == "TERMASUK_KATEGORI":
        run_write_cypher(
            """
            MERGE (p:Penyakit {nama: $source})
            MERGE (k:Kategori {nama: $target})
            MERGE (p)-[:TERMASUK_KATEGORI]->(k)
            """,
            {"source": source, "target": target}
        )


def save_graph_to_neo4j(graph_data):
    for entity in graph_data["entities"]:
        create_entity(entity["type"], entity["name"])

    for rel in graph_data["relationships"]:
        create_relationship(rel)


# =========================
# GRAPH RAG
# =========================
def build_answer_prompt(question, cypher, graph_results):
    return f"""
You are a medical knowledge graph assistant.

Answer the user's question using only the retrieved Neo4j graph results.

Rules:
- Use Indonesian language.
- Do not add medical facts outside the graph results.
- If the graph results are empty, say that the data was not found in the graph.
- Mention that the answer is based on retrieved graph data.
- Do not give medical diagnosis or treatment advice.
- Convert raw graph results into a clear explanation.
- If possible, provide:
  1. Ringkasan jawaban
  2. Data yang ditemukan dari graph
  3. Insight singkat dari hubungan antar node
- Keep the answer concise but more informative than raw query results.

User question:
{question}

Generated Cypher:
{cypher}

Neo4j graph results:
{json.dumps(graph_results, ensure_ascii=False, indent=2)}
""".strip()


def generate_rag_answer(question, cypher, graph_results):
    messages = [
        {
            "role": "system",
            "content": "You answer using only retrieved graph context."
        },
        {
            "role": "user",
            "content": build_answer_prompt(question, cypher, graph_results)
        }
    ]

    return call_openrouter(messages, temperature=0.2, max_tokens=1000).strip()


# =========================
# UI HEADER
# =========================
st.title("🧬 Medical Knowledge Graph + LLM")
st.caption("Text-to-Cypher • Text-to-Graph Builder • Graph RAG dengan Neo4j dan OpenRouter")

with st.sidebar:
    st.header("⚙️ Status Sistem")

    st.write("**Neo4j URI:**", NEO4J_URI)
    st.write("**Database:**", NEO4J_DATABASE)
    st.write("**OpenRouter Model:**", OPENROUTER_MODEL)

    if st.button("Cek Koneksi Neo4j"):
        try:
            get_driver()
            st.success("Koneksi Neo4j berhasil.")
        except Exception as e:
            st.error(f"Koneksi Neo4j gagal: {e}")

    st.divider()

    if st.button("Refresh Ringkasan Graph"):
        try:
            nodes, rels = get_graph_summary()
            st.session_state["nodes_summary"] = nodes
            st.session_state["rels_summary"] = rels
        except Exception as e:
            st.error(e)

    if "nodes_summary" in st.session_state:
        st.subheader("Node")
        st.dataframe(pd.DataFrame(st.session_state["nodes_summary"]), use_container_width=True)

    if "rels_summary" in st.session_state:
        st.subheader("Relasi")
        st.dataframe(pd.DataFrame(st.session_state["rels_summary"]), use_container_width=True)


# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "🔎 Text-to-Cypher",
    "🧱 Text-to-Graph Builder",
    "💬 Graph RAG"
])


# =========================
# TAB 1: TEXT TO CYPHER
# =========================
with tab1:
    st.subheader("🔎 Text-to-Cypher")
    st.write("LLM menerjemahkan pertanyaan natural language menjadi query Cypher, lalu query dijalankan ke Neo4j.")

    question = st.text_input(
        "Masukkan pertanyaan",
        value="penyakit dengan gejala fever",
        key="text_to_cypher_question"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        generate_button = st.button("Generate & Jalankan Cypher", type="primary", key="btn_t2c")

    if generate_button:
        try:
            with st.spinner("Menghasilkan Cypher dengan OpenRouter..."):
                cypher = generate_cypher(question)

            st.session_state["t2c_cypher"] = cypher

            valid, message = is_safe_cypher(cypher)

            st.markdown("### Generated Cypher")
            st.code(cypher, language="cypher")

            if not valid:
                st.error(message)
            else:
                with st.spinner("Menjalankan query ke Neo4j..."):
                    results = run_cypher(cypher)

                st.markdown("### Hasil Query")
                display_results(results)

        except Exception as e:
            st.error(f"Terjadi error: {e}")


# =========================
# TAB 2: TEXT TO GRAPH BUILDER
# =========================
with tab2:
    st.subheader("🧱 Text-to-Graph Builder")
    st.write("LLM mengekstraksi entitas dan relasi dari teks, lalu hasilnya disimpan sebagai node dan relationship di Neo4j.")

    sample_text = """Malaria is a disease that has symptoms such as fever, chills, and headache. Malaria can be treated with chloroquine. Tuberculosis has symptoms such as cough, fever, and weight loss. Tuberculosis can be treated with rifampicin."""

    input_text = st.text_area(
        "Masukkan teks medis",
        value=sample_text,
        height=180
    )

    build_button = st.button("Ekstrak Graph & Simpan ke Neo4j", type="primary", key="btn_graph_builder")

    if build_button:
        try:
            with st.spinner("LLM sedang mengekstraksi entitas dan relasi..."):
                graph_data = extract_graph_from_text(input_text)

            st.markdown("### Hasil Ekstraksi LLM")
            st.json(graph_data)

            with st.spinner("Menyimpan hasil graph ke Neo4j..."):
                save_graph_to_neo4j(graph_data)

            st.success("Graph berhasil ditambahkan ke Neo4j.")

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric("Jumlah Entitas Diekstrak", len(graph_data["entities"]))

            with col_b:
                st.metric("Jumlah Relasi Diekstrak", len(graph_data["relationships"]))

            st.markdown("### Ringkasan Graph Setelah Insert")
            nodes, rels = get_graph_summary()

            st.write("**Node:**")
            st.dataframe(pd.DataFrame(nodes), use_container_width=True)

            st.write("**Relasi:**")
            st.dataframe(pd.DataFrame(rels), use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error: {e}")


# =========================
# TAB 3: GRAPH RAG
# =========================
with tab3:
    st.subheader("💬 Graph RAG")
    st.write("Pertanyaan diubah menjadi Cypher, data diambil dari Neo4j, lalu LLM menyusun jawaban berdasarkan hasil retrieval graph.")

    rag_question = st.text_input(
        "Masukkan pertanyaan RAG",
        value="jelaskan penyakit malaria berdasarkan data graph, mencakup gejala dan obat yang tersedia",
        key="rag_question"
    )

    rag_button = st.button("Jalankan Graph RAG", type="primary", key="btn_rag")

    if rag_button:
        try:
            with st.spinner("Menghasilkan Cypher dari pertanyaan..."):
                cypher = generate_cypher(rag_question)

            st.markdown("### Generated Cypher")
            st.code(cypher, language="cypher")

            valid, message = is_safe_cypher(cypher)

            if not valid:
                st.error(message)
            else:
                with st.spinner("Mengambil data dari Neo4j..."):
                    graph_results = run_cypher(cypher)

                st.markdown("### Hasil Retrieval dari Graph")
                display_results(graph_results)

                with st.spinner("Menghasilkan jawaban RAG..."):
                    answer = generate_rag_answer(rag_question, cypher, graph_results)

                st.markdown("### Jawaban Graph RAG")
                st.info(answer)

        except Exception as e:
            st.error(f"Terjadi error: {e}")