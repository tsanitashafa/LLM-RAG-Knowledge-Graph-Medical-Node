from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import re
import requests

load_dotenv()

# =========================
# CONFIG
# =========================
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SCHEMA = """
Node labels:
- Penyakit {qid, nama, clusterId}
- Gejala {nama, clusterId}
- Obat {nama, clusterId}
- Kategori {nama, clusterId}

Relationships:
- (Penyakit)-[:MEMILIKI_GEJALA]->(Gejala)
- (Penyakit)-[:DIOBATI_DENGAN]->(Obat)
- (Penyakit)-[:TERMASUK_KATEGORI]->(Kategori)

Important:
- clusterId is the result of GDS Louvain community detection.
- If the user asks about cluster, community, kelompok, or hasil GDS, use clusterId.
- Cluster is NOT the same as Kategori.
- Do not use TERMASUK_KATEGORI for cluster questions.
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


# =========================
# PROMPT
# =========================
def build_prompt(question: str) -> str:
    return f"""
You are a Neo4j Cypher generator.

Convert the user's natural language question into a valid Cypher query.

Graph schema:
{SCHEMA}

Rules:
- Return Cypher query only.
- Use MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT, collect, count, DISTINCT only.
- Do not use CREATE, DELETE, SET, MERGE, DROP, REMOVE, LOAD CSV, CALL, APOC, or GDS.
- Use toLower() and CONTAINS for flexible text search.
- If the question asks about cluster/community, use the clusterId property.
- Cluster means GDS Louvain result stored in clusterId, not Kategori.
- Always add LIMIT 20 unless the query already uses LIMIT 1 for top result.
- Do not explain the query.
- Return only the Cypher query, without markdown code block.

Example:
Question: satu cluster dengan penyakit terbanyak dan sebutkan nama penyakitnya
Cypher:
MATCH (p:Penyakit)
WHERE p.clusterId IS NOT NULL
WITH p.clusterId AS cluster, collect(DISTINCT p.nama) AS daftarPenyakit, count(DISTINCT p) AS jumlahPenyakit
ORDER BY jumlahPenyakit DESC
LIMIT 1
RETURN cluster, jumlahPenyakit, daftarPenyakit[0..20] AS namaPenyakit

User question:
{question}
""".strip()


# =========================
# OPENROUTER
# =========================
def generate_cypher_with_openrouter(question: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY belum diisi di file .env")

    prompt = build_prompt(question)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You generate safe read-only Neo4j Cypher queries only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 500
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}:\n{response.text}"
        )

    data = response.json()
    cypher = data["choices"][0]["message"]["content"]

    return clean_cypher(cypher)


# =========================
# CLEAN & VALIDATE CYPHER
# =========================
def clean_cypher(text: str) -> str:
    text = text.strip()

    # Hapus markdown code block kalau LLM tetap mengeluarkan ```cypher
    text = re.sub(r"^```cypher", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # Hapus titik koma di akhir
    if text.endswith(";"):
        text = text[:-1]

    return text.strip()


def is_safe_cypher(cypher: str) -> bool:
    upper_query = cypher.upper().strip()

    if not upper_query.startswith("MATCH"):
        print("Query ditolak: query harus diawali MATCH.")
        return False

    if "RETURN" not in upper_query:
        print("Query ditolak: query harus memiliki RETURN.")
        return False

    if "LIMIT" not in upper_query:
        print("Query ditolak: query harus memiliki LIMIT.")
        return False

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in upper_query:
            print(f"Query ditolak: mengandung keyword berbahaya: {keyword}")
            return False

    return True


# =========================
# NEO4J
# =========================
def connect_neo4j():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    return driver


def run_cypher(driver, cypher: str):
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher)
        return [record.data() for record in result]


def print_results(results):
    print("\n=== HASIL QUERY ===")

    if not results:
        print("Tidak ada hasil.")
        print("Kemungkinan keyword tidak ada di data, atau nama label/relasi berbeda.")
        return

    for i, row in enumerate(results, start=1):
        print(f"\n{i}.")
        for key, value in row.items():
            print(f"   {key}: {value}")


# =========================
# MAIN
# =========================
def main():
    print("=== LLM TEXT-TO-CYPHER DENGAN OPENROUTER + NEO4J ===\n")

    question = input("Masukkan pertanyaan: ")

    driver = None

    try:
        print("\nMenghubungkan ke Neo4j...")
        driver = connect_neo4j()
        print("Koneksi Neo4j berhasil.")

        print("\nMengirim pertanyaan ke OpenRouter...")
        cypher = generate_cypher_with_openrouter(question)

        print("\n=== GENERATED CYPHER ===")
        print(cypher)

        if not is_safe_cypher(cypher):
            print("\nQuery tidak dijalankan karena tidak lolos validasi.")
            return

        print("\nMenjalankan query ke Neo4j...")
        results = run_cypher(driver, cypher)

        print_results(results)

    except Exception as e:
        print("\nTerjadi error:")
        print(e)

    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()