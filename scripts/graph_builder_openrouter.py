from neo4j import GraphDatabase
from dotenv import load_dotenv
import requests
import os
import json
import re

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

ALLOWED_ENTITY_TYPES = {"Penyakit", "Gejala", "Obat", "Kategori"}
ALLOWED_RELATION_TYPES = {
    "MEMILIKI_GEJALA",
    "DIOBATI_DENGAN",
    "TERMASUK_KATEGORI"
}


# =========================
# PROMPT GRAPH BUILDER
# =========================
def build_graph_extraction_prompt(text: str) -> str:
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


# =========================
# OPENROUTER
# =========================
def call_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY belum diisi di file .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You extract structured graph data as valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1200
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


def clean_json_response(text: str) -> str:
    text = text.strip()

    # Hapus markdown code block kalau ada
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    return text


def extract_graph_from_text(text: str) -> dict:
    prompt = build_graph_extraction_prompt(text)
    raw_response = call_openrouter(prompt)
    cleaned = clean_json_response(raw_response)

    try:
        graph_data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("Response LLM bukan JSON valid:")
        print(cleaned)
        raise

    return validate_graph_data(graph_data)


# =========================
# VALIDATION
# =========================
def normalize_name(name: str) -> str:
    return str(name).strip()


def validate_graph_data(graph_data: dict) -> dict:
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

        # Validasi arah relasi
        if relation == "MEMILIKI_GEJALA" and not (source_type == "Penyakit" and target_type == "Gejala"):
            continue
        if relation == "DIOBATI_DENGAN" and not (source_type == "Penyakit" and target_type == "Obat"):
            continue
        if relation == "TERMASUK_KATEGORI" and not (source_type == "Penyakit" and target_type == "Kategori"):
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


def create_constraints(driver):
    queries = [
        """
        CREATE CONSTRAINT penyakit_nama IF NOT EXISTS
        FOR (p:Penyakit)
        REQUIRE p.nama IS UNIQUE
        """,
        """
        CREATE CONSTRAINT gejala_nama IF NOT EXISTS
        FOR (g:Gejala)
        REQUIRE g.nama IS UNIQUE
        """,
        """
        CREATE CONSTRAINT obat_nama IF NOT EXISTS
        FOR (o:Obat)
        REQUIRE o.nama IS UNIQUE
        """,
        """
        CREATE CONSTRAINT kategori_nama IF NOT EXISTS
        FOR (k:Kategori)
        REQUIRE k.nama IS UNIQUE
        """
    ]

    with driver.session(database=NEO4J_DATABASE) as session:
        for query in queries:
            session.run(query)


def create_entity(session, entity_type: str, name: str):
    if entity_type == "Penyakit":
        session.run(
            "MERGE (n:Penyakit {nama: $name})",
            name=name
        )
    elif entity_type == "Gejala":
        session.run(
            "MERGE (n:Gejala {nama: $name})",
            name=name
        )
    elif entity_type == "Obat":
        session.run(
            "MERGE (n:Obat {nama: $name})",
            name=name
        )
    elif entity_type == "Kategori":
        session.run(
            "MERGE (n:Kategori {nama: $name})",
            name=name
        )


def create_relationship(session, rel: dict):
    relation = rel["relation"]
    source = rel["source"]
    target = rel["target"]

    if relation == "MEMILIKI_GEJALA":
        session.run(
            """
            MERGE (p:Penyakit {nama: $source})
            MERGE (g:Gejala {nama: $target})
            MERGE (p)-[:MEMILIKI_GEJALA]->(g)
            """,
            source=source,
            target=target
        )

    elif relation == "DIOBATI_DENGAN":
        session.run(
            """
            MERGE (p:Penyakit {nama: $source})
            MERGE (o:Obat {nama: $target})
            MERGE (p)-[:DIOBATI_DENGAN]->(o)
            """,
            source=source,
            target=target
        )

    elif relation == "TERMASUK_KATEGORI":
        session.run(
            """
            MERGE (p:Penyakit {nama: $source})
            MERGE (k:Kategori {nama: $target})
            MERGE (p)-[:TERMASUK_KATEGORI]->(k)
            """,
            source=source,
            target=target
        )


def save_graph_to_neo4j(driver, graph_data: dict):
    create_constraints(driver)

    with driver.session(database=NEO4J_DATABASE) as session:
        for entity in graph_data["entities"]:
            create_entity(session, entity["type"], entity["name"])

        for rel in graph_data["relationships"]:
            create_relationship(session, rel)


def show_summary(driver):
    with driver.session(database=NEO4J_DATABASE) as session:
        print("\n=== JUMLAH NODE ===")
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS jumlah
            ORDER BY jumlah DESC
        """)
        for row in result:
            print(row.data())

        print("\n=== JUMLAH RELASI ===")
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS relasi, count(r) AS jumlah
            ORDER BY jumlah DESC
        """)
        for row in result:
            print(row.data())


# =========================
# MAIN
# =========================
def main():
    print("=== LLM TEXT-TO-GRAPH BUILDER DENGAN OPENROUTER + NEO4J ===\n")
    print("Masukkan teks medis.")
    print("Kalau sudah selesai, ketik END di baris baru.\n")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    input_text = "\n".join(lines).strip()

    if not input_text:
        print("Teks kosong.")
        return

    driver = None

    try:
        print("\nMenghubungkan ke Neo4j...")
        driver = connect_neo4j()
        print("Koneksi Neo4j berhasil.")

        print("\nMengirim teks ke OpenRouter untuk ekstraksi graph...")
        graph_data = extract_graph_from_text(input_text)

        print("\n=== HASIL EKSTRAKSI LLM ===")
        print(json.dumps(graph_data, indent=2, ensure_ascii=False))

        print("\nMenyimpan hasil ekstraksi ke Neo4j...")
        save_graph_to_neo4j(driver, graph_data)
        print("Graph berhasil ditambahkan ke Neo4j.")

        show_summary(driver)

    except Exception as e:
        print("\nTerjadi error:")
        print(e)

    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    main()