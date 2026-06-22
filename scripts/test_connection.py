from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

try:
    driver.verify_connectivity()
    print("Koneksi ke Neo4j berhasil.")

    with driver.session(database=DATABASE) as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n) AS Label, count(n) AS Jumlah
            LIMIT 10
        """)

        print("\nData di database:")
        for row in result:
            print(row.data())

except Exception as e:
    print("Koneksi gagal:")
    print(e)

finally:
    driver.close()