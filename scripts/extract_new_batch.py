#!/usr/bin/env python3
"""Extract concepts for the new batch passages."""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db.postgres import get_pg_pool
from lib.ingest.concept_extractor import ConceptExtractor

def main():
    pool = get_pg_pool()
    extractor = ConceptExtractor()

    print("Starting targeted concept extraction for new batch...")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.passage_id, p.passage_text
                FROM passages p
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE d.ingest_batch = 'zotero_ingest_1768571104'
                AND NOT EXISTS (SELECT 1 FROM passage_concepts pc WHERE pc.passage_id = p.passage_id)
                LIMIT 200
            """)

            passages = list(cur.fetchall())
            print(f"Found {len(passages)} passages to process")

            processed = 0
            total_concepts = 0

            for row in passages:
                passage_id = row['passage_id']
                text = row['passage_text']

                result = extractor.extract(text)

                if result.success and result.concepts:
                    for c in result.concepts:
                        cur.execute("""
                            INSERT INTO passage_concepts (
                                passage_id, concept_name, concept_type, confidence,
                                extractor_model, extractor_version
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (passage_id, concept_name, extractor_version) DO NOTHING
                        """, (
                            passage_id, c.name, c.type, c.confidence,
                            'gemini-2.0-flash', 'v3.0'
                        ))
                    total_concepts += len(result.concepts)
                    conn.commit()

                processed += 1
                if processed % 20 == 0:
                    print(f"Progress: {processed}/{len(passages)}, {total_concepts} concepts")

                time.sleep(0.3)

            print(f"\nComplete: {processed} passages, {total_concepts} concepts extracted")

if __name__ == "__main__":
    main()
