"""Try the RAG pipeline end to end with the REAL embedding model and TinyLlama.

Run from the project root (Ollama must be running):

    python scripts/try_rag.py
"""

import os
import sys
import time
from pathlib import Path

# Make `import app` work no matter where the script is started from,
# and use the project's ./data folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from app import rag, sessions  # noqa: E402

SAMPLE_TEXT = """\
Acme Robotics Company Handbook

Acme Robotics was founded in 2015 in Denver, Colorado by Maria Lopez and Tom Chen.
The company builds warehouse robots that move boxes and pallets.

Support hours: Our customer support team is available Monday to Friday, from
8 AM to 6 PM Mountain Time. Support is closed on weekends and public holidays.

Refund policy: Customers may return any robot within 30 days of delivery for a
full refund, as long as the robot is in its original packaging. After 30 days,
only repairs under the two-year warranty are available.

Shipping: Orders ship from the Denver warehouse within 3 business days.
"""

QUESTIONS = [
    "When was Acme Robotics founded, and where?",  # answerable
    "How many days do customers have to return a robot for a refund?",  # answerable
    "What is the CEO's favorite color?",  # NOT in the text -> should say "I don't know"
]


def main() -> None:
    print("Loading embedding model and LLM client (first run downloads the model)...")
    embeddings = rag.load_embeddings()
    llm = rag.load_llm()

    sid = sessions.create_session()
    try:
        # Put the sample file into the session's uploads folder, then ingest it.
        sample = sessions.uploads_dir(sid) / "acme_handbook.txt"
        sample.write_text(SAMPLE_TEXT, encoding="utf-8")
        chunks = rag.ingest(sid, sample, embeddings)
        print(f"Ingested {sample.name}: {chunks} chunks\n")

        for question in QUESTIONS:
            start = time.time()
            result = rag.ask(sid, question, embeddings, llm)
            print(f"Q: {question}")
            print(f"A: {result['answer']}")
            print(f"   sources: {result['sources']}   ({time.time() - start:.1f}s)\n")
    finally:
        sessions.delete_session(sid)  # tidy up the demo session


if __name__ == "__main__":
    main()
