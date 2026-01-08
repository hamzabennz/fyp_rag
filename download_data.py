import os
import subprocess
from datasets import load_dataset
from tqdm import tqdm


def ingest_enron_split(
    split: str = "test",
    limit: int | None = None,          # None = ingest everything
    ingest_script: str = "ingest.py",
    strategy: str = "semantic",
    device: str = "cuda",
    resume_log: str | None = "ingested_ids.txt",  # set None to disable
):
    print(f"Loading EnronQA corpus split='{split}'...")
    dataset = load_dataset("MichaelR207/enron_qa_0922", split=split)

    total = len(dataset) if limit is None else min(limit, len(dataset))
    print(f"Ingesting {total} emails into your RAG system...")

    # Optional resume support (skip already ingested paths)
    already_done = set()
    if resume_log:
        if os.path.exists(resume_log):
            with open(resume_log, "r", encoding="utf-8") as f:
                already_done = set(line.strip() for line in f if line.strip())

    # Iterate full split (or first N if limit is set)
    iterable = dataset if limit is None else dataset.select(range(total))

    for record in tqdm(iterable, total=total):
        email_text = record["email"]
        email_path = record["path"]  # used as unique source id

        if resume_log and email_path in already_done:
            continue

        command = [
            "python",
            ingest_script,
            "--source",
            email_path,
            "--text",
            email_text,
            "--strategy",
            strategy,
            "--device",
            device,
            "--resource-type",  # Added to specify collection
            "enron",            # Collection name
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            if resume_log:
                with open(resume_log, "a", encoding="utf-8") as f:
                    f.write(email_path + "\n")
        except subprocess.CalledProcessError as e:
            print(f"\nError ingesting email {email_path}:\n{e.stderr}\n")
            continue

    print(f"\nSuccessfully finished ingesting {total} documents from split='{split}'.")


ingest_enron_split(split="test", limit=1000)