#!/usr/bin/env python3
"""
Batch Ingestion Script for Evidence Files (Emails, Transactions, SMS)

Ingests JSON batches of emails, bank transactions, and SMS messages into the RAG system.
Converts structured data to readable text format for semantic chunking.

Usage:
    python ingest_batches.py --data-dir ./current_docs/evidence --strategy semantic --device cuda
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load and parse JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in {file_path}: {e}")
        return []
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return []


def format_email(email: Dict[str, Any], batch_num: int, idx: int) -> str:
    """Format email as readable text."""
    return f"""EMAIL RECORD
ID: {email.get('id', 'N/A')}
Timestamp: {email.get('timestamp', 'N/A')}
From: {email.get('from', 'N/A')}
To: {email.get('to', 'N/A')}
Subject: {email.get('subject', 'N/A')}
Body:
{email.get('body', 'N/A')}
---
"""


def format_transaction(txn: Dict[str, Any], batch_num: int, idx: int) -> str:
    """Format bank transaction as readable text."""
    return f"""BANK TRANSACTION RECORD
Transaction ID: {txn.get('transaction_id', 'N/A')}
Date: {txn.get('date', 'N/A')}
From Account: {txn.get('from_account', 'N/A')}
To Account: {txn.get('to_account', 'N/A')}
Amount: {txn.get('amount', 'N/A')} {txn.get('currency', 'USD')}
Type: {txn.get('type', 'N/A')}
Description: {txn.get('description', 'N/A')}
Bank: {txn.get('bank', 'N/A')}
---
"""


def format_sms(sms: Dict[str, Any], batch_num: int, idx: int) -> str:
    """Format SMS as readable text."""
    return f"""SMS RECORD
Timestamp: {sms.get('timestamp', 'N/A')}
Sender: {sms.get('sender', 'N/A')}
Receiver: {sms.get('receiver', 'N/A')}
Message: {sms.get('message', 'N/A')}
---
"""


def ingest_batch(
    batch_path: Path,
    batch_type: str,
    batch_num: int,
    strategy: str = "semantic",
    device: str = "cuda",
) -> int:
    """
    Ingest a single batch file.

    Args:
        batch_path: Path to batch JSON file
        batch_type: Type of batch ('emails', 'transactions', 'sms')
        batch_num: Batch number (1-5)
        strategy: Chunking strategy
        device: Device to use

    Returns:
        Number of records ingested
    """
    logger.info(f"Processing {batch_type.upper()} Batch {batch_num}: {batch_path.name}")

    # Load JSON
    records = load_json_file(batch_path)
    if not records:
        logger.warning(f"No records found in {batch_path.name}")
        return 0

    # Choose formatter
    if batch_type == "emails":
        formatter = format_email
    elif batch_type == "transactions":
        formatter = format_transaction
    elif batch_type == "sms":
        formatter = format_sms
    else:
        logger.error(f"Unknown batch type: {batch_type}")
        return 0

    # Format and ingest each record
    ingested = 0
    for idx, record in enumerate(records):
        try:
            # Format record as text
            text = formatter(record, batch_num, idx)

            # Generate source ID (unique identifier for this record)
            if batch_type == "emails":
                source = f"emails_batch_{batch_num}_id_{record.get('id', idx)}"
            elif batch_type == "transactions":
                source = f"transactions_batch_{batch_num}_txn_{record.get('transaction_id', idx)}"
            elif batch_type == "sms":
                timestamp = record.get('timestamp', '')
                sender = record.get('sender', '')
                source = f"sms_batch_{batch_num}_{sender}_{timestamp}".replace(':', '-').replace('+', '_')
            else:
                source = f"{batch_type}_batch_{batch_num}_{idx}"

            # Ingest via ingest.py with resource type
            command = [
                "python", "ingest.py",
                "--source", source,
                "--text", text,
                "--strategy", strategy,
                "--device", device,
                "--resource-type", batch_type,  # Add resource type
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                ingested += 1
            else:
                logger.error(f"Failed to ingest {source}: {result.stderr}")

        except Exception as e:
            logger.error(f"Error processing record {idx} in {batch_type} batch {batch_num}: {e}")

    logger.info(f"✓ Ingested {ingested}/{len(records)} {batch_type} from batch {batch_num}")
    return ingested


def main():
    parser = argparse.ArgumentParser(
        description="Ingest JSON evidence batches (emails, transactions, SMS) into RAG system"
    )

    parser.add_argument(
        "--data-dir",
        default="./current_docs/evidence",
        help="Base directory containing evidence subdirectories (default: ./current_docs/evidence)",
    )

    parser.add_argument(
        "--strategy",
        choices=["semantic", "layout", "hybrid"],
        default="semantic",
        help="Chunking strategy (default: semantic)",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cuda",
        help="Device to use (default: cuda)",
    )

    parser.add_argument(
        "--batch-types",
        nargs="+",
        choices=["emails", "transactions", "sms"],
        default=["emails", "transactions", "sms"],
        help="Which batch types to ingest (default: all)",
    )

    parser.add_argument(
        "--batch-nums",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Which batch numbers to ingest (default: 1-5)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Starting Batch Ingestion")
    logger.info("=" * 80)
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch Types: {args.batch_types}")
    logger.info(f"Batch Numbers: {args.batch_nums}")
    logger.info("=" * 80)

    total_ingested = 0

    # Process each batch type and batch number
    for batch_type in args.batch_types:
        for batch_num in args.batch_nums:
            batch_file = data_dir / batch_type / f"{batch_type}_batch_{batch_num:02d}.json"

            if not batch_file.exists():
                logger.warning(f"File not found: {batch_file}")
                continue

            count = ingest_batch(
                batch_file,
                batch_type,
                batch_num,
                strategy=args.strategy,
                device=args.device,
            )
            total_ingested += count

    logger.info("=" * 80)
    logger.info(f"✓ Batch ingestion complete!")
    logger.info(f"  Total records ingested: {total_ingested}")
    logger.info("=" * 80)

    # Show stats
    logger.info("\nRunning /stats endpoint...")
    logger.info("Use: curl http://localhost:5000/stats")


if __name__ == "__main__":
    main()
