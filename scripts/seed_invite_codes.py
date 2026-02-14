"""
Seed the invite_codes collection with one-time-use invite codes.

Uses synchronous pymongo so it can be run as a standalone script without
an async event loop.

Usage (from rpg-agents project root):

  # 10 single-use codes against local Mongo
  python scripts/seed_invite_codes.py

  # 50 single-use codes
  python scripts/seed_invite_codes.py --count 50

  # 5 codes with 10 uses each
  python scripts/seed_invite_codes.py --count 5 --max-uses 10

  # Unlimited use codes
  python scripts/seed_invite_codes.py --count 3 --unlimited

  # Codes that expire in 168 hours (1 week)
  python scripts/seed_invite_codes.py --count 20 --expires-hours 168

  # Fetch URI from gcloud secret for prod
  MONGO_URL=$(gcloud secrets versions access latest --secret=mongodb-uri) \
    python scripts/seed_invite_codes.py --count 50

Environment variables:
  MONGO_URL  - MongoDB connection string (default: mongodb://localhost:27017)
  MONGO_DB   - Database name            (default: rpg_mcp)
"""

import argparse
import os
import secrets
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from pymongo import MongoClient
from pymongo.errors import BulkWriteError


def generate_code() -> str:
    """Generate a URL-safe random invite code."""
    return secrets.token_urlsafe(8)


def build_codes(count: int, max_uses: Optional[int], expires_at: Optional[datetime], created_by: str) -> List[Dict]:
    """Build a list of invite-code documents ready for insertion."""
    now = datetime.utcnow()
    codes = []
    for _ in range(count):
        doc = {
            "code": generate_code(),
            "created_by": created_by,
            "max_uses": max_uses,
            "uses": 0,
            "expires_at": expires_at,
            "created_at": now,
        }
        codes.append(doc)
    return codes


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed invite codes into MongoDB.")
    parser.add_argument("--count", type=int, default=10, help="Number of codes to generate (default: 10)")
    parser.add_argument("--max-uses", type=int, default=1, help="Max uses per code (default: 1). Ignored if --unlimited.")
    parser.add_argument("--unlimited", action="store_true", help="Allow unlimited uses per code")
    parser.add_argument("--expires-hours", type=float, default=None, help="Hours until codes expire (default: never)")
    parser.add_argument("--created-by", type=str, default="system", help="Creator label (default: system)")
    parser.add_argument("--output", type=str, default="invite_codes.txt", help="File to save generated codes (default: invite_codes.txt)")
    args = parser.parse_args()

    # --- Resolve settings ---
    max_uses = None if args.unlimited else args.max_uses  # type: Optional[int]
    expires_at = None  # type: Optional[datetime]
    if args.expires_hours is not None:
        expires_at = datetime.utcnow() + timedelta(hours=args.expires_hours)

    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "rpg_mcp")

    # Redact password in displayed URI for safety
    display_url = mongo_url
    if "@" in mongo_url:
        prefix, rest = mongo_url.split("@", 1)
        scheme_end = prefix.find("://")
        if scheme_end != -1:
            display_url = f"{prefix[:scheme_end + 3]}***:***@{rest}"

    print(f"Connecting to MongoDB: {display_url}")
    print(f"Database: {db_name}\n")

    # --- Connect ---
    try:
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        # Force a connection check
        client.admin.command("ping")
    except Exception as exc:
        print(f"ERROR: Could not connect to MongoDB: {exc}", file=sys.stderr)
        return 1

    db = client[db_name]

    # Ensure unique index on code
    db.invite_codes.create_index("code", unique=True)

    # --- Generate codes ---
    print(f"Generating {args.count} invite codes...")
    if max_uses is not None:
        print(f"  Max uses: {max_uses}")
    else:
        print("  Max uses: unlimited")
    if expires_at:
        print(f"  Expires: {expires_at.isoformat()}Z")
    else:
        print("  Expires: never")
    print()

    codes = build_codes(
        count=args.count,
        max_uses=max_uses,
        expires_at=expires_at,
        created_by=args.created_by,
    )

    # --- Insert ---
    try:
        result = db.invite_codes.insert_many(codes, ordered=False)
        inserted_count = len(result.inserted_ids)
    except BulkWriteError as bwe:
        inserted_count = bwe.details.get("nInserted", 0)
        dup_count = len(bwe.details.get("writeErrors", []))
        print(f"WARNING: {dup_count} duplicate code(s) skipped (extremely unlikely collision).")

    print(f"Successfully inserted {inserted_count} invite codes\n")

    # --- Display codes ---
    code_values = [doc["code"] for doc in codes]

    print("Generated Codes:")
    print("=" * 60)
    for code in code_values:
        print(f"  {code}")
    print("=" * 60)
    print()

    # --- Save to file ---
    out_path = Path(args.output)
    out_path.write_text("\n".join(code_values) + "\n", encoding="utf-8")
    print(f"Codes saved to {out_path}")
    print()

    # --- Stats ---
    total = db.invite_codes.count_documents({})
    active_filter = {
        "$and": [
            {"$or": [{"expires_at": None}, {"expires_at": {"$gt": datetime.utcnow()}}]},
            {"$or": [{"max_uses": None}, {"$expr": {"$lt": ["$uses", "$max_uses"]}}]},
        ]
    }
    active = db.invite_codes.count_documents(active_filter)

    print("Database Statistics:")
    print(f"  Total invite codes: {total}")
    print(f"  Active codes:       {active}")

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
