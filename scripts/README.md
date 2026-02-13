# RPG Agents Scripts

Utility scripts for managing the RPG Agents application.

## seed_invite_codes.py

Generate and insert invite codes into MongoDB for user registration.

### Prerequisites

```bash
# Install dependencies (from rpg-agents root)
pip install -r requirements.txt
```

### Usage

```bash
# Basic usage - 10 single-use codes
python scripts/seed_invite_codes.py

# Generate 50 single-use codes for production
python scripts/seed_invite_codes.py --count 50

# Generate 5 codes with 10 uses each
python scripts/seed_invite_codes.py --count 5 --max-uses 10

# Generate unlimited use codes (careful!)
python scripts/seed_invite_codes.py --count 3 --unlimited

# Generate codes that expire in 1 week (168 hours)
python scripts/seed_invite_codes.py --count 20 --expires-hours 168

# Generate codes that expire in 30 days
python scripts/seed_invite_codes.py --count 100 --max-uses 1 --expires-hours 720
```

### Environment Variables

- `MONGO_URL`: MongoDB connection string (default: `mongodb://localhost:27017`)
- `MONGO_DB`: Database name (default: `rpg_mcp`)

### Production Examples

```bash
# Local development
python scripts/seed_invite_codes.py --count 10

# Production via Kubernetes (port-forward)
kubectl port-forward -n conjurers-table statefulset/mongodb 27017:27017 &
MONGO_URL="mongodb://admin:$(kubectl get secret mongodb-secret -n conjurers-table -o jsonpath='{.data.mongodb-password}' | base64 -d)@localhost:27017" \
  python scripts/seed_invite_codes.py --count 50 --max-uses 1 --expires-hours 720

# Production via kubectl exec (directly on MongoDB pod)
kubectl exec -it mongodb-0 -n conjurers-table -- mongosh mongodb://admin:PASSWORD@localhost:27017/rpg_mcp --eval '
db.invite_codes.insertMany([
  { code: "CODE1", created_by: "admin", max_uses: 1, uses: 0, expires_at: null, created_at: new Date() },
  { code: "CODE2", created_by: "admin", max_uses: 1, uses: 0, expires_at: null, created_at: new Date() }
])
'
```

### Output

The script will:
1. Connect to MongoDB
2. Generate random invite codes
3. Insert them into the `invite_codes` collection
4. Display the codes in the terminal
5. Save codes to `invite_codes.txt`
6. Show database statistics

Example output:

```
Connecting to MongoDB: mongodb://localhost:27017

Generating 10 invite codes...
  Max uses: 1
  Expires: never

✓ Successfully inserted 10 invite codes

Generated Codes:
================================================================================
  XvZ8Kj2P-Q
  LmN3Rt9Y-w
  ...
================================================================================

✓ Codes saved to invite_codes.txt

Database Statistics:
  Total invite codes: 10
  Active codes: 10
```

### Security Notes

- **Protect invite codes:** Treat them like passwords
- **Use single-use codes** for production (`--max-uses 1`)
- **Set expiration** for time-limited campaigns (`--expires-hours 168`)
- **Don't commit** `invite_codes.txt` to git (already in .gitignore)
- **Rotate codes regularly:** Delete used/expired codes periodically

### Monitoring Invite Code Usage

Query MongoDB to check invite code usage:

```javascript
// List all codes with usage stats
db.invite_codes.find({}, {
  code: 1,
  uses: 1,
  max_uses: 1,
  expires_at: 1,
  created_at: 1
}).sort({ created_at: -1 })

// Find active codes (not expired, not exhausted)
db.invite_codes.find({
  $and: [
    {
      $or: [
        { expires_at: null },
        { expires_at: { $gt: new Date() } }
      ]
    },
    {
      $or: [
        { max_uses: null },
        { $expr: { $lt: ["$uses", "$max_uses"] } }
      ]
    }
  ]
})

// Find exhausted codes
db.invite_codes.find({
  max_uses: { $ne: null },
  $expr: { $gte: ["$uses", "$max_uses"] }
})

// Find expired codes
db.invite_codes.find({
  expires_at: { $lte: new Date() }
})

// Delete used codes (cleanup)
db.invite_codes.deleteMany({
  max_uses: 1,
  uses: { $gte: 1 }
})
```

### Initial Production Setup

For initial production deployment, create codes for your team:

```bash
# Create 50 single-use codes that expire in 30 days
# These can be shared with initial users
python scripts/seed_invite_codes.py \
  --count 50 \
  --max-uses 1 \
  --expires-hours 720 \
  --created-by "admin-$(date +%Y%m%d)"

# Save the output invite_codes.txt somewhere safe
# Distribute codes via secure channel (not email/slack)
```

### Troubleshooting

**Connection refused:**
- Verify MongoDB is running: `docker ps` or `kubectl get pods -n conjurers-table`
- Check MONGO_URL is correct
- Ensure MongoDB authentication (if enabled)

**Authentication failed:**
- Check MongoDB username/password
- Update connection string: `mongodb://username:password@host:port`

**Permission denied:**
- Ensure MongoDB user has write access to the database
- Check database name matches (MONGO_DB)

**Script not found:**
- Run from rpg-agents directory: `cd rpg-agents`
- Or use full path: `python /path/to/rpg-agents/scripts/seed_invite_codes.py`
