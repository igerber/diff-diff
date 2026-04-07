#!/bin/bash
# Automated tests for check-plan-review.sh
# Run: bash .claude/hooks/test-check-plan-review.sh

set -u

HOOK_SCRIPT="$(cd "$(dirname "$0")" && pwd)/check-plan-review.sh"
TMPBASE=$(mktemp -d)
trap 'rm -rf "$TMPBASE"' EXIT

PASS=0
FAIL=0

run_test() {
  local name="$1" case_dir="$2" expect="$3"
  local output
  output=$(HOME="$case_dir" bash "$HOOK_SCRIPT" 2>/dev/null)
  if [ "$expect" = "allow" ]; then
    if echo "$output" | grep -q '"permissionDecision":"deny"'; then
      echo "FAIL: $name (expected ALLOW, got DENY)"
      echo "  output: $output"
      FAIL=$((FAIL + 1))
    else
      echo "PASS: $name"
      PASS=$((PASS + 1))
    fi
  else
    if echo "$output" | grep -q '"permissionDecision":"deny"'; then
      echo "PASS: $name"
      PASS=$((PASS + 1))
    else
      echo "FAIL: $name (expected DENY, got ALLOW)"
      echo "  output: $output"
      FAIL=$((FAIL + 1))
    fi
  fi
}

# Helper: create a minimal review file with YAML frontmatter
# Usage: create_review <review-path> <plan-path>
create_review() {
  local review_path="$1" plan_path="$2"
  cat > "$review_path" <<EOF
---
plan: $plan_path
reviewed_at: 2026-01-01T00:00:00Z
assessment: "No critical issues found"
critical_count: 0
medium_count: 0
low_count: 0
flags: []
---
Review content.
EOF
}

# ============================================================
# Case 1: No sentinel, no plans → ALLOW
# ============================================================
CASE_DIR="$TMPBASE/case1"
mkdir -p "$CASE_DIR/.claude/plans"
run_test "Case 1: No sentinel, no plans → ALLOW" "$CASE_DIR" "allow"

# ============================================================
# Case 2: Sentinel → plan with fresh review → ALLOW
# ============================================================
CASE_DIR="$TMPBASE/case2"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/test-plan.md"
REVIEW="$CASE_DIR/.claude/plans/test-plan.review.md"
echo "# Plan" > "$PLAN"
touch -t 202601010001 "$PLAN"
create_review "$REVIEW" "$PLAN"
touch -t 202601010002 "$REVIEW"
echo "$PLAN" > "$CASE_DIR/.claude/plans/.last-reviewed"
run_test "Case 2: Sentinel → fresh review → ALLOW" "$CASE_DIR" "allow"

# ============================================================
# Case 3: Sentinel → plan with stale review → DENY
# ============================================================
CASE_DIR="$TMPBASE/case3"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/test-plan.md"
REVIEW="$CASE_DIR/.claude/plans/test-plan.review.md"
echo "# Plan" > "$PLAN"
touch -t 202601010002 "$PLAN"
create_review "$REVIEW" "$PLAN"
touch -t 202601010001 "$REVIEW"
echo "$PLAN" > "$CASE_DIR/.claude/plans/.last-reviewed"
run_test "Case 3: Sentinel → stale review → DENY" "$CASE_DIR" "deny"

# ============================================================
# Case 4: Sentinel → plan with no review → DENY
# ============================================================
CASE_DIR="$TMPBASE/case4"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/test-plan.md"
echo "# Plan" > "$PLAN"
echo "$PLAN" > "$CASE_DIR/.claude/plans/.last-reviewed"
run_test "Case 4: Sentinel → no review → DENY" "$CASE_DIR" "deny"

# ============================================================
# Case 5: Sentinel → non-existent file, fallback plan has review
#         with matching plan: field → ALLOW
# ============================================================
CASE_DIR="$TMPBASE/case5"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/real-plan.md"
REVIEW="$CASE_DIR/.claude/plans/real-plan.review.md"
echo "# Plan" > "$PLAN"
touch -t 202601010001 "$PLAN"
create_review "$REVIEW" "$PLAN"
touch -t 202601010002 "$REVIEW"
# Sentinel points to a non-existent file
echo "$CASE_DIR/.claude/plans/deleted-plan.md" > "$CASE_DIR/.claude/plans/.last-reviewed"
run_test "Case 5: Sentinel → non-existent file, fallback with review → ALLOW" "$CASE_DIR" "allow"

# ============================================================
# Case 6: No sentinel, fallback newest plan with review → ALLOW
# ============================================================
CASE_DIR="$TMPBASE/case6"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/test-plan.md"
REVIEW="$CASE_DIR/.claude/plans/test-plan.review.md"
echo "# Plan" > "$PLAN"
touch -t 202601010001 "$PLAN"
create_review "$REVIEW" "$PLAN"
touch -t 202601010002 "$REVIEW"
# No sentinel file
run_test "Case 6: No sentinel, fallback with review → ALLOW" "$CASE_DIR" "allow"

# ============================================================
# Case 7: No sentinel, fallback newest plan without review → DENY
# ============================================================
CASE_DIR="$TMPBASE/case7"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/test-plan.md"
echo "# Plan" > "$PLAN"
# No sentinel, no review
run_test "Case 7: No sentinel, fallback without review → DENY" "$CASE_DIR" "deny"

# ============================================================
# Case 8: Review file plan: field doesn't match plan path → DENY
# ============================================================
CASE_DIR="$TMPBASE/case8"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/test-plan.md"
REVIEW="$CASE_DIR/.claude/plans/test-plan.review.md"
echo "# Plan" > "$PLAN"
touch -t 202601010001 "$PLAN"
# Review points to a different plan
create_review "$REVIEW" "/some/other/plan.md"
touch -t 202601010002 "$REVIEW"
echo "$PLAN" > "$CASE_DIR/.claude/plans/.last-reviewed"
run_test "Case 8: Review plan: field mismatch → DENY" "$CASE_DIR" "deny"

# ============================================================
# Case 9: Review file has no plan: field → DENY
# ============================================================
CASE_DIR="$TMPBASE/case9"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN="$CASE_DIR/.claude/plans/test-plan.md"
REVIEW="$CASE_DIR/.claude/plans/test-plan.review.md"
echo "# Plan" > "$PLAN"
touch -t 202601010001 "$PLAN"
# Review without plan: field in frontmatter
cat > "$REVIEW" <<EOF
---
reviewed_at: 2026-01-01T00:00:00Z
assessment: "No critical issues found"
---
Review without plan field.
EOF
touch -t 202601010002 "$REVIEW"
echo "$PLAN" > "$CASE_DIR/.claude/plans/.last-reviewed"
run_test "Case 9: Review has no plan: field → DENY" "$CASE_DIR" "deny"

# ============================================================
# Case 10: Sentinel → plan-A with review, newer plan-B also has
#          review → ALLOW (sentinel is trusted, not overridden)
# ============================================================
CASE_DIR="$TMPBASE/case10"
mkdir -p "$CASE_DIR/.claude/plans"
PLAN_A="$CASE_DIR/.claude/plans/plan-a.md"
REVIEW_A="$CASE_DIR/.claude/plans/plan-a.review.md"
PLAN_B="$CASE_DIR/.claude/plans/plan-b.md"
REVIEW_B="$CASE_DIR/.claude/plans/plan-b.review.md"
echo "# Plan A" > "$PLAN_A"
touch -t 202601010001 "$PLAN_A"
create_review "$REVIEW_A" "$PLAN_A"
touch -t 202601010002 "$REVIEW_A"
# Plan B is newer
echo "# Plan B" > "$PLAN_B"
touch -t 202601010003 "$PLAN_B"
create_review "$REVIEW_B" "$PLAN_B"
touch -t 202601010004 "$REVIEW_B"
# Sentinel points to older plan-A (intentional multi-plan workflow)
echo "$PLAN_A" > "$CASE_DIR/.claude/plans/.last-reviewed"
run_test "Case 10: Sentinel trusts plan-A, newer plan-B exists → ALLOW" "$CASE_DIR" "allow"

# ============================================================
# Summary
# ============================================================
echo "---"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
