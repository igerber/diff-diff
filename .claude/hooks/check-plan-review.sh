#!/bin/bash
# PreToolUse hook for ExitPlanMode: ensure a plan review exists before approval.
#
# Output protocol: PreToolUse hooks must output JSON to stdout and exit 0.
#   Allow: exit 0 (no output, or JSON with permissionDecision: "allow")
#   Deny:  exit 0 with JSON { hookSpecificOutput: { permissionDecision: "deny", ... } }
#   Error: exit 2 means hook error (not a deliberate block) — avoid this.
#
# Strategy:
#   1. Read ~/.claude/plans/.last-reviewed sentinel (written by review step)
#   2. If sentinel exists, use its contents as the plan path
#   3. If no sentinel, fall back to most recent .md in ~/.claude/plans/
#   4. Check for .review.md in ~/.claude/plans/ (by plan basename) — deny if missing
#   5. Check staleness — deny if plan is newer than review
#
# Known limitations:
#   - The ls -t fallback (step 3) can pick the wrong plan if multiple files exist.
#   - A stale sentinel can approve an unreviewed newer plan. This occurs only when:
#     (a) the sentinel points to an older plan with a valid review, AND
#     (b) a newer plan was created without updating the sentinel.
#     CLAUDE.md instructs sentinel updates on new plan creation (Plan Review section), making (b)
#     unlikely in practice. A runtime guard was tested and removed (cb8d5e3) because it
#     caused false denials in multi-plan workflows. Automated tests in
#     test-check-plan-review.sh verify existing behavior; full mitigation would require
#     tool-level integration (not hook-level).
#   - The -nt comparison has 1-second granularity on macOS. A plan edited and
#     reviewed within the same second could produce a false "fresh" result. In
#     practice, reviews always take longer.
#   - Review files are derived from plan basename only. Two plans with the same
#     filename in different directories would map to the same review file. Claude
#     Code always creates plans in ~/.claude/plans/, so this is unlikely.
#
# Dependencies: None (uses printf for JSON output, no jq required).

deny() {
  # Output JSON deny decision to stdout, then exit 0 (not exit 2)
  # Sanitize message: escape double quotes and backslashes for valid JSON
  local msg
  msg=$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}' "$msg"
  exit 0
}

# Validate that a review file's YAML plan: field matches the expected plan path.
# Usage: validate_review_plan_field <review-file> <expected-plan-path>
# Returns 0 if match, 1 if mismatch. Returns 1 if plan: field is missing (safe default).
validate_review_plan_field() {
  local review_file="$1" expected="$2"
  local yaml_plan
  # Extract YAML frontmatter (lines 2 through next ---), then grep for plan: field.
  # Uses a pipeline instead of nested sed braces for BSD sed (macOS) compatibility.
  yaml_plan=$(sed -n '2,/^---$/p' "$review_file" | grep '^plan:' | head -1 | sed 's/^plan:[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//;s/^'"'"'//;s/'"'"'$//')
  # Expand ~ in the YAML value
  yaml_plan="${yaml_plan/#\~/$HOME}"
  [ "$yaml_plan" = "$expected" ]
}

PLANS_DIR="$HOME/.claude/plans"
SENTINEL="$PLANS_DIR/.last-reviewed"

# Step 1-2: Try sentinel first
if [ -f "$SENTINEL" ]; then
  PLAN_FILE=$(head -1 "$SENTINEL" 2>/dev/null)
  # Expand ~ if present
  PLAN_FILE="${PLAN_FILE/#\~/$HOME}"
  if [ -n "$PLAN_FILE" ] && [ -f "$PLAN_FILE" ]; then
    PLAN_BASENAME=$(basename "$PLAN_FILE")
    REVIEW_FILE="$PLANS_DIR/${PLAN_BASENAME%.md}.review.md"
    if [ -f "$REVIEW_FILE" ]; then
      if ! validate_review_plan_field "$REVIEW_FILE" "$PLAN_FILE"; then
        deny "Review file $REVIEW_FILE is for a different plan (expected: $PLAN_FILE). Re-run the review."
      fi
      # Deny if plan was modified after its review (stale review)
      if [ "$PLAN_FILE" -nt "$REVIEW_FILE" ]; then
        deny "Plan review is stale: $PLAN_FILE was modified after $REVIEW_FILE. Re-run the review before approval."
      fi
      exit 0  # Review exists and is fresh, allow
    else
      deny "No plan review found for: $PLAN_FILE. Expected: $REVIEW_FILE. Run a plan review before presenting for approval."
    fi
  fi
fi

# Step 3: Fall back to most recent plan file
PLAN_FILE=$(ls -t "$PLANS_DIR"/*.md 2>/dev/null | grep -v '\.review\.md$' | head -1)

if [ -z "$PLAN_FILE" ]; then
  # No plan files at all — allow ExitPlanMode (not a plan-mode session)
  exit 0
fi

# Step 4-5: Check for review and staleness
PLAN_BASENAME=$(basename "$PLAN_FILE")
REVIEW_FILE="$PLANS_DIR/${PLAN_BASENAME%.md}.review.md"

if [ -f "$REVIEW_FILE" ]; then
  if ! validate_review_plan_field "$REVIEW_FILE" "$PLAN_FILE"; then
    deny "Review file $REVIEW_FILE is for a different plan (expected: $PLAN_FILE). Re-run the review."
  fi
  if [ "$PLAN_FILE" -nt "$REVIEW_FILE" ]; then
    deny "Plan review is stale: $PLAN_FILE was modified after $REVIEW_FILE. Re-run the review before approval."
  fi
  exit 0  # Review exists and is fresh, allow
else
  deny "No plan review found. Expected: $REVIEW_FILE. Run the plan review workflow: offer review via AskUserQuestion, spawn /review-plan if requested, or write a Skipped marker. See CLAUDE.md Plan Review section."
fi

# Manual verification checklist:
#   1. No sentinel, no plans: ExitPlanMode should ALLOW (not a plan session)
#   2. Sentinel points to plan with fresh review: ALLOW
#   3. Sentinel points to plan with stale review: DENY
#   4. Sentinel points to plan with no review: DENY
#   5. Sentinel stale (wrong plan), review exists for fallback plan: validate plan: field
#   6. No sentinel, fallback to most recent plan with review: ALLOW
#   7. No sentinel, fallback to most recent plan without review: DENY
#   8. Review file plan: field doesn't match plan path: DENY
#   9. Review file has no plan: field (e.g., old format): DENY (empty string != plan path)
