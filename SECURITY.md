# Security Policy

## Supported Versions

Only the latest published release of `diff-diff` receives security fixes.
Older releases are not patched.

| Version | Supported |
| ------- | --------- |
| Latest release | Yes |
| All older releases | No |

## Reporting a Vulnerability

If you have discovered a security vulnerability in `diff-diff`, please report
it privately rather than opening a public issue.

**Preferred channel: GitHub private vulnerability reporting.**
Open the [Security tab](https://github.com/igerber/diff-diff/security) of this
repository and click "Report a vulnerability." This keeps the report private
between you and the maintainer.

When reporting, please include:

- A description of the issue and the surface it affects (Python API, Rust
  extension, build pipeline, etc.).
- Steps to reproduce, ideally with a minimal code sample or input data.
- The version of `diff-diff` and Python you tested against.
- Any suggested mitigation, if you have one.

## Response Expectations

This project is maintained by a single individual. As a guideline:

- **Triage**: within 7 business days of receipt.
- **Fix or mitigation timeline**: communicated after triage; depends on
  severity and complexity.

If you do not receive an acknowledgement within 7 business days, please feel
free to send a follow-up via the same private reporting channel.

## Scope

In scope:

- The `diff_diff` Python package.
- The bundled Rust extension under `rust/`.
- Build and release infrastructure under `.github/workflows/`.

Out of scope:

- Vulnerabilities that require an attacker to already control the Python
  interpreter or local filesystem.
- Issues in transitive dependencies (please report to the upstream project;
  Dependabot handles automated patching here).
- Numerical correctness questions or methodology disagreements (please open
  a regular issue or discussion instead).
