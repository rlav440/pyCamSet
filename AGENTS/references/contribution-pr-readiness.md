# pyCamSet contribution and PR-readiness guide

**Purpose:** turn a pile of local commits into a small set of themed, evidence-backed
pull requests, and prove each one is ready before a human is asked to publish it. This
is the long form, written for an automated contributor that has to show its working.
Humans contributing by hand want [`CONTRIBUTING.md`](../../CONTRIBUTING.md), which says
the same things in a tenth of the space.

Nothing here is permission to publish. §0 is the rule that outranks the rest.

**Canonical location:** `AGENTS/references/` on the `development` branch. Keep this
workflow here; add pointers elsewhere rather than copying it.

`AGENTS/` is documentation about contributing, not part of the package. It ships no
code, and no PR should carry changes to it alongside changes to `pyCamSet/` — a diff
that edits both is two themes wearing one hat (§2.2).

## Hard stops and precedence

1. This guide prepares a local candidate. It does **not** push, open a PR, post a
   comment, publish a branch, or otherwise mutate an external system.
2. Explicit user approval is required before any push or PR. Approval is specific to
   the reviewed candidate head commit and exact diff; any fix, rebase, scope change, or
   PR-text change after approval invalidates it and requires fresh approval. `READY`
   means ready for that human decision, never authorized to publish.
3. Rule precedence is **safety > correctness > scope > brevity**.
4. A green test run is not a readiness verdict. pyCamSet has several ways for a run to
   be green and wrong — skipped tiers, a missing image corpus, a missing OpenGL
   context, a JIT setting that silences the tests that matter (§9.2). Name what ran and
   what did not. A run can also be green because of the machine it ran on rather than
   the code it ran against, which no ladder executed on that machine can detect
   (§9.3).
5. Use these evidence labels:
   - `VERIFIED-BY-EXECUTION` — command actually ran; retain exit code/output.
   - `VERIFIED-BY-READING` — directly checked in a named file or primary source.
   - `READ-ONLY-INFERENCE` — reasoned from evidence, not directly established.
   - `OPEN-GAP` — required fact, policy, approval, or decision is unresolved.

   Every readiness-record **evidence claim or evidence entry** must carry one of these
   labels; ordinary metadata fields may remain unlabeled, but an unlabeled factual
   assertion is not eligible for `READY`. `READ-ONLY-INFERENCE` may support a
   contextual statement; it cannot close a provenance, licensing, or
   required-validation item. Those need reading/execution evidence or `OPEN-GAP`.

## Repository facts this guide depends on

Re-verify each of these at preflight; they are current-source questions, not constants.

| Fact | Value | Source |
|---|---|---|
| Upstream | `rlav440/pyCamSet` | [`README.md`](../../README.md) badges, `pyproject.toml` `[project.urls]` |
| Contributor fork | your fork, usually `origin` | `git remote -v` |
| Base branch for PRs | `development` | every PR #19–#28 targets it (`gh pr list --json baseRefName`) |
| Release branch | `master` (upstream `HEAD` symref) | `git ls-remote --symref <upstream> HEAD` |
| License | Apache-2.0, whole repository | [`LICENSE`](../../LICENSE), `pyproject.toml` `license = "Apache-2.0"` |
| Test CI | 9-job matrix, two pytest tiers | [`.github/workflows/test.yml`](../../.github/workflows/test.yml) |
| Docs CI | `mkdocs build --strict` on every PR | [`.github/workflows/docs.yml`](../../.github/workflows/docs.yml) |
| Packaging CI | `python -m build` + `twine check --strict` on every PR | [`.github/workflows/release.yml`](../../.github/workflows/release.yml) |
| Lint/format/type CI | **none exists** | no ruff/flake8/black/mypy/pre-commit config in `pyproject.toml` or the repo |
| `CONTRIBUTING.md` | the short human form of this guide | [`CONTRIBUTING.md`](../../CONTRIBUTING.md) |
| `CODE_OF_CONDUCT.md` | **absent**, though `README.md` links it | same; the README badge line points at a file that is not in the tree |
| AI-use policy, DCO/CLA, PR template | **none found** | no `.github/PULL_REQUEST_TEMPLATE*`, no policy file at the resolved branch |

The remaining absences are `OPEN-GAP`, not permission. Do not invent a project rule to fill
one, and re-check each at every preflight — upstream guidance changes.

## Required references

- `AGENTS/references/operating-discipline.md` — evidence, absence-search,
  anti-fabrication, scope, and verification rules. **Not present in this repository at
  the time of writing**; record that as `OPEN-GAP` and apply the rules stated inline
  here until it exists.
- Upstream [`README.md`](../../README.md), [`mkdocs.yml`](../../mkdocs.yml),
  [`pyproject.toml`](../../pyproject.toml), and the three workflows under
  [`.github/workflows/`](../../.github/workflows/).
- [`tests/conftest.py`](../../tests/conftest.py) — the marker contract, the headless
  pinning, and every environment gate that decides which tests actually run.
- [`docs/architecture.md`](../../docs/architecture.md) and
  [`docs/internals/workflow.md`](../../docs/internals/workflow.md) — the subsystem
  boundaries a themed PR is drawn along.
- [`setup_scripts/write_core_requirements.py`](../../setup_scripts/write_core_requirements.py)
  and [`tests/test_core_requirements.py`](../../tests/test_core_requirements.py) — the
  generated-file invariant any dependency change must hold.

## Outputs and verdicts

Produce a short readiness record per candidate PR containing:

- exact repository path, remotes, base branch/commit, candidate head commit, current
  branch, and `git status --short --branch`;
- the theme statement: one sentence naming the single observable change;
- human owner identity and confirmation date/method (for AI-assisted work);
- independent reviewer identity/process, artifact head commit, and review date;
- issue/feature motivation and duplicate-search result;
- changed-path inventory with one disposition per path;
- subsystem key and cross-PR overlap result;
- provenance/licensing result for any added third-party material;
- install-tier result (§6);
- review findings and dispositions;
- commands, versions, real exit codes, and skipped/unavailable checks;
- one verdict and the next safe action.

Use these verdicts literally:

- `READY` — theme, scope, provenance, separation, review, and **every check required to
  validate the changed behaviour** pass on the exact candidate; no required check is
  unavailable. This verdict is waiting for explicit human publication approval.
- `NOT-READY` — a correctable code, test, scope, provenance, duplicate, or PR-text
  defect remains, including an unresolved path disposition.
- `OPEN-GAP` — a human, upstream-policy, licensing, or required-evidence question is
  unresolved, including a required validation check that could not run. Do not fill it
  with inference.
- `SPLIT-REQUIRED` — the candidate carries more than one theme, or collides with a
  concurrently-open PR, and must be separated before review.

## 1. Upstream preflight

Before shaping a candidate:

- Read upstream `README.md`, the three workflow files, `mkdocs.yml`, `pyproject.toml`,
  and any contribution instructions exposed by the repo, **at the resolved upstream
  branch** — not from the local checkout, which may be behind.
- Confirm the current base branch and exact remote. `master` is the release branch and
  upstream `HEAD`; `development` is where work lands. Do not infer the base from a
  stale local branch name.
- Resolve `<base>` and `<candidate>` explicitly. A candidate must be a committed head
  before `READY`; an uncommitted draft is `NOT-READY` and uses the working-tree diff
  form until a clean contribution commit exists.
- Verify `<base>` is an ancestor of `<candidate>` before using a three-dot inventory:
  `git merge-base --is-ancestor <base> <candidate>`. If it fails, record base drift and
  rebase only in an isolated contribution branch before re-inventorying.
- Search open and recently closed issues and PRs for the same topic, and check recent
  commits on `development`. A likely duplicate is a stop-and-report condition, not a
  reason to implement a second patch. Record each query, its result count, and the
  access date.
- Record URLs, revision/commit, and access date for every source. Search snippets are
  leads, not evidence.

```sh
git remote -v
git fetch <upstream> development
git ls-remote --symref <upstream> HEAD
git status --short --branch
git log -1 --oneline
git merge-base --is-ancestor <base> <candidate>
git diff --name-status <base>...<candidate>
git log --oneline <base>..<candidate>

gh pr list --repo rlav440/pyCamSet --state all --limit 40 \
  --json number,title,baseRefName,headRefName,state
gh issue list --repo rlav440/pyCamSet --state all --search "<topic>"
```

For an uncommitted draft, do not invent a candidate commit. Use `git diff <base> --`
and `git diff --cached` for triage, record the draft as `NOT-READY`, then review the
committed head again once a real contribution commit exists.

### 1.1 Upstream CI state at the base

Record whether the base commit is green upstream **before** attributing any local
failure to the candidate:

```sh
gh run list --repo rlav440/pyCamSet --branch development --limit 10 \
  --json databaseId,headSha,workflowName,conclusion,createdAt
gh run view <run-id> --repo rlav440/pyCamSet --json jobs \
  --jq '.jobs[] | {name, conclusion}'
```

Bind every result to the **full** base SHA, not to the branch name — a branch moves
between the fetch and the read. A red upstream job is evidence that a local failure may
pre-exist the candidate (§9.4); it is never a licence to skip a local check.

## 2. Themed PR planning: turning commits into pull requests

This is the step that decides whether the rest of the work is reviewable. Do it before
redrafting anything.

A **theme** is one observable change a maintainer can hold in their head: one problem,
one behaviour, one reviewable diff. "Everything I did this week" is not a theme. "Fix
the three GUI bugs I found" is not a theme either — it is three.

### 2.1 Inventory the commits

```sh
git log --oneline --no-merges <base>..<candidate>
git log --name-only --no-merges --format='--- %h %s' <base>..<candidate>
```

For every commit record: subject, touched paths, derived subsystem (§4), and the theme
you are assigning it to. A commit that lands in two themes is a commit that must be
split, not a reason to merge the themes.

### 2.2 Group into PRs

Rules, in precedence order:

1. **One subsystem per PR.** Derive the subsystem from the paths (§4.1), never from
   intent. A change touching `pyCamSet/gui/` and `pyCamSet/optimisation/` is two PRs
   unless the coupling *is* the change and is stated in the body.
2. **No file appears in two concurrently-open PRs.** This is the constraint that keeps
   a batch from conflicting with itself; see §4.2.
3. **Each PR must stand alone on `development`.** It must install and pass the check
   ladder (§9.1) with no other PR merged. If PR B needs PR A, say so with
   `Depends on: #<A>`, open A first, and do not open B until A merges.
4. **Tests ship with the behaviour they cover.** A behaviour PR with its tests in a
   different PR is split in the wrong direction — merge them.
5. **Mechanical churn is its own PR.** Formatting, renames, import sweeps, and
   generated-file regeneration go in a separate, obviously-mechanical PR so they do not
   hide a behaviour change in a 2000-line diff.
6. **Docs follow their subject.** A page describing new behaviour belongs in that
   behaviour's PR; a docs-only correction is its own PR.

Aim for a PR whose diff a reviewer can read in one sitting. If you cannot write the
`## Summary` in two sentences without the word "and", it is not one theme.

### 2.3 Build the branches

Each theme gets its own branch cut from the current `development`:

```sh
git fetch <upstream> development
git switch -c <theme-branch> <upstream>/development
git cherry-pick <commits for this theme>        # or: git rebase -i to split/reorder
```

After building each branch, re-derive its diff against the same base and confirm the
grouping actually held:

```sh
git diff --name-only <upstream>/development...<theme-branch>
```

A path you did not expect means the grouping is wrong, not that the list needs
widening. Re-plan rather than absorb it.

Record the plan as a table — theme, branch, subsystem, paths, dependencies, order — and
keep it with the readiness records. Every later section runs **per branch**.

## 3. Candidate inventory and filtering

Compare against the intended base commit, not against whatever the local branch has
accumulated. After the ancestry check succeeds:

```sh
git diff --name-status <base>...<candidate>
git diff --stat <base>...<candidate>
git diff --check <base>...<candidate>
git status --short
```

Enumerate staged, untracked, and deleted paths from `git status --short` as well. Every
path must receive exactly one disposition — `include`, `split`, `remove`, or
`investigate` — before `READY`. An `investigate` disposition is temporary: it must
resolve to one of the other three; unresolved investigation is `NOT-READY`.

Classify each path as:

- library source (`pyCamSet/**`), further keyed by subsystem (§4.1);
- tests (`tests/**`) and the image corpus (`tests/test_data/**`);
- documentation (`docs/**`, `mkdocs.yml`, `README.md`);
- build, packaging, and CI (`pyproject.toml`, `requirements*.txt`,
  `.github/workflows/**`, `.gitattributes`, `.gitignore`);
- developer tooling (`scripts/**`, `setup_scripts/**`);
- generated output (`pyCamSet/optimisation/template_functions/**`,
  `requirements_core.txt`, anything under `docs/build/`);
- third-party or vendored material;
- local working documentation (`AGENTS/**`) — never in an upstream candidate;
- unrelated local work.

Remove or split:

- unrelated local edits and opportunistic refactors;
- line-ending churn and generated binaries;
- secrets, credentials, temporary files, debug prints, conflict markers;
- `__pycache__/`, `.idea/`, `.coverage*`, `htmlcov/`, `bin/`, `dump/`, and anything
  else `.gitignore` already names — if one of these is in the diff, the ignore rule was
  bypassed, and that is the defect to fix;
- committed `.camset` files and `detected_datapoints*.pickle` caches under
  `tests/test_data/` — `.gitignore` excludes them because a stale cache outlives the
  OpenCV version that produced it and would hide exactly the detection changes
  `tests/test_detection_consistency.py` exists to catch;
- scratch directories, working notes, and review logs kept outside the package.

Review staged files individually. Never `git add -A` merely because the tree is
non-empty.

### 3.1 Line endings

`.gitattributes` normalises the repository to LF (`* text=auto`, plus explicit `eol=lf`
for `.py`, `.md`, `.toml`, `.yml`, and friends) — a rule added after the PR #18 style
review. On a Windows checkout a whole-file CRLF diff is a configuration fault, not a
change. Check before interpreting a large diff:

```sh
git diff --stat <base>...<candidate>
git diff --ignore-cr-at-eol --stat <base>...<candidate>
```

A large difference between those two is line-ending churn. Fix the checkout; do not
submit the churn.

### 3.2 Secret scan

```sh
git grep -n -E '(api[_-]?key|password|secret|token|private[_-]?key)' <candidate> --
git diff <base> <candidate> -- > <diff-output-file>
# Record git diff's exit code; only after exit 0:
grep -nE '(api[_-]?key|password|secret|token|private[_-]?key)' <diff-output-file>
```

The first scans the candidate tree; record its exit code (`1` means no matches, other
non-zero values mean a failed or unavailable scan). The second writes the diff before
scanning so a failed diff cannot be masked by a clean grep. The final `grep` exit `1`
means "no matching line" **only** when the preceding diff exited `0`; `grep` exit `2`
or a non-zero diff is a failed check. Treat matches as leads requiring review, not as
proof of a secret; any unresolved match blocks `READY`.

## 4. One PR per subsystem, and cross-PR file overlap

### 4.1 Subsystem map

Derive the key from the path. A candidate whose paths span more than one key is
`SPLIT-REQUIRED` unless the body states the coupling and the maintainer has been asked.

| Key | Paths |
|---|---|
| `cameras` | `pyCamSet/cameras/**` |
| `calibration` | `pyCamSet/calibration/**` |
| `targets` | `pyCamSet/calibration_targets/**` (incl. `core`, `charuco`, `ccube`, `puzzleboard*`) |
| `markers` | `pyCamSet/calibration_targets/markers/**` |
| `optimisation` | `pyCamSet/optimisation/**` |
| `reconstruction` | `pyCamSet/reconstruction/**` |
| `workflow` | `pyCamSet/workflow/**` (incl. `tuning`) |
| `gui` | `pyCamSet/gui/**` |
| `utils` | `pyCamSet/utils/**` |
| `docs` | `docs/**`, `mkdocs.yml`, `README.md`, `scripts/doc_scenes.py`, `scripts/griffe_rest_roles.py` |
| `build` | `pyproject.toml`, `requirements*.txt`, `.github/workflows/**`, `.gitattributes`, `.gitignore`, `setup_scripts/**` |
| `tests` | `tests/**` — **not** a standalone key; tests inherit the key of the code they cover |

`markers` is listed separately from `targets` because the ArUco backend registry is
independently replaceable (PR #24 isolated it) and is the most common cross-cutting
edit inside the targets tree. If a path matches no row, stop and record the gap rather
than guessing a key.

### 4.2 Overlap check

Before submitting, and again immediately before publication, check the candidate's file
set against **every other open PR** on upstream:

```sh
# every open PR and its files
for pr in $(gh pr list --repo rlav440/pyCamSet --state open --json number --jq '.[].number'); do
  echo "--- PR #$pr"
  gh pr view "$pr" --repo rlav440/pyCamSet --json files --jq '.files[].path'
done

# this candidate's file set
git diff --name-only <base>...<candidate>
```

Any intersection, or any same-subsystem collision, is `SPLIT-REQUIRED` or a
serialisation requirement: merge or close one before opening the next. PRs #22–#28
landed cleanly as a batch because the branches were disjoint by construction; that is
the property to preserve, and it has to be re-checked after every rebase, because a
rebase can pull in files the plan never assigned.

Resolve each PR to its current head and base SHAs when doing this — an overlap computed
from a branch name that has since moved is not evidence.

## 5. Licensing, provenance, and third-party material

pyCamSet is Apache-2.0 throughout, so there is no internal licence boundary to police.
What still needs a record is anything entering the repository from outside it.

For each added third-party code path, algorithm implementation, image, fixture, sample
dataset, icon, or documentation excerpt, record:

- source URL or file;
- access date;
- actual licence and terms (not "free" or "public");
- how it was used; and
- any restrictions, attribution, or notice obligations.

Specific live cases:

- **PuzzleBoard.** The target implementation is an optional dependency pulled from
  `git+https://github.com/PStelldinger/PuzzleBoard.git` (Peer Stelldinger and the HAW
  Hamburg authors). Any change to how it is used, vendored, or attributed must re-check
  its licence at the pinned revision and keep the README attribution accurate. It must
  stay an optional extra — see §6.
- **The image corpus.** `tests/test_data/` is roughly 44 MB of checked-in calibration
  images. Adding to it needs an explicit provenance record — who captured it, under what
  terms — and a size justification; it is fetched by every CI job on every platform.
- **Generated material.** `pyCamSet/optimisation/template_functions/**` is written by
  `abstract_function_blocks.make_*` and is `.gitignore`d. It must not appear in a
  candidate diff. `requirements_core.txt` is generated too, but *is* tracked — see §6.
- **Dependency additions.** A new runtime dependency changes what every downstream user
  installs. Record its licence, and check §6 before adding it anywhere.

If any of these is unresolved, the verdict is `OPEN-GAP`, not `READY`. No upstream DCO,
CLA, copyright-assignment process, or contributor-clearance rule was found at the
resolved branch (see the facts table); keep that status `OPEN-GAP` and do not infer that
employer-owned, contractor-owned, third-party, or model-produced material can be
licensed by the contributor. The human owner must be able to state their authority to
submit the change.

## 6. Install-tier and dependency-boundary gate

pyCamSet ships four install tiers, and CI keeps one job on the lean one. A dependency
change that ignores the tiers is green locally and broken for users.

| Tier | What it is | Constraint |
|---|---|---|
| lean | `pip install pyCamSet --no-deps` + `requirements_core.txt` | everything except the GUI toolkit must work, plotting included |
| default | `pip install pyCamSet` | adds PySide6; the `pycamset` entry point works |
| extras | `[viz]`, `[optimisation]`, `[puzzle]`, `[dev]`, `[docs]` | each import must be guarded; absence must degrade, not crash |
| `[all]` | defined in terms of the others | never enumerate the extras again by hand |

Rules:

- **`requirements_core.txt` is generated.** Change `[project] dependencies` and you must
  regenerate it, or `tests/test_core_requirements.py::test_the_file_matches_pyproject`
  fails:

  ```sh
  python setup_scripts/write_core_requirements.py
  ```

- **No direct-URL runtime dependency.** PyPI rejects a distribution whose
  `Requires-Dist` carries a direct URL, and the release workflow refuses one at tag time
  by design. `puzzle_board` is a direct URL, which is exactly why it is an optional
  extra and must stay one.
- **Plotting is core, the GUI is not.** `pyvista` and `matplotlib` must survive the lean
  install; `PySide6` must not be in it. Both are asserted in
  `tests/test_core_requirements.py`.
- **Optional imports are guarded at the call site.** `open3d`, `optuna`, `aruco2`, and
  `puzzle_board` are absent in at least one supported environment. A module-level import
  of one of these makes the file uncollectable, not skippable — `tests/conftest.py` has
  to carry a `collect_ignore_glob` entry for `test_aruco2_backend.py` for exactly that
  reason.
- **OpenCV floor and ceiling.** `opencv-python>=4.8` with no upper bound is deliberate;
  the 4.x/5.x differences are handled at the call sites. Do not add an upper bound to
  make a test pass — see §9.2.

If the candidate changes any dependency, run the lean install path as part of
verification, not just the default one.

## 7. Redraft the PR for the maintainer

Write a PR that can be understood without reconstructing how it was produced. One
observable change, one problem, one reviewable diff.

### 7.1 Commit messages

The repository's own convention is a prose imperative sentence saying what the commit
makes true — *"Measure coverage on the code that ships"*, *"Gate the two tests that
render on whether this machine can"*, *"Make a tag prove itself before it publishes"*.
No `feat:`/`fix:` prefix, no scope parentheses, no ticket number.

An earlier batch used `feat:`/`fix:` **titles**, which is a drift from the
maintainer's own history rather than a convention to copy. Match the repository. Verify
before writing:

```sh
git log --format='%s' <upstream>/development -40
```

Keep commits coherent and explainable. After a rebase, obtain fresh approval under Hard
stop 2 for the new head before any force-push, and use `--force-with-lease` only.

### 7.2 Title

A precise, searchable sentence describing the change and its domain, in the same voice
as the commits. Do not claim a broader result than the tests establish.

### 7.3 Body template

```md
## Summary
<One or two sentences: what changes and why. If you need "and", re-read §2.2.>

## Motivation / issue
<Observed problem, user impact, and link to the existing issue or discussion.>

## Scope
### Included
- <paths and behaviour included>
### Deliberately excluded
- <related work left out, or "none">

## Subsystem and dependencies
- Subsystem: <key from §4.1>
- Overlap check: <clean against open PRs #…, date>
- Depends on: <PR number, or "none">

## Implementation
<Short map from behaviour to files. Explain non-obvious choices and trade-offs.>

## Install tiers and dependencies
- Dependency change: <none | added X (licence) | moved X between tiers>
- requirements_core.txt: <unchanged | regenerated>
- Lean install exercised: <yes/no, command>

## Verification
| Evidence type | Command | Result | Evidence / limitation |
|---|---|---:|---|
| `VERIFIED-BY-EXECUTION` | `pytest -m "not data" --timeout=600` | `0` | <passed/skipped counts> |
| `VERIFIED-BY-EXECUTION` | `python -m pytest --timeout=1800` | `0` | <counts; name every skip reason> |
| `VERIFIED-BY-EXECUTION` | `python -m mkdocs build --strict` | `0` | <or `not run` → OPEN-GAP> |
| `VERIFIED-BY-READING` | `<file, revision>` | `checked` | <claim and scope> |

## Environment
<python, numpy, cv2, numba versions — the output of the CI's own probe. A
platform-specific result is not portable evidence without them.>

## What did not run
<Every tier, marker, or matrix cell not exercised locally, and why. `not run` is
OPEN-GAP, never a silent pass.>

## Review notes
<Known trade-offs, open gaps, questions for the maintainer, suggested review path.>

## Linked work
<Issue, dependent PR, duplicate-check result.>
```

Every body claim must map to a changed file, a named source, or an executed check.
Delete unsupported claims or mark them `OPEN-GAP`.

### 7.4 Detailed, and therefore short

Completeness and length are not the same thing, and it is length the maintainer pays
for. A body that records every check, every gap and every trade-off in the fewest
words that still carry them is doing its job; the same content at three times the
size is not more rigorous, it is less likely to be read to the end, and the one line
that mattered — the unverified branch, the open question — is what gets lost.

Write the long version if it helps you think, then cut it. In particular:

- The template's headings are a checklist, not a quota. A heading with nothing to say
  under it should say so in a clause, or go.
- Say a thing once. Evidence belongs in the verification table; do not narrate it
  again in prose above.
- Numbers instead of adjectives. "877 passed, same failure set as base" beats a
  sentence about thorough testing, and is shorter.
- Cut the process. How the change was arrived at, what was tried first, and how long
  it took are not review inputs.
- Keep every `OPEN-GAP`, every limitation, and every question. These are the last
  things to cut, not the first — brevity that removes them is dishonesty with better
  formatting.

Rule precedence puts brevity last (§0) for a reason: if shortening a body would drop
a caveat, keep the caveat and cut elsewhere.

### 7.5 Make review easy

- Explain the user's problem before the implementation detail.
- Show exact commands and real exit codes, including the checks that did not run.
- Name every skipped test tier and why it skipped — `data`, `gui`, `needs_opengl`, and
  `needs_jit` skips are normal and must be visible, not hidden inside a "passed" count.
- Call out generated files, compatibility risk, and test limitations.
- Ask focused questions instead of prescribing that the maintainer accept the design.
- Respond to feedback specifically, update the description when scope changes, and thank
  the reviewer. Do not argue from model authority.
- Do not repeatedly bump a maintainer; one concise status message is enough.

### 7.6 Submission cadence

Batched submission is acceptable but governed: keep a batch small enough that a
maintainer can review it, re-run the overlap check (§4.2) at every batch boundary, and
write a PR body that references **that PR's** content — identical boilerplate across a
batch is a defect. Already-opened PRs are never rolled back; if an outcome is unknown,
look it up before retrying.

## 8. AI-assisted contribution gate

The upstream repository's AI-use policy, if any, must be checked at review time. No
policy file was found at the resolved branch (facts table); record that as `OPEN-GAP`
and follow any human instruction. Do not infer permission or prohibition from silence.

If a resolved upstream policy prohibits AI-assisted contributions, set `NOT-READY` (or
`OPEN-GAP` pending a maintainer decision), do not request approval, and rework or
abandon the candidate. A policy requiring *disclosure* is a different thing from one
prohibiting the *contribution*.

Regardless of policy, an AI-assisted candidate must:

- have a human owner who understands and can explain every changed path;
- be independently checked against the upstream source and the intended issue; and
- include real tests and review evidence rather than a model self-report.

Do not paste confidential, restricted, or unlicensed material into a prompt or issue. Do
not claim generated code is original merely because a model produced it.

## 9. Independent review and executable verification

Use a reviewer or process different from the generator, and — where the reviewer is a
person — different from the human owner. A second agent profile or automated checker may
be the documented non-human process when no second human is available; a human owner
reviewing their own candidate is not an independent review. Review the actual candidate
diff and the surrounding source, not just the draft PR text. Record reviewer identity,
process, artifact head commit, and date, and record each finding with path/line,
severity, evidence, required correction, and disposition. Re-review after fixes; two
quiet rounds do not prove correctness.

Before verifying, fetch the current upstream base and re-check ancestry:

```sh
git fetch <upstream> development
git merge-base --is-ancestor <base> <candidate>
```

If ancestry fails, rebase only in an isolated contribution branch, then recompute and
re-review the diff.

### 9.1 The local check ladder

Run these on the exact candidate commit, from a clean tree. They mirror the upstream
workflows; nothing here is a script that must be written first.

```sh
# 0. environment probe — record this with every result
python -c "import sys, numpy, cv2, numba; print('python', sys.version); \
print('numpy', numpy.__version__); print('cv2', cv2.__version__); print('numba', numba.__version__)"

# 1. the install under test
pip install -e . --no-deps
pip install -r requirements_core.txt
pip install "pytest>=7.0" pytest-timeout pytest-cov

# 2. Qt starts (skip on the lean path)
QT_QPA_PLATFORM=offscreen python -c "import PySide6; from PySide6.QtWidgets import QApplication; QApplication([]); print('PySide6', PySide6.__version__, 'ok')"

# 3. fast tier — no image corpus, no display, JIT on
pytest -m "not data" --timeout=600 --cov --cov-report=term-missing

# 4. full suite
python -m pytest --timeout=1800

# 5. targeted regression for the changed behaviour
python -m pytest tests/<the tests that cover this change> -v

# 6. docs, if docs/, mkdocs.yml, or any documented docstring changed
pip install -e ".[docs]"
python -m mkdocs build --strict

# 7. packaging, if pyproject.toml or requirements changed
python -m build
python -m twine check --strict dist/*

# 8. generated-file invariant, if dependencies changed
python setup_scripts/write_core_requirements.py
git diff --exit-code requirements_core.txt
```

Steps 3, 4, 6 and 7 correspond 1:1 to jobs that run on every pull request. Step 6 is
required whenever a docstring on a documented API changes, because `mkdocstrings`
renders the reST `:param:` entries and `--strict` turns a warning into a failure.

There is **no lint, format, or type-check stage** — none is configured upstream. Do not
introduce one inside a behaviour PR; that is a `build`-subsystem change of its own
(§2.2 rule 5).

`requirements_dev.txt` still names the sphinx toolchain that the mkdocs move replaced,
and omits `pytest-timeout` and `pytest-cov`. Install from the ladder above or from the
`pyproject.toml` extras, not from that file, and record the staleness rather than
silently working around it.

The CI matrix is 9 jobs — 3 platforms × Python 3.11/3.12 at `opencv-python>=5`, plus
three explicit `<5` jobs (ubuntu, windows, macos-14 on 3.12), plus a lean ubuntu/3.11
job. A local run covers one cell. Say which cell, and mark the rest `OPEN-GAP` relying
on upstream CI.

### 9.2 Traps that make a green run lie

Each of these is documented in the repository and has already cost a real failure.

- **`NUMBA_DISABLE_JIT` must stay off.** Setting it makes the `@njit` kernel bodies
  visible to coverage, and in exchange skips every `needs_jit` test — including the
  guard against a kernel reading past its input buffer, the fault that silently dropped
  69 of 183 Jacobian columns on arm64. With JIT off, `njit` returns the plain function,
  so anything reaching for `.py_func` cannot fail the way it is asserted to. Do not
  re-enable it to flatter a coverage number.
- **Coverage reads low by construction.** `compiled_helpers.py` reports about 14% under
  JIT rather than the roughly 46% an interpreted run shows, and the project total sits
  near 33%. Those kernels are covered behaviourally by the bundle adjustments in the
  `data`-marked tests. A coverage delta is not a review finding on its own.
- **OpenCV 4 vs 5 shifts detected ChArUco corners by about half a pixel.** Calibrations
  are not numerically comparable across the majors. A tolerance that passes on one major
  and fails on the other is a test whose tolerance needs justifying per major — not a
  reason to add an upper bound to `opencv-python`.
- **A missing image corpus skips the whole calibration path silently.**
  `tests/test_data/` gates every `data`-marked test — the largest marker group in the
  suite — through the `data_dir` fixture. A shallow clone or an sdist install produces a
  green run that exercised almost none of that path. Always report the skip count from
  the run itself rather than assuming it.
- **No OpenGL context skips the rendering tests.** `conftest.opengl_is_available()`
  probes in a subprocess because a machine that cannot make a context does not raise —
  it faults, taking the session with it. Three tests carry `needs_opengl`.
- **No PySide6 skips every `gui`-marked test**, the second-largest marker group. That
  is the lean path and a real CI job — but a local run on the lean install has not
  tested the GUI, and must not be reported as though it had.
- **The legacy ChArUco calibration API is absent across the whole supported OpenCV
  band**, so `bundle_correctness_test.py` skips wholesale. That is expected; do not
  "fix" it by pinning OpenCV.
- **Tests run in a temporary directory with a redirected config directory.**
  `isolated_cwd` and `isolated_user_config` are autouse. A test that depends on the
  repository root, or on the developer's real settings, is the bug.
- **The RNG is seeded** (`deterministic_rng`, seed `20260909`). A test that passes only
  under one seed is not passing.

### 9.3 The trap the ladder cannot catch: a test that encodes your machine

Every trap in §9.2 is a run that is green and wrong. This is the other direction: a
test that passes locally for a reason that is not the code, and fails on CI for a
reason that is not the change.

A real instance, and the shape to recognise. A helper registers the conda
`Library/bin` directory so `cairosvg` can find the native Cairo library, and its
interesting half — the registration — never runs on a machine that already finds
Cairo. To exercise it, the test forced the first `import cairosvg` to fail, then
asserted the helper recovered:

```python
assert ok, "the helper gave up where it should have recovered"
```

That is true on the conda layout it was written on, and false on CI. GitHub's Windows
runners are a stock CPython with no `Library/bin` to register and no native Cairo to
find, so the helper correctly reported failure and the test called that a bug. Three
Windows jobs red, for a machine difference the author's machine cannot produce.

Note what did *not* prevent it. The full suite passed. The fast tier passed. The
docs build passed. The candidate was `READY` by every check in §9.1, honestly run.
A ladder cannot catch this, because the ladder runs on the machine that holds the
assumption.

So it has to be caught while writing the assertion. Before asserting an outcome, ask
what about the machine makes it true:

- **Assert the mechanism, not the machine's capability.** That the retry happened,
  that the directory was registered when there was one to register, that the return
  value matches what the environment can actually deliver — all hold everywhere. That
  the recovery *succeeded* holds only where the missing piece exists.
- **Make the expectation conditional on the thing it depends on**, and say so:
  `if os.path.isdir(expected): ... else: assert registered == []`. A conditional
  assertion that names its condition is documentation; a skip is a hole.
- **Anything derived from `sys.prefix`, `PATH`, a native library, a GPU, a drive
  letter, or an installed optional package is a machine fact**, not a code fact.
  Native-library and DLL-search behaviour is the most common source, because it is
  the part of the environment a developer never configured deliberately.
- **A test written specifically because a branch is hard to reach locally deserves
  the most suspicion**, not the least. It exists because the local machine is not
  representative; that is the premise, so do not then assert the local machine's
  answer.

When the honest answer really is environment-dependent, say which environment in the
PR body under *What did not run* rather than asserting past it. `OPEN-GAP` on a
platform you do not have costs a maintainer nothing; a red CI job costs them a
review cycle.

### 9.4 Failure triage

A local failure is a **regression** unless proven otherwise. To claim it pre-exists the
candidate, show it at the base:

```sh
git switch --detach <base>
python -m pytest <the failing test> -v          # record exit code and output
git switch --detach <candidate>
python -m pytest <the failing test> -v
```

Both runs must use the same environment probe (§9.1 step 0). A failure that reproduces
at the base is disclosed in the PR body with that evidence; a failure that does not
blocks submission. An upstream red job (§1.1) is supporting evidence, not a substitute
for the replay, and never blesses a failure in a subsystem the candidate touches.

If a check cannot run at all — no display, no corpus, no platform — name the check and
mark it `OPEN-GAP`/`NOT-READY`. A missing tool is an environment error, never an
approvable gap.

### 9.5 Minimum evidence record

```text
candidate: <repository, branch, base, head>
theme: <one sentence>
subsystem: <key>
status: <git status --short --branch>
human_owner: <identity, confirmation method/date, or OPEN-GAP>
reviewer: <independent reviewer identity/process, artifact head, date, or OPEN-GAP>
duplicate_check: <queries, result counts, access date>
path_dispositions: <all paths resolved to include/split/remove>
overlap_check: <open PRs compared, result, date>
provenance: <third-party material and licence result, or none>
install_tiers: <lean/default/extras result>
environment: <python, numpy, cv2, numba>
review_findings: <report and dispositions>
checks:
  - command: <exact command>
    label: VERIFIED-BY-EXECUTION | VERIFIED-BY-READING | READ-ONLY-INFERENCE | OPEN-GAP
    exit: <integer>
    result: <passed/failed/skipped counts>
    evidence: <path or captured output>
not_run:
  - <check, tier, or matrix cell, and why>
open_gaps:
  - <explicit unresolved item, or none>
verdict: READY | NOT-READY | OPEN-GAP | SPLIT-REQUIRED
publication: STOPPED — explicit user approval still required
```

## 10. Final gate before asking to publish

- [ ] The candidate is one theme, stated in one sentence without "and".
- [ ] Duplicate and scope checks happened before redrafting **and were repeated
      immediately before requesting approval**, with queries, counts, and date.
- [ ] Every changed path has a resolved disposition; no `investigate` remains.
- [ ] No `AGENTS/`, scratch, cache, generated, or `.gitignore`d path is in the diff.
- [ ] Line endings are LF; no CRLF churn.
- [ ] Secret scan ran with recorded exit codes; no unresolved match.
- [ ] Subsystem key derived from paths; overlap check clean against every open PR, run
      after the final rebase.
- [ ] Provenance recorded for any added third-party material; PuzzleBoard is still an
      optional extra and no runtime dependency carries a direct URL.
- [ ] `requirements_core.txt` regenerated if dependencies changed, and
      `tests/test_core_requirements.py` passes.
- [ ] Fast tier and full suite passed on the exact candidate, with skip counts and
      reasons recorded and the environment probe captured.
- [ ] Targeted regression tests exist for every changed behaviour and were run.
- [ ] `mkdocs build --strict` passed if docs or documented docstrings changed;
      `twine check --strict` passed if packaging changed.
- [ ] Every failure is either absent or replayed at the base with evidence; no silent
      red, and no `NUMBA_DISABLE_JIT` workaround.
- [ ] Matrix cells not exercised locally are listed as `OPEN-GAP`.
- [ ] Commit messages match the repository's prose-imperative convention.
- [ ] PR text is narrow, factual, and maps every claim to evidence; the "what did not
      run" section is present and honest.
- [ ] Independent review findings are disposed; no blocker remains.
- [ ] Human owner recorded, with an explicit authority-to-submit affirmation; if the
      owner or an independent reviewer is unavailable, the verdict is downgraded and the
      gap is visible.
- [ ] No push, fork, PR, or public comment has occurred.
- [ ] The next step is explicit user approval, not an implied authorization.

## 11. What this guide deliberately does not automate

Stated so the omissions are visible rather than assumed covered. None of these exists
for pyCamSet, and none should be cited as if it did:

- a scripted pre-submit gate reproducing the CI matrix locally;
- a machine-readable result manifest, attestation, or ratification artifact;
- a file-set reservation or locking mechanism across concurrent PRs — §4.2 is a manual
  check, and it is only as good as the moment it was run;
- a known-failures registry — §9.4 replaces it with a per-failure base replay;
- a publication wrapper; publication is a human action taken after approval;
- a post-submission monitoring loop — watch a PR's checks with
  `gh pr checks <n> --repo rlav440/pyCamSet` and escalate by hand.

If one of these is built later it belongs under `AGENTS/`, or as its own
`build`-subsystem PR, and this section is where its existence gets recorded.

## Primary references

Re-resolve the upstream branch before each contribution; these identify repository paths
and a current-source check, not permanent branch names.

- https://github.com/rlav440/pyCamSet — `README.md`, `pyproject.toml`, `mkdocs.yml`, and
  `.github/workflows/{test,docs,release}.yml` at the resolved branch.
- https://github.com/rlav440/pyCamSet/issues — duplicate and motivation search.
- https://rlav440.github.io/pyCamSet/dev/ — the development documentation build.
- https://docs.github.com/en/pull-requests
- https://opensource.guide/how-to-contribute/
- https://devguide.python.org/getting-started/pull-request-lifecycle
- https://devguide.python.org/getting-started/ai-tools

This guide does not claim upstream has a `CONTRIBUTING.md`, a code of conduct file, an
AI-use policy, a DCO/CLA, or a PR template. Those are current-source questions and
remain `OPEN-GAP` until verified at the resolved branch.
