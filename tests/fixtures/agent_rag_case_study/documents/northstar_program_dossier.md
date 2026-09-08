# Northstar Commerce Modernization Program Dossier

## Document control and reading guide

This dossier consolidates steering decisions, operating assumptions, dependency notes, and delivery commitments for the fictional Northstar commerce modernization program. It is intentionally written like a long internal program record: important facts are surrounded by related but non-identical details, dates recur in different contexts, and several teams share similar responsibilities. Evidence labels are benchmark annotations and are not business identifiers.

## Program mandate

Northstar replaces the legacy merchant onboarding portal, settlement adapter, and account review workflow. The program began after regional teams reported inconsistent onboarding lead times and duplicate manual checks. Its executive sponsor is the Chief Operating Officer, while delivery ownership remains distributed across platform, risk, data, and support teams. The scope explicitly excludes warehouse robotics, consumer mobile checkout, and payroll systems.

## Baseline milestones

The initial plan used three release trains. Foundation was scheduled for February, migration tooling for April, and controlled rollout for June. These dates are planning anchors rather than approval dates. Teams must not infer production authorization from a completed development milestone, because release approval is recorded separately by the governance office.

## Atlas gateway delay

[EVIDENCE:NS-101] The Atlas gateway missed its 2026-02-14 integration milestone because the external payment provider changed the token-exchange contract twice and the shared certification environment was unavailable for nine business days. The steering group rejected the proposal to bypass certification. It moved the integration milestone to 2026-03-08 and required contract tests before every release candidate.

The gateway team had previously reported smaller delays caused by test-data refreshes, but those were not the cause of the February milestone miss. A separate user-interface accessibility review continued on schedule and did not block the gateway. This distinction is important because several weekly summaries use the generic phrase “environment issue.”

## Atlas recovery decision

[EVIDENCE:NS-102] On 2026-02-20, the architecture council froze Atlas token schema version 3.2, assigned contract-test ownership to Priya Nair, and approved a recovery budget ceiling of USD 185,000. The budget covers certification support and temporary test infrastructure; it does not authorize additional production vendors.

Finance records also contain an earlier USD 150,000 planning reserve. That reserve was superseded by the council decision and must not be reported as the approved ceiling. Priya owns contract-test delivery, while release approval belongs to a different role described later in this dossier.

## Beacon analytics dependency

[EVIDENCE:NS-201] Beacon’s operational dashboard was delayed when the Ledger and Fulfillment source teams delivered conformed event tables eleven days late. The dashboard code itself passed acceptance tests. Miguel Santos owns the daily freshness review, and the revised dashboard availability date is 2026-04-17.

Beacon also consumes a marketing taxonomy feed, but that feed was delivered on time and was not on the critical path. The team retained the original visualization library after a short evaluation; no library migration contributed to the delay.

## Beacon remediation controls

[EVIDENCE:NS-202] For the first six weeks after launch, Beacon must run freshness validation at 07:30 UTC and publish a signed exception note by 09:00 UTC whenever either source is stale. Morgan Lee is the escalation owner, while Miguel Santos remains accountable for the daily review.

The signed exception note is a control artifact, not a replacement for the incident process. Repeated freshness failures still require an incident ticket and a weekly trend review.

## Cygnus knowledge pilot

[EVIDENCE:NS-301] Cygnus is the support knowledge-search pilot. It shipped the hybrid retrieval prototype on 2026-01-29. The pilot combines semantic dense retrieval with BM25 keyword retrieval, merges candidates with Reciprocal Rank Fusion, and applies a CrossEncoder to complex questions.

The pilot does not use customer conversations as training data. Its initial corpus consists of approved troubleshooting guides, product notices, and redacted incident summaries. Search telemetry is retained for thirty days for quality analysis.

## Cygnus answer policy

[EVIDENCE:NS-302] When retrieved evidence does not support a claim, Cygnus must explicitly say that the available evidence is insufficient. Low-confidence answers require manual review by the support quality queue, and every document-grounded answer must include a source citation.

The assistant may summarize conflicting documents, but it must identify the conflict rather than silently choosing the most recent-looking statement. Dates printed inside an example are not authoritative unless the surrounding section marks them as an approved decision.

## Release governance

[EVIDENCE:NS-401] Elena Park owns Northstar production release approval. The next formal readiness review is 2026-04-22. A release candidate may enter the staging environment before that meeting, but production promotion requires Elena’s recorded approval and completed security evidence.

The architecture council advises on schema compatibility and recovery scope; it does not grant production authorization. Similarly, project managers may reschedule internal milestones but cannot waive release controls.

## Customer migration waves

[EVIDENCE:NS-501]
Migration is divided into employee accounts, ten pilot merchants, and regional cohorts. Pilot merchants receive parallel statement comparisons for two settlement cycles. Regional cohorts are ordered by integration complexity, not revenue. No customer is migrated during the final two business days of a financial quarter.

## Data reconciliation

[EVIDENCE:NS-502]
Settlement totals are reconciled at transaction, merchant, and regional levels. A mismatch below the operational tolerance is logged but still investigated if it persists for three days. The reconciliation owner coordinates with finance but does not approve financial adjustments.

## Training and communications

[EVIDENCE:NS-503]
Support training includes sandbox exercises, failure-handling drills, and citation review. Communications must distinguish pilot availability from general availability. Internal demonstration dates are deliberately omitted from customer notices to avoid treating demonstrations as release commitments.

## Vendor management

[EVIDENCE:NS-504]
The payment provider, identity-verification supplier, and managed observability vendor follow separate escalation paths. Contract renewals are reviewed quarterly. None of the vendor account managers owns an internal technical control, even when they help diagnose an outage.

## Program risks

[EVIDENCE:NS-505]
Current risks include certification capacity, stale analytical feeds, migration reconciliation effort, and reviewer availability. Each risk has a mitigation owner and review cadence. Risk severity is not inferred from budget size because several low-cost controls protect high-impact workflows.

## Closeout criteria

[EVIDENCE:NS-506]
Northstar closes only after production stabilization, evidence retention checks, and ownership handoff. Completion of feature development alone is insufficient. Open incidents, missing citations, or unresolved reconciliation discrepancies block closeout even if all planned releases have shipped.
