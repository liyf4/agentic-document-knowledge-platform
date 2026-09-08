# Northstar Reliability and Operations Manual

## Purpose and scope

This manual defines operational responsibilities for Northstar services. It contains alert thresholds, recovery objectives, escalation rules, backup requirements, and post-incident actions. Similar numerical values appear in several sections because services have different objectives; answers must retain the service name and context.

## Severity model

[EVIDENCE:OPS-100]
Severity One means a broad production outage, confirmed data corruption, or an active security compromise. Severity Two covers material degradation with a functioning workaround. Severity Three covers limited defects without material customer impact. Business priority labels in the delivery register do not replace this incident severity model.

## Payment authorization objectives

[EVIDENCE:OPS-110] The payment authorization service has a recovery time objective of 25 minutes and a recovery point objective of 2 minutes. During a Severity One event, the incident commander is Jordan Kim, and the first customer-status update is due within 15 minutes.

The settlement export service has different objectives and must not be confused with authorization. A delayed export normally affects reporting rather than real-time transaction approval.

## Settlement export objectives

[EVIDENCE:OPS-120] The settlement export service has a recovery time objective of 90 minutes and a recovery point objective of 15 minutes. Casey Wu owns the recovery procedure. Exports may be replayed only after checksum validation confirms that the previous batch was not partially delivered.

Historical planning documents mentioned a two-hour recovery target. That value is obsolete. The ninety-minute objective in this manual is the approved operating target.

## Backup schedule

[EVIDENCE:OPS-210] Configuration snapshots run every six hours and are retained for 35 days. Transaction journal backups run continuously, with a verified restore exercise every second Tuesday. The infrastructure reliability team owns restore testing, and failed exercises require a corrective action within five business days.

Application logs are retained under a different policy and are not backups. Exporting logs does not satisfy the restore-test control.

## Capacity guardrails

[EVIDENCE:OPS-220] The authorization API enters protective shedding when sustained utilization exceeds 82 percent for ten minutes. The system must preserve status and reversal endpoints, while optional reporting requests may receive HTTP 429 responses. Automatic shedding ends only after utilization remains below 70 percent for fifteen minutes.

A single utilization spike does not trigger shedding. Operators may manually enable protection during an incident, but they must record the reason in the incident timeline.

## Database failover

[EVIDENCE:OPS-230] Database failover requires two signals: replication lag above 45 seconds and a failed primary health probe in three consecutive checks. The database lead authorizes failover; the incident commander coordinates communications but does not execute the database role change.

Read-replica lag alone may be caused by analytical load and is not sufficient. Operators should first suspend nonessential reporting queries when the primary remains healthy.

## Alert ownership

[EVIDENCE:OPS-240]
Each alert has a primary and secondary rotation. The primary acknowledges pages, while the secondary prepares diagnostics and stakeholder context. Ownership transfers must be recorded at shift change. Unacknowledged pages escalate after five minutes.

## Incident communications

[EVIDENCE:OPS-310] Public incident updates are published every 30 minutes during an active Severity One event, even when there is no material change. The communications lead is Amara Singh. Technical responders provide verified facts but must not speculate about cause or recovery time.

Internal engineering updates may be more frequent. The public cadence remains thirty minutes unless legal or security leadership explicitly imposes a different schedule.

## Rollback policy

[EVIDENCE:OPS-320] A release is rolled back when error rate exceeds 3 percent for five consecutive minutes or when reconciliation identifies confirmed financial corruption. Latency alone triggers investigation at the warning threshold but does not automatically mandate rollback.

The release owner may pause deployment before either threshold is met. Pausing is preventive and should not be recorded as an automatic rollback event.

## Cache degradation

[EVIDENCE:OPS-330]
Cache loss increases database traffic and may cause stale nonfinancial views. Operators first disable expensive recommendation widgets, then increase database observation frequency. They must not flush all application caches simultaneously during peak traffic.

## Queue recovery

[EVIDENCE:OPS-340]
Backlogged notification queues are drained by customer priority and event age. Payment reversals remain ahead of marketing messages. Operators verify idempotency keys before replay and stop if duplicate delivery rises above the operational tolerance.

## Certificate rotation

[EVIDENCE:OPS-350]
Certificates are rotated at least twenty days before expiration. Emergency rotation requires validation in staging and a two-person production check. Private keys are never pasted into incident tickets, chat transcripts, or retrieved knowledge documents.

## Observability retention

[EVIDENCE:OPS-360]
High-cardinality traces are retained for seven days, aggregated service metrics for thirteen months, and incident timelines for seven years. Retention periods support different purposes and should not be substituted for one another.

## Post-incident review

[EVIDENCE:OPS-410] A Severity One review is due within five business days. It must identify contributing conditions, detection gaps, customer impact, and corrective-action owners. The review facilitator is independent of the incident commander, and action items remain open until evidence of completion is attached.

The review is blameless but not anonymous. Named ownership enables follow-through without treating individual error as the sole cause of a system failure.

## Maintenance windows

[EVIDENCE:OPS-420]
Routine database maintenance occurs Sundays from 02:00 to 04:00 UTC. Customer-visible work requires advance notice. Emergency maintenance may occur outside the window when delaying work creates greater operational risk.

## Manual operation

[EVIDENCE:OPS-430]
Manual settlement steps require a two-person check and a reconciled input file. Temporary spreadsheets must be stored in the controlled operations area and removed after the official record is produced.

## Service retirement

[EVIDENCE:OPS-440]
A retiring service remains monitored until traffic is zero, dependent jobs are migrated, and rollback is no longer required. Removing a dashboard before those conditions are met creates an avoidable observability gap.
