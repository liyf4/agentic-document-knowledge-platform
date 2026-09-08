# Northstar Delivery Work Items

## Register scope

This document converts the active delivery register into narrative Markdown sections. Each work item has one accountable owner, one current state, and one dependency. Historical proposals in other documents do not override this active register.

## Atlas token contract tests

[EVIDENCE:DEL-001] Work item AT-401 is owned by Priya Nair. It requires completion of the token contract test suite by 2026-03-03. Its current status is in progress, priority is high, and it depends on token schema version 3.2.

## Atlas certification data

[EVIDENCE:DEL-002] Work item AT-402 is owned by Jon Bell. The certification test-data refresh was completed on 2026-02-28. It had medium priority and depended on payment-provider access.

## Atlas rollback documentation

[EVIDENCE:DEL-003] Work item AT-403 is owned by Lena Ortiz. Documenting gateway rollback steps is due 2026-03-05, remains to do, has medium priority, and awaits an operations review.

## Atlas certification environment

[EVIDENCE:DEL-004] Work item AT-404 is owned by Chen Wei. Stabilizing the shared certification environment is blocked, has high priority, is due 2026-03-01, and depends on a payment-provider firewall change.

## Atlas release decision

[EVIDENCE:DEL-005] Work item AT-405 is owned by Elena Park. Recording the production approval decision is due 2026-04-22, remains to do, has high priority, and depends on complete readiness evidence.

## Beacon freshness review

[EVIDENCE:DEL-006] Work item BC-501 is owned by Miguel Santos. Creating the daily source freshness review was completed on 2026-04-01. It had high priority and depended on Ledger and Fulfillment event tables.

## Beacon escalation playbook

[EVIDENCE:DEL-007] Work item BC-502 is owned by Morgan Lee. Publishing the freshness escalation playbook is in progress, due 2026-04-04, medium priority, and depends on the signed exception-note template.

## Beacon segmentation query

[EVIDENCE:DEL-008] Work item BC-503 is owned by Riley Shaw. Rebuilding the regional segmentation query is in progress, due 2026-04-09, medium priority, and depends on conformed dimensions.

## Beacon ownership map

[EVIDENCE:DEL-009] Work item BC-504 is owned by Taylor Reed. Documenting the dashboard ownership map is due 2026-04-11, remains to do, has low priority, and depends on the support roster.

## Beacon launch communication

[EVIDENCE:DEL-010] Work item BC-505 is owned by Amara Singh. Preparing the launch communication cadence is due 2026-04-14, remains to do, has medium priority, and depends on status-page review.

## Cygnus dense benchmark

[EVIDENCE:DEL-011] Work item CY-601 is owned by Nora Blake. Validation of the dense retrieval benchmark was completed on 2026-01-20 with high priority against the approved corpus.

## Cygnus BM25 benchmark

[EVIDENCE:DEL-012] Work item CY-602 is owned by Samir Gupta. Validation of the BM25 keyword benchmark was completed on 2026-01-21 with high priority against the approved corpus.

## Cygnus RRF trace

[EVIDENCE:DEL-013] Work item CY-603 is owned by Uma Roy. Implementing the Reciprocal Rank Fusion trace was completed on 2026-01-23 with high priority and depended on both dense and BM25 outputs.

## Cygnus reranker telemetry

[EVIDENCE:DEL-014] Work item CY-604 is owned by Grace Liu. Adding CrossEncoder latency telemetry is in progress, due 2026-01-27, medium priority, and depends on the reranker runtime.

## Cygnus refusal tests

[EVIDENCE:DEL-015] Work item CY-605 is owned by Owen Price. Writing insufficient-evidence refusal tests was completed on 2026-01-28 with high priority and depends on the approved answer policy.

## Cygnus citation regression

[EVIDENCE:DEL-016] Work item CY-606 is owned by Fatima Noor. Creating the citation display regression suite is in progress, due 2026-02-02, medium priority, and depends on the frontend citation schema.

## Cygnus multilingual sample

[EVIDENCE:DEL-017] Work item CY-607 is owned by Diego Cruz. Preparing the multilingual query sample remains to do, is due 2026-02-06, has low priority, and depends on translation review.

## Cygnus isolation fixtures

[EVIDENCE:DEL-018] Work item CY-608 is owned by Hana Mori. Auditing session-isolation fixtures is blocked, due 2026-02-07, high priority, and depends on test-tenant provisioning.

## Authorization recovery exercise

[EVIDENCE:DEL-019] Work item OP-701 is owned by Jordan Kim. Running the authorization recovery exercise remains to do, is due 2026-03-10, high priority, and depends on the incident simulation.

## Settlement checksum verification

[EVIDENCE:DEL-020] Work item OP-702 is owned by Casey Wu. Verifying the settlement replay checksum is in progress, due 2026-03-12, high priority, and depends on an export sample batch.

## Incident templates

[EVIDENCE:DEL-021] Work item OP-703 is owned by Amara Singh. Updating public incident templates was completed on 2026-03-15 with medium priority after legal review.

## Protective shedding test

[EVIDENCE:DEL-022] Work item OP-704 is owned by Jordan Kim. Testing protective traffic shedding remains to do, is due 2026-03-18, high priority, and depends on the load environment.

## Restore exercise

[EVIDENCE:DEL-023] Work item OP-705 is owned by Casey Wu. Completing the configuration restore exercise remains to do, is due 2026-03-20, high priority, and depends on a valid backup snapshot.

## Privileged access review

[EVIDENCE:DEL-024] Work item GV-801 is owned by Asha Patel. Completing the monthly privileged-access review is in progress, due 2026-03-31, high priority, and depends on the identity export.

## Emergency access evidence

[EVIDENCE:DEL-025] Work item GV-802 is owned by Leo Martin. Reviewing emergency-access evidence was completed on 2026-03-07 with high priority using incident records.

## Evaluation retention validation

[EVIDENCE:DEL-026] Work item GV-803 is owned by Asha Patel. Validating benchmark-report retention remains to do, is due 2026-03-09, medium priority, and depends on the evaluation archive.

## Exception expiration

[EVIDENCE:DEL-027] Work item GV-804 is owned by Leo Martin. Expiring outdated control exceptions is in progress, due 2026-03-11, high priority, and depends on approved risk records.

## Supplier activity inspection

[EVIDENCE:DEL-028] Work item GV-805 is owned by Maya Chen. Inspecting supplier-account activity remains to do, is due 2026-03-14, medium priority, and depends on maintenance logs.

## Citation workshop

[EVIDENCE:DEL-029] Work item TR-901 is owned by Kai Evans. Delivering the support citation workshop remains to do, is due 2026-03-22, medium priority, and depends on approved Cygnus examples.

## Communication drill

[EVIDENCE:DEL-030] Work item TR-902 is owned by Inez Silva. Running the incident communication drill remains to do, is due 2026-03-24, medium priority, and depends on current status templates.
