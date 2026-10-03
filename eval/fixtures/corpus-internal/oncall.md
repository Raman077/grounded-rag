# Internal On-call Runbook

Restricted to the engineering group.

## Paging

The primary on-call engineer is paged through PagerDuty for any Sev-1 incident. The
escalation contact after 15 minutes without acknowledgement is the duty manager.

## Database failover

To fail over the metadata database, run `nimbusctl db failover --region <region>` from a
bastion host. Failover takes about 90 seconds.
