# API Rate Limits

The Nimbus REST API limits how many requests a project can make. Limits apply per project,
not per API key.

## Request limits by plan

- Starter: 10 requests per second, burst up to 20.
- Pro: 100 requests per second, burst up to 200.
- Business: 1,000 requests per second, burst up to 2,000.
- Enterprise: negotiated per contract.

## When a limit is exceeded

The API responds with HTTP 429 Too Many Requests and a `Retry-After` header giving the
number of seconds to wait. Clients should back off exponentially starting from that value.

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 2
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
```

## Upload size

A single upload request can carry at most 5 GB. Larger objects must use multipart upload,
which supports objects up to 5 TB in parts of 5 MB to 5 GB.
