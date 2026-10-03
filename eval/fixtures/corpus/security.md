# Security and Compliance

## Encryption

All data is encrypted at rest with AES-256. Data in transit is protected with TLS 1.3; TLS
1.2 is accepted for older clients until 31 December 2026. Customers on the Business and
Enterprise plans can bring their own keys (BYOK) through AWS KMS or Azure Key Vault.

## Certifications

Nimbus holds SOC 2 Type II and ISO 27001 certifications. The SOC 2 report is renewed every
twelve months and can be requested under NDA from the trust portal. Nimbus is not HIPAA
certified, but Enterprise customers can sign a Business Associate Agreement (BAA).

## Data residency

Customers choose a home region when creating a project. Available regions are us-east,
us-west, eu-central and ap-southeast. Data never leaves the home region unless cross-region
replication is turned on explicitly.

## Access control

Single sign-on with SAML 2.0 and OIDC is available on the Business and Enterprise plans.
Multi-factor authentication is mandatory for all accounts with administrator rights.
Audit logs are kept for 400 days on Enterprise and 90 days on other paid plans.
