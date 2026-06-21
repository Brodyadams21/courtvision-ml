# CourtVision Cloud Training Guide

This document tracks the AWS setup for Phase 9 cloud-assisted training.

## Safety guardrails

- AWS profile: `courtvision`
- Region: `us-east-1`
- Monthly budget alert created before running SageMaker
- No SageMaker endpoints in Phase 9
- No notebook instances in Phase 9
- No always-on EC2 instances
- Training jobs must use a max runtime limit
- Temporary cloud artifacts should be cleaned up after verification

## S3 bucket

The real bucket name should not be hardcoded in public repo files. Set it locally:

```powershell
$AccountId = aws sts get-caller-identity --query Account --output text
$env:COURTVISION_S3_BUCKET = "courtvision-ml-brody-$AccountId-us-east-1"
```

The bucket is configured with:

- Public access blocked
- Default AES256 server-side encryption
- Lifecycle expiration for temporary/ objects after 7 days
