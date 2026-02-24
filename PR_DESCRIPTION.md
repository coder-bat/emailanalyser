## Summary

This PR addresses 12 critical security and performance issues identified in a comprehensive security audit.

## Security Fixes (5 Critical)

### 1. IMAP Command Injection Prevention
- Enhanced `_validate_imap_fetch_set()` to detect injection attempts
- Added regex pattern to block dangerous characters
- Validates UID ranges (e.g., `1:100`) safely

### 2. Path Traversal Protection
- Fixed `get_latest_file()` to validate patterns before use
- Added double-check: pattern validation + result path verification
- Prevents access to files outside `OUTPUT_DIR`

### 3. Job ID Injection Prevention  
- Added `validate_job_id()` to ensure only alphanumeric, hyphens, underscores
- Prevents path traversal via job_id parameter

### 4. DoS Prevention
- Added `MAX_MAX_EMAILS = 50000` upper bound
- Validates and sanitizes max_emails parameter
- Added password length limit (256 chars) with truncation warning

### 5. Input Sanitization
- Enhanced `validate_categories()` to reject newlines and control characters
- Added comprehensive `validate_path_pattern()` function

## Performance Improvements (4)

### 1. Memory Leak Fix
- Added TTL-based job cleanup (1 hour default)
- Tracks `job_access_times` for proper expiration
- Enforces `MAX_JOBS` limit when TTL cleanup insufficient

### 2. Worker Configuration
- Changed Gunicorn workers from 2 to 1 for job consistency
- Added `--max-requests 1000` for worker recycling
- Added `--max-requests-jitter 50` to prevent thundering herd

### 3. Resource Limits
- Docker: CPU limit 2 cores, memory limit 2GB
- Docker: CPU reservation 0.5 cores, memory reservation 512MB

### 4. Filesystem Security
- Docker: `read_only: true` for immutable container filesystem
- Docker: `tmpfs` for /tmp with `noexec,nosuid`
- Docker: `security_opt: no-new-privileges:true`

## Error Handling Improvements (3)

1. Added explicit error messages for validation failures
2. Added logging for rejected inputs
3. Added graceful degradation with sanitized defaults

## Testing
- All 18 existing tests pass
- No breaking changes to API

## Configuration Changes

New environment variables:
- `JOB_TTL_SECONDS` (default: 3600) - TTL for completed jobs
- `MAX_JOBS` (default: 50) - Maximum concurrent jobs to track

## Checklist
- [x] Security audit completed
- [x] All tests passing
- [x] No breaking changes
- [x] Documentation updated
- [x] Docker security hardening applied
