# Coursework submission ZIP and demonstration

The complete, previously verified submission ZIP is committed as 40 numbered binary parts because its 327,903,892 bytes exceed the normal GitHub file limit. The parts reconstruct the original ZIP byte for byte.

From the repository root, run:

```bash
python3 tools/restore_submission_zip.py
```

This creates `submission/AI-course-work-submission.zip`, verifies every part, checks the full ZIP SHA-256, checks all 861 archive members and validates their CRCs. It needs only the Python standard library. To verify an existing restored ZIP, add `--verify-only`; use `--output /your/path/AI-course-work-submission.zip` to select another destination. A different existing output is never overwritten.

Original ZIP SHA-256: `ec820860787851c2381bd9106ed03ea259f54c9334cf8e575309d5d380d90fcd`.

The [demonstration video](demonstration.mp4) is committed directly alongside these archive parts at `submission/demonstration.mp4`. The identical original remains at `reports/demonstration.mp4` and is also included inside the submission ZIP. Its duration is 130.04 seconds.

The archive is the immutable completed submission from commit `049a9c9e0c9928e422cdb7c4852e11722c4b6594`. All 860 indexed file hashes match the submitted evidence. Its coursework protocol disclosure and individual presentation requirements remain documented inside it. The archive-part restore command is a repository delivery utility added after that snapshot.
