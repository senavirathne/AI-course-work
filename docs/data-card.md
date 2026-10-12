# RAW-FABRID V1 data card

The complete publisher archive was acquired on 11 October 2026. It contains 709 original 1792x1024 grayscale images, 204 original masks and 19,852 grayscale 256x256 patches. The archive is 4,133,046,386 bytes; its verified SHA-256 is `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`. Source: https://data.mendeley.com/datasets/db6g85xsyg/1, DOI 10.17632/db6g85xsyg.1. Attribution: Ruben Perez-Llorens, Teresa Albero-Albero and Javier Silvestre-Blanes (2026). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.

The publisher describes an industrial Basler line-scan system with dual LED lighting and 4 pixels/mm. Its original border crop precedes the released images. We use the released patches, preserve their resolution and apply only documented classifier normalisation and training flips. Phone photographs and production deployment are outside the verified domain.

| Verified property | Observation / treatment |
| --- | --- |
| Original publisher partitions | train/good: 14,196; test/good: 4,969; test/defect: 687 |
| Classes | 19,165 explicitly normal patches; 687 defective patches |
| Patch CSV fields | `patch_name,parent_raw,type,defect_area` |
| Source CSV fields | `filename,roll_id,has_defect,total_area_px,num_defects` |
| Source mapping | Every patch joins to one of 709 originals through `parent_raw` |
| Origin grouping | Seven explicit `roll_id` values, joined through publisher source metadata |
| Mask paths | Every positive patch has a corresponding mask with the same filename |
| Mask quality issue | 471 of 687 patch masks have intermediate grey edge values |
| Foreground rule | Exactly-white pixels (255) match CSV `defect_area` for all 687 masks; other grey values are excluded. Raw masks are preserved |
| Target | At least one exact-white annotated pixel gives defective=1. Normality is established from metadata and documented folders, never from a missing mask alone |
| Parent labels | A patch from a defective parent can still be normal; parent-level labels are not copied to patches |
| Exact duplicate audit | No identical patch pixel arrays |
| Near duplicate audit | 324,109 screened candidate pairs, 7,587 full pixel comparisons, no confirmed additional matches at mean absolute distance <=3/255 |
| Duplicate limitations | The screen does not establish absence of rotations, different crops or undocumented relationships |
| Classifier input | Pixel content only; labels, masks, source IDs, filenames and origin IDs never enter the classifier |
| Genuine engineering sample | 168 original patches and 84 corresponding masks, all seven origin groups; defects intentionally oversampled for testing, not assessed metrics |

The field name `roll_id` suggests rolls, while the publisher README describes an anonymised fabric-origin ID. We conservatively keep every such group intact. We do not establish whether the seven groups correspond one-to-one to physical rolls, weave types, manufacturers or capture sessions. The evaluation measures generalisation to held-out publisher origin groups, not certified generalisation to arbitrary new factories or fabrics.

The approved supervised research split combines all labelled patches. The publisher's normal-only training folder is an anomaly-detection convention and cannot supply positive supervised examples. This research split is a different protocol from that benchmark.

| Partition | Publisher origin IDs | Patches | Defective | Source images | Group count |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | Rollo2A, Rollo4A, Rollo5A, Rollo7A | 13,384 | 539 | 478 | 4 |
| Validation | Rollo6A | 3,472 | 52 | 124 | 1 |
| Test | Rollo1A, Rollo3A | 2,996 | 96 | 107 | 2 |

The seed is 42. Of 128 integrity-only assignment draws, draw 32 provided feasible class coverage and ratios: 67.42/17.49/15.09 percent. No fitted model influenced the assignment. The immutable split SHA-256 is `af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f`. Only four training groups, one validation group and two test groups substantially limit inference: thousands of related patches are not thousands of independent manufacturing observations.

Training-only EDA reports 4.03 percent defective patches, a minimum defect area of one pixel and median positive defect area of 4.72 percent. These facts constrain imbalance treatment, patch resolution, capacity and explanation interpretation. See `eda-statistics.json` and `eda-modelling-findings.json` for measured evidence.

Historical synthetic examples remain under `development-run/` as engineering history. They are not RAW-FABRID and do not contribute to coursework metrics.
