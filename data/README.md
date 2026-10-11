---
license: cc-by-sa-4.0
pretty_name: CS 489 F26 A1 Wikipedia Corpus
language:
  - fi
  - hu
  - de
  - ru
  - ar
  - zh
  - vi
task_categories:
  - text-generation
tags:
  - wikipedia
  - cs489
---

# CS 489 A1 data

Seven languages from Wikimedia dumps. `raw` contains uncleaned dump fragments
(wiki markup, templates, HTML entities, reference markers, tables, and
inter-language link clutter are left in on purpose: cleaning them is Part 1).
`clean_reference/{wiki}/dev.txt` contains the first 50 physical lines of the
standard-solution-cleaned dev corpus for each language. These excerpts show the
expected degree of cleaning but are not fallback corpora for model training or
evaluation.

Blank lines separate paragraphs as well as documents in the raw `.txt` files,
so document boundaries cannot be recovered reliably from the released text.
The reference cleaner reads the same raw train and dev files released to students.
The `clean chars` columns below record collection-time targets from an internal
baseline cleaner; they are not the standard solution's cleaned byte counts.

| wiki | language | train docs | train clean chars | dev docs | dev clean chars |
|------|----------|-----------:|------------------:|---------:|----------------:|
| fi | Finnish | 1,053 | 10,001,453 | 100 | 1,004,034 |
| hu | Hungarian | 1,447 | 10,002,446 | 150 | 1,004,101 |
| de | German | 890 | 10,013,493 | 92 | 1,011,170 |
| ru | Russian | 981 | 10,003,182 | 95 | 1,003,274 |
| ar | Arabic | 1,179 | 10,004,816 | 128 | 1,003,006 |
| zh | Mandarin Chinese | 1,517 | 10,000,299 | 171 | 1,009,408 |
| vi | Vietnamese | 1,444 | 10,000,125 | 149 | 1,019,470 |

The full test inputs are held by the TA. See the assignment handout for the
submission paths and artifact-based grading procedure.
