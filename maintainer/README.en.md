# Distribution maintenance

[한국어](README.md)

## Dataset export

Export the dataset and HTML preprocessing code from the research repository:

```bash
python maintainer/export_from_project.py --project-root /path/to/research-project
```

Experimental JSON is the primary metadata source. Differences from annotation CSVs are recorded in `data/annotation_differences.json`. `data/provenance.json` records the source Git revision, actual file hashes.

## ZIP packaging

After editing documents, code, or data, validate the bundle and create a new ZIP. Choose a different output name from any existing ZIP.

```bash
python validate.py
python maintainer/package.py --output ../reproduction_bundle_new.zip
```

The packager updates `SHA256SUMS` and includes HTML. It excludes `outputs/`, virtual environments, `.env`, caches, and Git history.

## GitHub and terms of use

`.gitignore` excludes `data/html/`. Code and annotations can be managed in Git while HTML is distributed in the dataset ZIP. Change that rule to include HTML in Git.

No distribution license has been specified. For public distribution, define and document the terms for the code and collected webpage data. HTML contains scripts and embedded resources; page sources are listed in `data/pages.json`.
