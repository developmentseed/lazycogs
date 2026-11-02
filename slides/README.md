# lazycogs presentations

This is the Slidev project for lazycogs talks. It uses the DevelopmentSeed theme from [ds-slidev-template](https://github.com/developmentseed/ds-slidev-template). The theme source and its README are retained under `theme/`; the template repository at the reference revision did not include a license file.

## Edit and preview

Install the pinned dependencies and start the local Slidev server:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

Edit `slides.md` to change the deck. The starter content is intentionally short; replace the future-talk placeholder when preparing a specific talk.

## Build for the website

The production site is hosted under `/lazycogs/`. Build the deck with its full presentation base path and output directly into the MkDocs site directory:

```sh
pnpm exec slidev build --base /lazycogs/presentations/lazycogs/ --out ../site/presentations/lazycogs
```

The GitHub Actions docs workflow builds MkDocs first, builds Slidev into that output, then publishes the combined site once with `mkdocs gh-deploy --dirty --force`. MkDocs 1.6.1 does not provide a `gh-deploy --no-build` option; `--dirty` is used so its deployment build does not clean the Slidev output before publication. Do not use Slidev's standalone deployment configuration for the shared Pages site.
