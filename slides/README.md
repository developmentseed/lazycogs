# lazycogs presentations

This is the Slidev project for lazycogs talks. It uses the DevelopmentSeed theme from [ds-slidev-template](https://github.com/developmentseed/ds-slidev-template). The theme source and its README are retained under `theme/`; the template repository at the reference revision did not include a license file.

## Edit and preview

Install the pinned dependencies and start the FOSS4G North America 2026 presentation locally from this directory:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

Edit `foss4gna-2026/slides.md` to change the deck. The FOSS4GNA deck uses talk-local components under `foss4gna-2026/components/` and verified static assets under `foss4gna-2026/assets/`. HTML Xarray representations are served as local imported assets in sandboxed iframes so their original HTML and styling remain intact offline. The closing-slide QR is a committed local SVG; it does not call a remote QR service at presentation time. Southwest dates, datatake labels, and source-overview levels come from the committed manifest; the slides distinguish the 300 m target-grid array from separate 1,200 m display previews. The working talk structure is in [`foss4gna-2026/outline.md`](foss4gna-2026/outline.md); the submitted proposal and talk-specific guidance are in [`foss4gna-2026/AGENTS.md`](foss4gna-2026/AGENTS.md).

The talk runtime has no remote font dependency: the shared theme uses its bundled Antarctica font with system sans-serif and monospace fallbacks, and Slidev webfonts are disabled for this deck. Use Slidev's presenter view for speaker notes; the production build emits the website presentation. PDF export is optional and requires `playwright-chromium` plus its browser, which are not part of the pinned dependencies. Once available, run `pnpm exec slidev export foss4gna-2026/slides.md --format pdf --output .cache/foss4gna-2026.pdf` from `slides/`. Check exported slides for reveal-dependent content: the Southwest source-pass sequence has a print fallback with the three flat mosaics.

## Build for the website

The production site is hosted under `/lazycogs/`. Build the deck with its full presentation base path and output directly into the MkDocs site directory:

```sh
pnpm build
```

The build script writes to `../site/presentations/foss4gna-2026` with the base `/lazycogs/presentations/foss4gna-2026/`. The GitHub Actions docs workflow builds MkDocs first, builds Slidev into that output, then publishes the combined site once with `mkdocs gh-deploy --dirty --force`. MkDocs 1.6.1 does not provide a `gh-deploy --no-build` option; `--dirty` is used so its deployment build does not clean the Slidev output before publication. Do not use Slidev's standalone deployment configuration for the shared Pages site.

The deck uses Slidev's native `publicDir` Vite setting to share assets from `slides/public/` while keeping the presentation source in its own directory. The theme and package dependencies remain shared under `slides/`.
