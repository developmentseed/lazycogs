# lazycogs presentations

This is the Slidev project for lazycogs talks. It uses the DevelopmentSeed theme from [ds-slidev-template](https://github.com/developmentseed/ds-slidev-template). The theme source and its README are retained under `theme/`; the template repository at the reference revision did not include a license file. The project maintainer confirmed permission to publish the bundled theme and its assets in this deck; this does not assert a broader upstream license.

## Edit and preview

Install the pinned dependencies and start the FOSS4G North America 2026 presentation locally from this directory:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

Edit `foss4gna-2026/slides.md` to change the deck. The sixteen-slide FOSS4GNA talk uses short highlighted Python snippets, actual Xarray HTML representations, and prepared notebook results. The Southwest example uses the March–June monthly array and the notebook's saved May image, extracted under `foss4gna-2026/assets/southwest-notebook/`; it does not use the older seasonal preview assets. HLS results and provenance are under `foss4gna-2026/assets/hls/`. The stac-geoparquet introduction uses two diagrams from Pete Gadomski's presentation, with visible source links and provenance under `foss4gna-2026/assets/stac-geoparquet/`. An image-only slide after “What if we also defer STAC searches?” plays the [Tenor GIF](https://tenor.com/vZ9l.gif), bundled locally as `foss4gna-2026/assets/deck/duper-office.gif`. The library overview uses three panels over the DS template's Tanezrouft Basin image, bundled under `public/images/theme/`.

The small components under `foss4gna-2026/components/` embed local images and sandboxed Xarray HTML documents. The logo-only introduction and closing slides periodically switch googly eyes on and off instantly, with pupils looking right. Reduced-motion and print views retain the resting expression. `assets/deck/xarray.css` adapts the Xarray-specific rules from `../docs/stylesheets/extra.css` and is injected inside each iframe so color variables, grid/list layouts, and attribute styling work offline. The source HTML representations are retained unchanged. The closing QR code uses the theme's `CurrentUrlQRCode` component with an explicit destination URL and the `slidev-addon-qrcode` addon. It renders locally with the bundled DS symbol; no remote QR service is called. The working structure, pacing, code/output crosswalk, and caveats are in [`foss4gna-2026/outline.md`](foss4gna-2026/outline.md); the submitted proposal is preserved in [`foss4gna-2026/AGENTS.md`](foss4gna-2026/AGENTS.md).

The talk runtime has no remote font dependency: the shared theme uses its bundled Antarctica font with system sans-serif and monospace fallbacks, and Slidev webfonts are disabled for this deck. Use Slidev's presenter view for speaker notes; the production build emits the website presentation. PDF export is optional and requires `playwright-chromium` plus its browser, which are not part of the pinned dependencies. Once available, run `pnpm exec slidev export foss4gna-2026/slides.md --format pdf --output .cache/foss4gna-2026.pdf` from `slides/`. Check exported slides for reveal-dependent content: slide 15 has one native click that replaces the monthly open/repr with the May selection and image. The closing-logo animation has a static print fallback; there are no source-pass diagrams.

## Build for the website

The production site is hosted under `/lazycogs/`. Build the deck with its full presentation base path and output directly into the MkDocs site directory:

```sh
pnpm build
```

The build script writes to `../site/presentations/foss4gna-2026` with the base `/lazycogs/presentations/foss4gna-2026/`. The deck uses Slidev's hash routing, so links such as `/lazycogs/presentations/foss4gna-2026/#/14` survive refreshes without server-side SPA rewrites.

Pull requests to `main` validate the combined build without publishing. Pushes to `main`, or manual workflow runs on `main`, build MkDocs first, build Slidev into that output, then publish the combined site once with `mkdocs gh-deploy --dirty --force`. Main-branch runs are serialized to prevent overlapping or out-of-order deployments. MkDocs 1.6.1 does not provide a `gh-deploy --no-build` option; `--dirty` is used so its deployment build does not clean the Slidev output before publication. Do not use Slidev's standalone deployment configuration for the shared Pages site.

The deck's favicon is a copy of the MkDocs site's `docs/favicon.svg`, stored in `public/favicon.svg`.

The deck uses Slidev's native `publicDir` Vite setting to share assets from `slides/public/` while keeping the presentation source in its own directory. The theme and package dependencies remain shared under `slides/`.
