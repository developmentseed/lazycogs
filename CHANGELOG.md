# Changelog

## [0.7.0](https://github.com/developmentseed/lazycogs/compare/v0.6.2...v0.7.0) (2026-07-01)


### Features

* apply max_concurrent_reads across time steps, add option to raise storage errors ([#81](https://github.com/developmentseed/lazycogs/issues/81)) ([66c6309](https://github.com/developmentseed/lazycogs/commit/66c6309c358ef4e53c02a34bafd112c504bcddf6))

## [0.6.2](https://github.com/developmentseed/lazycogs/compare/v0.6.1...v0.6.2) (2026-06-30)


### Bug Fixes

* match explain() time steps by coordinate value not indexer key ([#82](https://github.com/developmentseed/lazycogs/issues/82)) ([2ade771](https://github.com/developmentseed/lazycogs/commit/2ade771da2accdb4d807616537cd3f9c1580937e))

## [0.6.1](https://github.com/developmentseed/lazycogs/compare/v0.6.0...v0.6.1) (2026-06-26)


### Bug Fixes

* require rustac&gt;=0.9.13 for correct handling of start/end_datetimes ([#77](https://github.com/developmentseed/lazycogs/issues/77)) ([f4f574c](https://github.com/developmentseed/lazycogs/commit/f4f574c1261ab0b3f2d3d15f6d019edc71911d1a)), closes [#76](https://github.com/developmentseed/lazycogs/issues/76)

## [0.6.0](https://github.com/developmentseed/lazycogs/compare/v0.5.0...v0.6.0) (2026-06-23)


### Features

* add options for hourly or exact timestamp temporal grouping ([#74](https://github.com/developmentseed/lazycogs/issues/74)) ([a742078](https://github.com/developmentseed/lazycogs/commit/a742078074efdae9e2baaccccec52ab209970800))

## [0.5.0](https://github.com/developmentseed/lazycogs/compare/v0.4.1...v0.5.0) (2026-06-01)


### Features

* lazycogs interoperable with rioxarray ([#71](https://github.com/developmentseed/lazycogs/issues/71)) ([58fae42](https://github.com/developmentseed/lazycogs/commit/58fae4236d69eb0145d06dcf03a5089950c76ec5)), closes [#70](https://github.com/developmentseed/lazycogs/issues/70)

## [0.4.1](https://github.com/developmentseed/lazycogs/compare/v0.4.0...v0.4.1) (2026-06-01)


### Bug Fixes

* add _WindowContext class for use in explain operations ([#69](https://github.com/developmentseed/lazycogs/issues/69)) ([a54b640](https://github.com/developmentseed/lazycogs/commit/a54b6404a328abd130b7cb92964c02ee5ad4d0bc)), closes [#68](https://github.com/developmentseed/lazycogs/issues/68)
* keep lazy runtime state out of DataArray attrs ([#66](https://github.com/developmentseed/lazycogs/issues/66)) ([fc4b3d0](https://github.com/developmentseed/lazycogs/commit/fc4b3d09f8b67f515b2a4ab6088bcd6e28dade5d)), closes [#65](https://github.com/developmentseed/lazycogs/issues/65) [#64](https://github.com/developmentseed/lazycogs/issues/64)

## [0.4.0](https://github.com/developmentseed/lazycogs/compare/v0.3.1...v0.4.0) (2026-05-26)


### Features

* improve nodata and dtype handling ([#63](https://github.com/developmentseed/lazycogs/issues/63)) ([4dbd618](https://github.com/developmentseed/lazycogs/commit/4dbd618bf1d0e2af9ec8ef36b81639831fc08fb5))


### Bug Fixes

* replace ObjectStore type requirement with async_geotiff.Store ([#60](https://github.com/developmentseed/lazycogs/issues/60)) ([ea2837a](https://github.com/developmentseed/lazycogs/commit/ea2837a8d7260a6c3890b4723a83c1230bde8a3e))

## [0.3.1](https://github.com/developmentseed/lazycogs/compare/v0.3.0...v0.3.1) (2026-05-15)


### Bug Fixes

* eagerly load spatial coords to avoid mismatch in sel operations ([#57](https://github.com/developmentseed/lazycogs/issues/57)) ([1f66381](https://github.com/developmentseed/lazycogs/commit/1f66381582b13a73084fbe24f35709aecfc8ee3e))

## [0.3.0](https://github.com/developmentseed/lazycogs/compare/v0.2.0...v0.3.0) (2026-05-07)


### Features

* implement xarray async capability ([#46](https://github.com/developmentseed/lazycogs/issues/46)) ([e7a501c](https://github.com/developmentseed/lazycogs/commit/e7a501caeecee7f9cc2c0fb104fd7af2a03172ad))
* use rasterix to handle x/y dimensions ([#52](https://github.com/developmentseed/lazycogs/issues/52)) ([0f02cc2](https://github.com/developmentseed/lazycogs/commit/0f02cc24f43bc3c9028d9ce33c1f595ce4c9de94))

## [0.2.0](https://github.com/developmentseed/lazycogs/compare/v0.1.2...v0.2.0) (2026-05-04)


### Features

* add run_chunk_async and run_chunk for direct non-xarray access pattern ([#45](https://github.com/developmentseed/lazycogs/issues/45)) ([31e1952](https://github.com/developmentseed/lazycogs/commit/31e195206f2bb8490766373694afd8ee9e8cc7b6))


### Bug Fixes

* remove unnecessary lock on duckdb searches ([#39](https://github.com/developmentseed/lazycogs/issues/39)) ([dede83c](https://github.com/developmentseed/lazycogs/commit/dede83cff9594576e04e47a08e943ce9a3527dab))

## [0.1.2](https://github.com/developmentseed/lazycogs/compare/v0.1.1...v0.1.2) (2026-04-29)


### Bug Fixes

* rip out fake-async open_async ([#37](https://github.com/developmentseed/lazycogs/issues/37)) ([cf07318](https://github.com/developmentseed/lazycogs/commit/cf07318c0c2414bf47251de72fde9c2189263d31)), closes [#26](https://github.com/developmentseed/lazycogs/issues/26)

## [0.1.1](https://github.com/developmentseed/lazycogs/compare/v0.1.0...v0.1.1) (2026-04-27)


### Documentation

* update installation instructions ([48c0a82](https://github.com/developmentseed/lazycogs/commit/48c0a826aa38f1ab90154a0951176bbdf9578d48))
* update links in README ([d522ba6](https://github.com/developmentseed/lazycogs/commit/d522ba6ab70dcb588318868261f025719c3b487a))

## [0.1.0](https://github.com/developmentseed/lazycogs/compare/v0.0.1...v0.1.0) (2026-04-27)


### Features

* initial release ([4b5241d](https://github.com/developmentseed/lazycogs/commit/4b5241d24653a28791f07759a4660d7582f2330c))
