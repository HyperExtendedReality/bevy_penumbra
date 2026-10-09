# Local Sponza assets

These files were supplied locally under `assets/` on October 6, 2026. No download
or redistribution source/license was supplied, so this inventory does not establish
redistribution permission. The glTF metadata names Babylon's 3ds Max exporter.
The demo reads the source files without rewriting them.

| Scene input | Nodes | Mesh primitives | Source triangles | Referenced images |
| --- | ---: | ---: | ---: | ---: |
| `NewSponza_Main_glTF_003.gltf` | 155 | 405 | 3,747,018 | 72 |
| `NewSponza_Curtains_glTF.gltf` | 17 | 43 | 1,997,366 | 8 |
| `NewSponza_IvyGrowth_glTF.gltf` | 2 | 2 | 5,065,558 | 3 |

The referenced binary and PNG files are present in the supplied folder. Each
referenced PNG is 4096×4096. Scene-node transforms are retained; the ivy root nodes
carry a scale of 100. The main scene has one `BLEND` dirt-decal material; other
source materials are opaque. The ivy's `IvyLeaf_BaseColor.png` is RGB and its glTF
material omits `alphaMode`, which means opaque. The leaves use geometric outlines.
The example's separate masked leaf cards use a generated RGBA image with explicit
zero/one alpha regions.

The current example applies a 512-pixel material texture limit and loads one scene
at a time. This keeps the demo's asset use bounded; it is not a source-resolution
texture residency result. Measurements from earlier renderer versions are not
included here because they do not describe the current Split Radiance Cascades
implementation.

SHA-256 of the supplied scene and mesh files:

| File | SHA-256 |
| --- | --- |
| `NewSponza_Main_glTF_003.gltf` | `e04c4c540c74bdddcbd3f590a85c14119bfd5839702246a23b3a737c0cce2400` |
| `NewSponza_Main_glTF_003.bin` | `903d443b46b51c8397c25b65286d9f9e111121f12a3e0e303f19019982568d6b` |
| `NewSponza_Curtains_glTF.gltf` | `b56968aeb691970b63027ae3ffbb0d77f1d736e7bdace5c70be106d6dcfbb516` |
| `NewSponza_Curtains_glTF.bin` | `c89c2876636a0baa9a506e3206d8336239d9a6b79e73c9e1dd53cd47f2caca0c` |
| `NewSponza_IvyGrowth_glTF.gltf` | `22039080d6de24569f377210b8ebbe59194b5d44de777679d9d87b69ffda2c1e` |
| `NewSponza_IvyGrowth_glTF.bin` | `4e2d5ac1b3d63f269e2ea030860bc40f222a6e81b86aef0055a0cb37446268ad` |