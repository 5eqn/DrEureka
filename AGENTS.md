# DrEureka Local Instructions

This DrEureka checkout has been migrated to target Go2 behavior.

Keep legacy Go1 names, package names, directory names, filenames, and import paths when they are part of the existing code layout or compatibility surface. In particular, names such as `go1_gym` may still be correct even when the active robot/task intent is Go2.

When editing prompts or user-facing task descriptions, prefer Go2 wording unless the text is quoting or documenting an unchanged compatibility name.

When running this repo inside Docker, mount the workspace root at `/workspace`, not at `/workspace/eureka-workspace`. Overlaying original Docker image contents under `/workspace` is expected; saved DrEureka/Isaac contracts may contain absolute paths such as `/workspace/thirdparties/...`.
