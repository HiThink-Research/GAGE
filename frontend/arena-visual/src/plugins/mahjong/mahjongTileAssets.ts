const mahjongTileAssetModules = import.meta.glob("./assets/tiles/*.svg", {
  eager: true,
  import: "default",
}) as Record<string, string>;

export const mahjongTileAssets = Object.fromEntries(
  Object.entries(mahjongTileAssetModules).map(([path, url]) => {
    const fileName = path.split("/").pop() ?? path;
    return [fileName.replace(".svg", ""), url];
  }),
);

export function resolveMahjongTileAsset(tileCode: string): string | null {
  return mahjongTileAssets[tileCode] ?? null;
}
