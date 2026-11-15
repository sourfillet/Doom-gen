"""Automated extraction and bundling of missing textures and flats across source WADs.

Allows generated maps to reference textures and flats from across the Doom corpus
(e.g. Doom 1, Plutonia, TNT, or PWADs) while targeting another base IWAD (e.g. Doom 2).
Missing assets are detected from the MapDocument, extracted on-demand from the source
WAD pool, registered on the GameProfile to prevent STARTAN2/fallback substitution,
and packaged into an AssetBundle for embedding or companion PWAD output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
from copy import deepcopy
import tempfile

from generation.game_profile import GameProfile
from wad_parser.texture_parser import parse_pnames, parse_texture_lump
from wad_parser.types import MapDocument, Texture
from wad_parser.wad_reader import WadFile
from wad_writer.assets import AssetBundle, EmbeddedTexture
from wad_writer.wad_writer import write_lumps


class AssetExtractor:
    """Pool of source WADs from which textures and flats can be queried and extracted."""

    def __init__(self, sources: Iterable[Path | str | WadFile] | None = None) -> None:
        self.sources: list[WadFile] = []
        self._flats: dict[str, tuple[WadFile, bytes]] = {}
        self._textures: dict[str, tuple[WadFile, Texture]] = {}

        if sources:
            for s in sources:
                self.add_source(s)

    @classmethod
    def from_directory(cls, dir_path: Path | str, pattern: str = "*.[wW][aA][dD]") -> "AssetExtractor":
        """Instantiate an extractor indexing all WAD files in the given directory."""
        extractor = cls()
        root = Path(dir_path)
        if root.is_dir():
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    try:
                        extractor.add_source(path)
                    except (ValueError, OSError):
                        continue
        return extractor

    def add_source(self, source: Path | str | WadFile) -> None:
        """Index a source WAD file, directory, or WadFile instance."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                for p in sorted(path.glob("*.[wW][aA][dD]")):
                    if p.is_file():
                        self.add_source(p)
                return
            wad = WadFile(str(path))
        else:
            wad = source

        # Validate and stage a whole source before changing the catalog. The
        # general texture parser is permissive for diagnostics; extraction
        # must not turn truncated directories/patch lists into usable assets.
        pnames_lump = wad.get_lump("PNAMES")
        tex1_lump = wad.get_lump("TEXTURE1")
        tex2_lump = wad.get_lump("TEXTURE2")
        textures = {}
        if pnames_lump is not None and (tex1_lump is not None or tex2_lump is not None):
            tables = {'PNAMES': pnames_lump,
                      'TEXTURE1': tex1_lump if tex1_lump is not None else bytes(4)}
            if tex2_lump is not None:
                tables['TEXTURE2'] = tex2_lump
            try:
                AssetBundle(base_tables=tables).validate()
            except ValueError as exc:
                raise ValueError(f'{wad.path}: invalid source texture tables: {exc}') from exc
            pnames = parse_pnames(pnames_lump)
            for t_lump in (tex1_lump, tex2_lump):
                if t_lump is not None:
                    for name, tex in parse_texture_lump(t_lump, pnames).items():
                        textures.setdefault(name.upper(), (wad, tex))
        flats = {name.upper(): (wad, data) for name, data in wad.find_flats().items()}
        self.sources.append(wad)
        for name, value in flats.items():
            self._flats.setdefault(name, value)
        for name, value in textures.items():
            self._textures.setdefault(name, value)

    @property
    def available_textures(self) -> set[str]:
        return set(self._textures.keys())

    @property
    def available_flats(self) -> set[str]:
        return set(self._flats.keys())

    def extract_for_map(self, doc: MapDocument, target_game: GameProfile | None = None,
                        target_iwad_path: Path | str | WadFile | None = None,
                        strict: bool = False) -> AssetBundle:
        """Scan a MapDocument for referenced textures/flats, find which are missing
        from the target game, and extract only the missing items into an AssetBundle.
        """
        referenced_textures: set[str] = set()
        for sd in doc.sidedefs:
            for tex in (sd.upper_tex, sd.lower_tex, sd.mid_tex):
                if tex and tex != "-":
                    referenced_textures.add(tex.upper())

        referenced_flats: set[str] = set()
        for s in doc.sectors:
            for flat in (s.floor_flat, s.ceil_flat):
                if flat and flat not in ("-", ""):
                    referenced_flats.add(flat.upper())

        if target_game is not None:
            needed_textures = {t for t in referenced_textures if t not in target_game.textures}
            needed_flats = {f for f in referenced_flats if f not in target_game.flats}
        else:
            needed_textures = referenced_textures
            needed_flats = referenced_flats

        iwad_path = target_iwad_path or (getattr(target_game, "wad_path", None) if target_game else None)
        return self.extract(textures=needed_textures, flats=needed_flats, target_iwad_path=iwad_path, strict=strict)

    def extract(self, textures: Iterable[str] = (), flats: Iterable[str] = (),
                target_iwad_path: Path | str | WadFile | None = None,
                strict: bool = True) -> AssetBundle:
        """Extract explicit textures and flats from indexed sources, resolving component
        patches, and optionally binding to a target base IWAD.
        """
        base_patches: set[str] = set()
        base_wad = None
        if target_iwad_path is not None:
            base_wad = (target_iwad_path if isinstance(target_iwad_path, WadFile)
                        else WadFile(str(target_iwad_path)))
            pnames_data = base_wad.get_lump("PNAMES")
            if pnames_data:
                base_patches = {p.upper() for p in parse_pnames(pnames_data)}

        extracted_flats: dict[str, bytes] = {}
        for up in sorted({flat.upper() for flat in flats}):
            if up in self._flats:
                extracted_flats[up] = self._flats[up][1]
            elif strict:
                raise KeyError(f"Flat {up!r} not found in any registered source WAD")

        extracted_textures: list[EmbeddedTexture] = []
        extracted_patches: dict[str, bytes] = {}

        # A name alone cannot establish patch identity across IWADs. Preserve
        # changed source pixels under a private name so base textures continue
        # referring to their original patches. Reserve every input lump name,
        # including flats and patches needed by later textures.
        occupied = {lump.name for wad in self.sources for lump in wad.lumps}
        if base_wad is not None:
            occupied.update(lump.name for lump in base_wad.lumps)
        aliases: dict[tuple[str, bytes], str] = {}

        for up in sorted({name.upper() for name in textures}):
            if up in self._textures:
                src_wad, tex = self._textures[up]
                patches_for_tex: list[tuple[str, int, int]] = []
                pending_patches: dict[str, bytes] = {}
                for p in tex.patches:
                    p_up = p.name.upper()
                    patch_lump = src_wad.get_lump(p_up)
                    if patch_lump is None:
                        for other in self.sources:
                            patch_lump = other.get_lump(p_up)
                            if patch_lump is not None:
                                break
                    base_lump = base_wad.get_lump(p_up) if base_wad is not None else None
                    if patch_lump is None:
                        patch_lump = base_lump
                    if patch_lump is None:
                        if strict:
                            raise KeyError(
                                f"Patch {p_up!r} for texture {up!r} not found in {src_wad.path} or any source WAD"
                            )
                        # Never register a texture with a truncated patch list.
                        break
                    if p_up in base_patches and patch_lump == base_lump:
                        name = p_up
                    else:
                        name = p_up
                        existing = pending_patches.get(name, extracted_patches.get(name))
                        source_conflict = any(other.get_lump(p_up) not in (None, patch_lump)
                                              for other in self.sources)
                        if (base_lump is not None or name in base_patches or source_conflict
                                or (existing is not None and existing != patch_lump)):
                            key = p_up, patch_lump
                            if key not in aliases:
                                salt = 0
                                while True:
                                    digest = hashlib.sha256(p_up.encode('ascii') + patch_lump
                                                            + str(salt).encode('ascii')).hexdigest()
                                    candidate = 'DG' + digest[:6].upper()
                                    if candidate not in occupied:
                                        aliases[key] = candidate
                                        occupied.add(candidate)
                                        break
                                    salt += 1
                            name = aliases[key]
                        pending_patches[name] = patch_lump
                    patches_for_tex.append((name, p.x_origin, p.y_origin))
                else:
                    extracted_patches.update(pending_patches)
                    extracted_textures.append(EmbeddedTexture(up, tex.width, tex.height, patches_for_tex))
            elif strict:
                raise KeyError(f"Texture {up!r} not found in any registered source WAD")

        bundle = AssetBundle(flats=extracted_flats, patches=extracted_patches, textures=extracted_textures)
        if base_wad is not None:
            # Use the exact input snapshot that supplied patch identity above.
            bundle.base_tables = {name: base_wad.get_lump(name) for name in
                ('PNAMES', 'TEXTURE1', 'TEXTURE2') if base_wad.get_lump(name) is not None}
            bundle.base_flat_names = set(base_wad.find_flat_names())
            if not {'PNAMES', 'TEXTURE1'} <= bundle.base_tables.keys():
                raise ValueError('base WAD needs PNAMES and TEXTURE1 tables')
        bundle.validate()
        return bundle

    def resolve_and_register(self, doc: MapDocument, target_game: GameProfile,
                             target_iwad_path: Path | str | None = None) -> AssetBundle:
        """Extract missing assets for doc, register them on target_game so enforcement
        keeps them, and return the populated AssetBundle.
        """
        bundle = self.extract_for_map(doc, target_game=target_game, target_iwad_path=target_iwad_path)
        if not bundle.is_empty:
            target_game.extend_assets(bundle.texture_names(), bundle.flat_names())
        return bundle

    @staticmethod
    def write_companion_wad(bundle: AssetBundle, out_path: Path | str) -> Path:
        """Write the asset bundle to a standalone resource PWAD."""
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix='.resources-') as directory:
            candidate = write_lumps(bundle.build_lumps(), Path(directory) / 'resources.wad')
            candidate.replace(path)
        return path


class AssetResolution:
    """Per-generation source resolution; caller profiles/bundles stay intact."""

    def __init__(self, game, assets, sources, target_port='zdoom'):
        if game is None:
            raise ValueError('source asset extraction requires a target game profile (--game)')
        self.game = deepcopy(game)
        self.bundle = deepcopy(assets) if assets is not None else AssetBundle()
        self.extractor = AssetExtractor(sources)
        self.target_port = target_port
        self.base_wad = WadFile(self.game.wad_path) if self.game.wad_path else None
        self.resolved = []

    def resolve(self, doc):
        extra = self.extractor.extract_for_map(doc, target_game=self.game, target_iwad_path=self.base_wad)
        if extra.is_empty:
            return
        if extra.textures and not extra.base_tables:
            raise ValueError('source texture extraction requires the target IWAD (--game path.wad), '
                             'rather than a vocabulary-only profile')
        merged = deepcopy(self.bundle)
        if merged.base_tables and extra.base_tables and merged.base_tables != extra.base_tables:
            raise ValueError('source assets and supplied bundle use different base texture tables')
        if extra.base_tables:
            merged.base_tables = extra.base_tables
            merged.base_flat_names = extra.base_flat_names
        for field in ('flats', 'patches'):
            target = getattr(merged, field)
            for name, data in getattr(extra, field).items():
                if name in target and target[name] != data:
                    raise ValueError(f'conflicting extracted {field}: {name}')
                target[name] = data
        textures = {texture.name.upper(): texture for texture in merged.textures}
        for texture in extra.textures:
            name = texture.name.upper()
            if name in textures and textures[name] != texture:
                raise ValueError(f'conflicting extracted texture: {name}')
            if name not in textures:
                merged.textures.append(texture)
                textures[name] = texture
        merged.require_target(self.target_port)
        self.bundle = merged
        self.game.extend_assets(extra.texture_names(), extra.flat_names())
        for kind, names, index in [('texture', extra.texture_names(), self.extractor._textures),
                                   ('flat', extra.flat_names(), self.extractor._flats)]:
            for name in sorted(names):
                source = index[name][0]
                self.resolved.append({'kind': kind, 'name': name,
                                      'source': source.path, 'source_sha256': source.hash})

    def report(self):
        return {'source_precedence': 'first indexed definition wins; directories sorted by path',
                'sources': [{'path': wad.path, 'sha256': wad.hash} for wad in self.extractor.sources],
                'target_iwad': ({'path': self.base_wad.path, 'sha256': self.base_wad.hash}
                                if self.base_wad is not None else None),
                'resolved': self.resolved,
                'patch_sha256': {name: hashlib.sha256(data).hexdigest()
                                 for name, data in sorted(self.bundle.patches.items())},
                'flat_sha256': {name: hashlib.sha256(data).hexdigest()
                                for name, data in sorted(self.bundle.flats.items())},
                'texture_patches': {texture.name: texture.patches for texture in self.bundle.textures}}

