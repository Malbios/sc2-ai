"""Makes MicroTraining410.SC2Map, the micro training map, from sharknice's MicroTraining.SC2Map.

    python tools/port_micro_training_map.py <libstorm.so> <MicroTraining.SC2Map> <MicroTraining410.SC2Map>

It keeps the terrain, map info, placed objects and regions, and drops the embedded game data and
its assets (so unit stats are plain SC2 4.10), plus the map script that turned fog of war off.
StormLib (github.com/ladislav-zezula/StormLib) reads and writes the MPQ archive; see
docs/linux-setup.md for building it.
"""

import ctypes
import sys

import mpyq

MPQ_OPEN_READ_ONLY = 0x100
MPQ_CREATE_LISTFILE = 0x00100000
MPQ_CREATE_ATTRIBUTES = 0x00200000
MPQ_CREATE_ARCHIVE_VERSIONS = [0x00000000, 0x01000000, 0x02000000, 0x03000000]
MPQ_FILE_COMPRESS = 0x00000200
MPQ_COMPRESSION_ZLIB = 0x02

DROPPED_PREFIXES = ("Assets\\", "Base.SC2Data\\", "scripts\\")
DROPPED_FILES = {
    "GameData.version",
    "PreloadAssetDB.txt",
    "enUS.SC2Data\\LocalizedData\\GameHotkeys.txt",
    "enUS.SC2Data\\LocalizedData\\ObjectStrings.txt",
    "(listfile)",
    "(attributes)",
}
GAME_DATA_COMPONENT = b'    <DataComponent Type="gada">GameData</DataComponent>\r\n'
# The original script only turned fog of war off, which leaves enemies as untargetable snapshots
# once a bot also enables full vision with debug_show_map.
MINIMAL_MAP_SCRIPT = b"""include "TriggerLibs/NativeLib"

void InitLibs () {
    libNtve_InitLib();
}

void InitMap () {
    InitLibs();
}
"""

library_path, source_path, target_path = sys.argv[1:4]
storm = ctypes.CDLL(library_path)
HANDLE = ctypes.c_void_p
DWORD = ctypes.c_uint32
storm.SFileOpenArchive.argtypes = [ctypes.c_char_p, DWORD, DWORD, ctypes.POINTER(HANDLE)]
storm.SFileOpenFileEx.argtypes = [HANDLE, ctypes.c_char_p, DWORD, ctypes.POINTER(HANDLE)]
storm.SFileGetFileSize.argtypes = [HANDLE, ctypes.POINTER(DWORD)]
storm.SFileGetFileSize.restype = DWORD
storm.SFileReadFile.argtypes = [HANDLE, ctypes.c_void_p, DWORD, ctypes.POINTER(DWORD), ctypes.c_void_p]
storm.SFileCloseFile.argtypes = [HANDLE]
storm.SFileCreateArchive.argtypes = [ctypes.c_char_p, DWORD, DWORD, ctypes.POINTER(HANDLE)]
storm.SFileCreateFile.argtypes = [HANDLE, ctypes.c_char_p, ctypes.c_uint64, DWORD, DWORD, DWORD, ctypes.POINTER(HANDLE)]
storm.SFileWriteFile.argtypes = [HANDLE, ctypes.c_void_p, DWORD, DWORD]
storm.SFileFinishFile.argtypes = [HANDLE]
storm.SFileCloseArchive.argtypes = [HANDLE]
storm.SErrGetLastError.restype = DWORD


def check(ok, what):
    if not ok:
        raise RuntimeError(f"{what} failed with StormLib error {storm.SErrGetLastError()}")


def read_file(archive, name: str) -> bytes:
    handle = HANDLE()
    check(storm.SFileOpenFileEx(archive, name.encode(), 0, ctypes.byref(handle)), f"open {name}")
    size = storm.SFileGetFileSize(handle, None)
    buffer = ctypes.create_string_buffer(size)
    read = DWORD()
    check(storm.SFileReadFile(handle, buffer, size, ctypes.byref(read), None), f"read {name}")
    storm.SFileCloseFile(handle)
    return buffer.raw[: read.value]


def write_file(archive, name: str, data: bytes):
    handle = HANDLE()
    check(storm.SFileCreateFile(archive, name.encode(), 0, len(data), 0, MPQ_FILE_COMPRESS, ctypes.byref(handle)),
          f"create {name}")
    if data:
        check(storm.SFileWriteFile(handle, data, len(data), MPQ_COMPRESSION_ZLIB), f"write {name}")
    check(storm.SFileFinishFile(handle), f"finish {name}")


listing = mpyq.MPQArchive(source_path)
format_version = listing.header["format_version"]
names = [name.decode() for name in listing.files]
kept = [name for name in names if not name.startswith(DROPPED_PREFIXES) and name not in DROPPED_FILES]

source = HANDLE()
check(storm.SFileOpenArchive(source_path.encode(), 0, MPQ_OPEN_READ_ONLY, ctypes.byref(source)), "open source")
target = HANDLE()
flags = MPQ_CREATE_LISTFILE | MPQ_CREATE_ATTRIBUTES | MPQ_CREATE_ARCHIVE_VERSIONS[format_version]
check(storm.SFileCreateArchive(target_path.encode(), flags, len(kept) + 8, ctypes.byref(target)), "create target")

for name in kept:
    data = read_file(source, name)
    if name == "ComponentList.SC2Components":
        assert GAME_DATA_COMPONENT in data, "component list layout changed"
        data = data.replace(GAME_DATA_COMPONENT, b"")
    if name == "MapScript.galaxy":
        data = MINIMAL_MAP_SCRIPT
    write_file(target, name, data)

check(storm.SFileCloseArchive(target), "close target")
storm.SFileCloseArchive(source)
print(f"PORT format v{format_version + 1}: kept {len(kept)} of {len(names)} files")
for name in kept:
    print(f"PORT kept {name}")
