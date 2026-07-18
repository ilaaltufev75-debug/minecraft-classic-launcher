import os, struct, marshal, zlib, sys
from pathlib import Path

def extract_archive(archive_path, output_dir):
    import PyInstaller.archive.readers
    archive = PyInstaller.archive.readers.CArchiveReader(archive_path)
    os.makedirs(output_dir, exist_ok=True)
    for name, data in archive.contents.items():
        out_path = os.path.join(output_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
    print(f"Extracted to {output_dir}")

if __name__ == '__main__':
    extract_archive(sys.argv[1], sys.argv[1] + '_extracted')
