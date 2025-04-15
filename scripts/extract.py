import tarfile
import os
import argparse

def extract_targz(tar_path, extract_path=None):
    """
    Extract a tar.gz file to specified directory
    
    Args:
        tar_path: Path to the tar.gz file
        extract_path: Directory to extract files to. If None, extracts to current directory
    """
    if not extract_path:
        extract_path = os.getcwd()
    
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
        
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print('done')
def main():
    parser = argparse.ArgumentParser(description='Extract tar.gz files')
    parser.add_argument('tar_path', help='Path to the tar.gz file')
    parser.add_argument('-o', '--output', help='Output directory (optional)', default=None)
    
    args = parser.parse_args()
    extract_targz(args.tar_path, args.output)
    print(f"Extracted {args.tar_path} successfully")

if __name__ == '__main__':
    main()
