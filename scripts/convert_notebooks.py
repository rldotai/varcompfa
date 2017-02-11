"""
Notebook conversion script.


Examples:
    python convert_notebooks.py notebook.ipynb --outdir ./html

    python convert_notebooks.py nb1.ipynb nb2.ipynb --version 4

    python convert_notebooks.py nbdir/*.ipynb --exclude _*


TODO
----
- Pre/post-process with renumbering
- Custom template?
- Option to execute files
"""
import argparse
import glob
import os
import sys
import nbconvert
import nbformat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', type=str, nargs='+',
                        help="files or directories to convert.")
    parser.add_argument('--outdir', type=str, default='.',
                        help="directory to store converted notebooks.")
    parser.add_argument('--exclude', type=str, default='',
                        help="pattern for files to exclude.")
    parser.add_argument('--version', default=nbformat.NO_CONVERT,
                        help="Version of the notebook to use when converting.")
    kwargs = vars(parser.parse_args())
    return kwargs

def get_path(path):
    """Expand and get the absolute path."""
    return os.path.abspath(os.path.expanduser(path))

def main(targets=list(), outdir='', exclude='', version=nbformat.NO_CONVERT):
    """The actual function that performs notebook conversion."""
    print(exclude)
    print(version)
    # argument checking
    if os.path.exists(outdir):
        assert(os.path.isdir(outdir))
    else:
        os.makedirs(outdir, exist_ok=True)

    html_exporter = nbconvert.HTMLExporter()
    html_exporter.template_file = 'basic'

    # Setup and configure file writer
    file_writer = nbconvert.writers.FilesWriter()
    file_writer.build_directory = outdir


    for i in targets:
        paths = glob.glob(i)
        for path in paths:
            assert(os.path.exists(path) and os.path.isfile(path))
            filename = os.path.basename(path)
            basename = filename[:filename.rfind('.')]
            if glob.fnmatch.fnmatch(filename, exclude):
                continue

            # Read the notebook and export as HTML
            print("Converting:", filename)
            notebook = nbformat.read(path, version)
            body, resources = html_exporter.from_notebook_node(notebook)

            # Write the HTML
            file_writer.write(body, resources, notebook_name=basename)

if __name__ == "__main__":
    main(**parse_args())
