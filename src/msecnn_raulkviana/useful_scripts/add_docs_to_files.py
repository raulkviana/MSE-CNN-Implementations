"""@package docstring 

@file add_docs_to_files.py 

@brief Add initial docstring to a directory with files 
 
@section libraries_add_docs_to_files Libraries 
- sys
- os
- utils

@section classes_add_docs_to_files Classes 
- None 
@section functions_add_docs_to_files Functions 
- main()
 
@section global_vars_add_docs_to_files Global Variables 
- None 

@section todo_add_docs_to_files TODO 
- None 

@section license License 
MIT License 
Copyright (c) 2022 Raul Kevin do Espirito Santo Viana
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@section author_add_docs_to_files Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2023-01-29 22:23:10.668041
"""

import sys
sys.path.append("./")
import utils
import os

def main():

    # Path to folder with files
    d_path = ""

    # Get all files
    list_files = os.listdir(d_path)

    # Update each file docs
    for f in list_files:
        if ".py" in f:
            full_path = os.path.join(d_path, f)
            utils.update_docstring(full_path)

if __name__ == "__main__":
    main()