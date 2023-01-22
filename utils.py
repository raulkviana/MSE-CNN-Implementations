"""@package docstring 

@file utils.py 

@brief __Summary__ 
 
@section libraries_utils Libraries 
- os
- def
- =
- "import"
- datetime
- __future__
- re
- get_import_names(path):

@section classes_utils Classes 
-                if "Class" in l and ":" in l:

 
@section functions_utils Functions 
-    def update_or_create(path)
-    def get_file_name(path)
-    def get_doc_string_data(path)
-    def get_function_names(path)
-                if "def" in l
-    def get_import_names(path)
- echo(string, padding=80)
- update_docstring(path)
-    def update_or_create(path)
-    def get_file_name(path)
-    def get_doc_string_data(path)
-    def get_function_names(path)
-                if "def" in l
-    def get_import_names(path)
-    def get_classes_names(path)
 
@section global_vars_utils Global Variables 
- None 

@section todo_utils TODO 
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

@section author_utils Author(s)
- Created by Raul Kevin Viana
- Last time modified is 2022-12-02 18:21:21.224776
"""


# =================================================================================================
# IMPORTS
# =================================================================================================

import os
from datetime import datetime 
import re


# =================================================================================================
# CONSTANTS
# =================================================================================================

DOCSTRING_BEGINNING_CREATE = "\"\"\"@package docstring \n\n\
@file {file_name}.py \n\n\
@brief {summary} \n\n\
@section libraries_{file_name} Libraries \n\
{libs} \n\
@section classes_{file_name} Classes \n\
{classes} \n\n\
@section functions_{file_name} Functions \n\
{functs}\n\
@section global_vars_{file_name} Global Variables \n\
{global_vars} \n\n\
@section todo_{file_name} TODO \n\
- None \n\n\
@section license License \n\
MIT License \n\
Copyright (c) 2022 Raul Kevin do Espirito Santo Viana\n\
Permission is hereby granted, free of charge, to any person obtaining a copy\n\
of this software and associated documentation files (the \"Software\"), to deal\n\
in the Software without restriction, including without limitation the rights\n\
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n\
copies of the Software, and to permit persons to whom the Software is\n\
furnished to do so, subject to the following conditions:\n\
The above copyright notice and this permission notice shall be included in all\n\
copies or substantial portions of the Software.\n\
THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n\
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n\
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n\
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n\
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n\
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n\
SOFTWARE.\n\n\
@section author_{file_name} Author(s)\n\
- Created by {author}\n\
- Last time modified is {day_time}\n\
\"\"\"\n\n\n"

DOCSTRING_BEGINNING_UPDATE = "\"\"\"@package docstring \n\n\
@file {file_name}.py \n\n\
@brief {summary} \n\
@section libraries_{file_name} Libraries \n\
{libs}\n\
@section classes_{file_name} Classes \n\
{classes} \n\
@section functions_{file_name} Functions \n\
{functs} \n\
@section global_vars_{file_name} Global Variables \n\
{global_vars}\
@section todo_{file_name} TODO \n\
{todo_stuff}\
@section license License \n\
MIT License \n\
Copyright (c) 2022 Raul Kevin do Espirito Santo Viana\n\
Permission is hereby granted, free of charge, to any person obtaining a copy\n\
of this software and associated documentation files (the \"Software\"), to deal\n\
in the Software without restriction, including without limitation the rights\n\
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n\
copies of the Software, and to permit persons to whom the Software is\n\
furnished to do so, subject to the following conditions:\n\
The above copyright notice and this permission notice shall be included in all\n\
copies or substantial portions of the Software.\n\
THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n\
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n\
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n\
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n\
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n\
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n\
SOFTWARE.\n\n\
@section author_{file_name} Author(s)\n\
- Created by {author}\n\
- Last time modified is {day_time}\n\
"

STRING_PATH_1 = "updated_docs"
STRING_PATH_1_2 = "created_docs"
STRING_PATH_2 = ".py"
NONE_STRING = "- None"
AUTHOR = "Raul Kevin Viana"

# =================================================================================================
# FUNCTIONS
# =================================================================================================

def echo(string, padding=80):
    """!
    @brief Prints and clears

    @param [in] string: String that will be printed
    @param [in] padding: Padding to avoid concatenations

    """
    padding = " " * (padding - len(string)) if padding else ""
    print(string + padding, end='\r')

def update_docstring(path):
    """!
    @brief Creates and updates the beginning docstring of python file

    @param [in] path: Path to a script
    """

    path = path.replace("\\", "/")

    def update_or_create(path):
        """!
        @brief Decide to add docstring to file or update it

        @param [in] path: Path to a script
        @return bool: True is Create; False is update
        """
        # Open File
        with open(path, "r") as file:
            # Read first line
            first_line = file.readline()

            # Verify first line and return appropriate action
            return len(re.findall("@package", first_line)) == 0

    def get_file_name(path):
        """!
        @brief Get file name from path

        @param [in] path: Path to a script
        @return file_name: Name of script
        """
        # Reverse path
        path = path[::-1]
        # Get name, but reverse
        file_name = path.split("/")[0]
        # Reverse name back to normal
        file_name = file_name[::-1]

        return file_name

    def get_doc_string_data(path):
        """!
        @brief Get data from doc_string

        @param [in] path: Path to a script
        @return summary, libs, functs, global_vars: Strings with information about docstring
        """
        # Possible values and their meaning: begining = 0, file_name = 1, summary = 2, libs = 3, functs = 4
        #                                    global_vars = 5, todo = 6, out_of_scope = 7
        sections = 0

        # Variables to store the info of the docstring
        summary = ""
        libs = ""
        functs = ""
        global_vars = ""
        todo_stuff = ""

        # Open File
        with open(path, "r") as file:
            # Read each line
            for line in file:
                # Check which section of the docstring is being analysed
                if "@package" in line:
                    sections = 0
                elif "@file" in line:
                    sections = 1
                elif "@brief" in line:
                    sections = 2
                elif "@section libraries" in line:
                    sections = 3
                elif "@section functions" in line:
                    sections = 4
                elif "@section global_vars" in line:
                    sections = 5
                elif "@section todo" in line:
                    sections = 6
                elif "@section license" in line:
                    sections = 7

                # Save strings to the right place
                if sections == 2:
                    if "@brief" in line:
                        summary += line[7:]
                elif sections == 3:
                    if not "@section libraries" in line:
                        libs += line
                elif sections == 4:
                    if not "@section functions" in line:
                        functs += line
                elif sections == 5:
                    if not "@section global_vars" in line:
                        global_vars += line
                elif sections == 6:
                    if not "@section todo" in line:
                        todo_stuff += line
                elif sections == 7:
                    break

        return summary, libs, functs, global_vars, todo_stuff

    def get_function_names(path):
        """!
        @brief Get name of the functions

        @param [in] path: Path to a script
        @return functs_names: Names of all functions in specific format
        """
        functs_names = ""
        with open(path, "r") as file:
            # Iterate file
            for l in file:
                if "def" in l:
                    funct = l.split(" ", 1)[1].replace(":","")
                    functs_names += "- "+funct   # Build Name structure

        return functs_names

    def get_import_names(path):
        """!
        @brief Get name of the libraries

        @param [in] path: Path to a script
        @return libs_names: Names of all libraries in specific format
        """
        libs_names = "" # Name Structure
        libs = set()
        with open(path, "r") as file:
            # Iterate file
            for l in file:
                if "import" in l or ("from" in l and "import" in l):
                    lib = l.split()[1]
                    libs.add(lib)  # Add library to set object

        # Name structure                        
        for lib in libs:
            libs_names += "- "+lib+"\n"  # Build Name structure

        return libs_names

    def get_classes_names(path):
        """!
        @brief Get name of the classes

        @param [in] path: Path to a script
        @return classes_names: Names of all classes in specific format
        """
        classes_names = "" # Name Structure
        with open(path, "r") as file:
            # Iterate file
            for l in file:
                if "class" in l and ":" in l:
                    class_name = l.split(" ", 1)[1].split("(")[0]
                    classes_names += "- "+class_name+"\n"  # Build Name structure

        return classes_names

    # Change process to right dir
    os.chdir(path[::-1].split("/", 1)[1][::-1])

    # False = Update; True = Create
    action = update_or_create(path)
    # Get file name
    file_name = get_file_name(path)

    if action: 
    # Create Docstring
        print("******Creating docstring******")
        # Open file
        with open(path, "r") as file: 
            # Get file name without extension
            file_name = file_name[:-3]

            # Strings to write
            summary = "__Summary__"
            
            # Update libraries and functions
            libs = get_import_names(path)
            functs = get_function_names(path)
            classes = get_classes_names(path)
            if len(functs) == 0:
                functs = NONE_STRING
            if len(libs) == 0:
                libs = NONE_STRING
            if len(classes) == 0:
                classes = NONE_STRING
            global_vars = NONE_STRING
            day_time = str(datetime.now())

            # Create directory
            try:
                os.mkdir(STRING_PATH_1_2)
            except:
                pass

            # Change dir to the above one
            os.chdir(STRING_PATH_1_2)

            # Open new file
            with open(file_name+STRING_PATH_2, "w") as new_file:
                # Write initial docstring
                new_file.write(DOCSTRING_BEGINNING_CREATE.format(file_name=file_name, summary=summary, global_vars=global_vars, libs=libs, functs=functs, day_time=day_time, author=AUTHOR, classes=classes))

                for line in file:
                    new_file.write(line)

    else:
    # Update docstring
        print("******Updating docstring******")
        # Open file
        with open(path, "r") as file: 
            # Get file name without extension
            file_name = file_name[:-3]

            # Get main info of docstring
            summary, libs, functs, global_vars, todo_stuff = get_doc_string_data(path)
            day_time = str(datetime.now())

            # Update libraries and functions
            libs = get_import_names(path)
            functs = get_function_names(path)
            classes = get_classes_names(path)
            if len(functs) == 0:
                functs = NONE_STRING
            if len(libs) == 0:
                libs = NONE_STRING
            if len(classes) == 0:
                classes = NONE_STRING

            # Create directory
            try:
                os.mkdir(STRING_PATH_1)
            except:
                pass
            
            # Change dir to the above one
            os.chdir(STRING_PATH_1)

            # Open new file
            with open(file_name+STRING_PATH_2, "w") as new_file:
                # Write initial docstring
                new_file.write(DOCSTRING_BEGINNING_UPDATE.format(file_name=file_name, summary=summary, global_vars=global_vars, libs=libs, functs=functs, day_time=day_time, todo_stuff=todo_stuff, author=AUTHOR, classes=classes))
                temp_flag = False

                for line in file:
                    if temp_flag:
                        new_file.write(line)

                    if "- Last time" in line:
                        temp_flag = True
                        
