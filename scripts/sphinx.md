## Setting up the documentation
1.Within your conda environment, `pip install sphinx`.
2.Navigate to your base git directory of your project and create a folder named docsrc. This folder will house all the source files for your documentation. For examples, see MABWiser and TextWiser.
3.Inside the docsrc folder, use the sphinx-quickstart command to set up the project skeleton. When asked if you want to separate source and build directories, say no. Fill in rest of the information according to your project.
4.Add the following to the generated Makefile, before the catch-all statement:
github:
    @make html
    @cp -a _build/html/. ../docs

This will automatically copy all your files to the docs directory upon calling make github.

5.Add the docsrc/_build/ folder to your .gitignore file so it doesn't get committed.

6.Update the documentation as you desire. 
Also see Documenting Code via Sphinx for further configuring your documentation and how to modify the .rst files. Note that the initial setup here is different to make it more compatible with GitHub Pages.
7.Call make github inside the docsrc directory to update the docs, and commit the results.
a.If you want read the docs template you need to pip install that and then in conf.py 
// this is to make it see the source code
import os
import sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.append(os.path.join(os.path.dirname(__name__),'..'))

//Set the the theme Read the docs 
html_theme='sphinx_rtd_theme'

8.Go to the docs folder and create an empty file named .nojekyll so GitHub renders everything properly.

### Setting up GitHub Pages
In the GitHub project, go to Settings, scroll to GitHub Pages, 
set up the Source such that it says master branch /docs folder. 
Congrats, you're done!