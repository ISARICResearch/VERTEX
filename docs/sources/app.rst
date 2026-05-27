.. _app:

Using the VERTEX App
====================

The VERTEX app can be used online at https://vertex.isaric.org, or you can build your own local version and run it (via Docker) using the following instructions:

1. Checkout the local VERTEX Git branch on which you want to build and run the app - usually this will be the ``main`` branch, but it could also be any feature or fix branch. If you have access to a command line shell you can do this using:

.. code:: shell

   git checkout <target branch name>

.. note::

   If you have any unsaved/unstaged/uncommited changes on your current branch then you can either stage and commit these (a combination of :command:`git add` and :command:`git commit`), or discard (:command:`git reset`) or stash them somewhere (:command:`git stash`), before switching to the target branch.

2. Build the Docker image (named ``isaric-vertex``) on the branch, from the root of the repository, using:

.. code:: shell

   docker build -t isaric-vertex .

3. Run the app in the container (named ``isaric-vertex``) using the image:

.. code:: shell

   docker run -it \
     --rm -v "$PWD":/app \
     -v "$(pwd)/demo-projects:/app/demo-projects" \
     -v "$(pwd)/projects:/app/projects" \
     -p 8050:8050 \
     -e VERTEX_PROJECTS_DIR="/app/projects" \
     -e VERTEX_ENABLE_SAVE_OUTPUTS="true" \
     -w /app isaric-vertex gunicorn \
     --workers 1 \
     --reload \
     --bind 0.0.0.0:8050 \
     vertex.descriptive_dashboard:server

The app will accessible at ``http://localhost:8050``.

.. note::

   If you're also running a local version of BRIDGE make sure the ports don't conflict - use port ``80`` for BRIDGE and port ``8050`` for VERTEX.
