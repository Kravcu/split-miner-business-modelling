# split-miner-business-modelling

Our try to implement the [Split Miner algorithm](https://kodu.ut.ee/~dumas/pubs/icdm2017-split-miner.pdf) for university classess at AGH UST Kraków.

This project was a failure, the discovery of gateway relations doesn't work properly.

## How to run
1) Clone repository
2) Install pygraphviz using guide from [PyGraphviz wiki](https://pygraphviz.github.io/documentation/stable/install.html)
3) At the moment of writing this worked for issues with `graphviz/cgraph.h not found` errors:
    ```
    pip install --global-option=build_ext --global-option="-I<path/to/graphviz/include/>" --global-option="-L<path/to/graphviz/lib/>" pygraphviz
    ```
3) Install all dependencies from `requirements.txt`.
4) In case of any import errors in test files you might need to mark src folder as sources root or add it to PATH.