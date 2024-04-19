# Spark Setup Ubuntu

- [Spark Setup Ubuntu](#lmd-assignment-3---all-pairs-document-similarity)
  - [Setup](#setup)
    - [Convert the Notebooks](#convert-the-notebooks)
    - [Compress the `<PROJECT-ROOT>/src` Folder into a `<PROJECT-ROOT>/src.zip` Archive](#compress-the-project-rootsrc-folder-into-a-project-rootsrczip-archive)
    - [Local Standalone Cluster Setup](#local-standalone-cluster-setup)


## Setup

### Convert the Notebooks

Each notebook was pushed to version control as the `.py` representation obtained via [jupytext](https://github.com/mwouts/jupytext). To convert them back to `.ipynb` format, execute, for each notebook, the `py_to_ipynb.sh` shell script in the corresponding directory.

### Compress the `<PROJECT-ROOT>/src` Folder into a `<PROJECT-ROOT>/src.zip` Archive

This is needed by the `benchmarks.ipynb` notebook to send local module dependencies to spark nodes.
**TODO** this should be implemented as an automatic operation

### Local Standalone Cluster Setup

This notebook relies on the setup of a local standalone cluster, therefore every node runs on localhost.

Below are the **installation steps for (Ubuntu 22.04)**:

- [Download spark](https://spark.apache.org/downloads.html)
  - Download the `.tar.gz` and extract it at some directory `/DIR`
  - rename the main folder to `/spark`, so that the spark installation path becomes `DIR/spark`.
- Set `$SPARK_HOME` env variable in `$HOME/./bashrc` by adding the below lines at the end of the `.bashrc` file.

```bash
# https://sparkbyexamples.com/spark/spark-installation-on-linux-ubuntu/
export SPARK_HOME=<PATH-TO-SPARK-INSTALL>
export PATH=$PATH:$SPARK_HOME/bin
```

- Create `$SPARK_HOME/conf/spark-env.sh` containing the following lines:

```bash
export PYTHONPATH=<PROJECT-CONDA-ENV-PATH>/bin/python
export PYSPARK_PYTHON=<PROJECT-CONDA-ENV-PATH>/bin/python
export PYSPARK_DRIVER_PYTHON=<PROJECT-CONDA-ENV-PATH>/bin/python
export SPARK_MASTER_HOST=127.0.0.1
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_WORKER_CORES=<MAX-CPU-CORES>
export SPARK_WORKER_MEMORY=<MAX-RAM-GB>g   # for example, 64g, 32g, 16g, etc.
```

- Make sure to set the following os env. vars in the Python script as follows:

```python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP'] = "127.0.0.1"

# UserWarning: 'PYARROW_IGNORE_TIMEZONE' environment variable was not set.
# It is required to set this environment variable to '1' in both driver and executor
#   sides if you use pyarrow>=2.0.0.
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
```

- Make sure to add local source files as dependencies of the SparkContext; they must be sent to worker nodes, which are not in the drivers' (this) working directory.
  - Make sure to **compress and add the whole source folder** (e.g. `src/`) **as a `.zip` archive**, so that the package structure is preserved.
  - Example:

```python
# Add local dependencies (local python source files) to SparkContext and sys.path
src_zip_path = os.path.abspath("../../src.zip")
spark.sparkContext.addPyFile(src_zip_path)
sys.path.insert(0, SparkFiles.getRootDirectory())
```

- Start the master and worker nodes by running `bash $SPARK_HOME/sbin/start-all.sh`
  - In case the following error happens: `localhost: ssh: connect to host localhost port 22: Connection refused`, make sure to have the ssh daemon installed and running. If not installed, run `sudo apt install openssh-server`.
- Stop the master and worker nodes by running `./$SPARK_HOME/sbin/stop-all.sh`
