"""Functionality for synchronous MATLAB interop over the disk."""

import logging
import os
import shutil
import tempfile
import time
from subprocess import Popen, PIPE

from scipy import io

from crowd.matlab.matlabdriver import MatlabDriver, MatlabDriverFactory
from crowd.util import on_euler

# TODO(andrei): Perhaps pass this to function instead of using globals.
if on_euler():
    # Local scratch folder that gets deleted automatically when the job is done.
    MATLAB_TEMP_DIR = '/scratch/'
else:
    MATLAB_TEMP_DIR = '/tmp/scratch/'
    if not os.path.exists(MATLAB_TEMP_DIR):
        os.mkdir(MATLAB_TEMP_DIR)


class MatlabDiskDriver(MatlabDriver):
    # TODO(andrei): Make method more generic if possible.
    def __init__(self):
        super().__init__()
        super().start()

    def _run_matlab_script(self, script, in_map) -> dict():
        super()._run_matlab_script(script, in_map)
        return matlab_via_disk_retries(
            in_map['X'],
            in_map['X_test'],
            in_map['y'],
            gp_script_name=script)


class MatlabDiskDriverFactory(MatlabDriverFactory):
    def build(self, **kw):
        return MatlabDiskDriver()


def matlab_via_disk_retries(X, X_test, y, gp_script_name, retries_left=5):
    """Simple retry mechanism for mysterious Euler failures."""
    try:
        return matlab_via_disk(X, X_test, y, gp_script_name)
    except BaseException as exception:
        # Note: 'BaseException' also covers 'SystemExit', 'KeyboardInterrupt',
        # GeneratorExit, in addition to conventional 'Exception' instances.
        if retries_left > 0:
            logging.warning("Detected failure while attempting MATLAB"
                            " computation. Retries left: %d", retries_left)
            logging.warning("Exception was: %s", exception)
            return matlab_via_disk_retries(X,
                                           X_test,
                                           y,
                                           gp_script_name,
                                           retries_left - 1)
        else:
            logging.error("Out of retries. Time to explode, sorry.")
            raise


# TODO(andrei): Refactor this so that the MATLAB interop itself is more generic.
def matlab_via_disk(X, X_test, y, gp_script_name):
    """Performs vote aggregation using Gaussian Processes in MATLAB.

    Interacts with MATLAB using disk files. Forks a new MATLAB every time.
    Very slow.
    """
    with tempfile.TemporaryDirectory(prefix='matlab_', dir=MATLAB_TEMP_DIR) as temp_dir:
        temp_dir_pid = temp_dir + str(os.getpid())
        matlab_folder_name = os.path.join(temp_dir_pid, 'matlab')

        mlab_start_ms = int(time.time() * 1000)
        # folder_id = random.randint(0, sys.maxsize)
        # matlab_folder_name = MATLAB_TEMP_DIR + 'matlab_' + str(folder_id)
        try:
            shutil.copytree('matlab', matlab_folder_name)
        except shutil.Error as e:
            print("Fatal error setting up the temporary MATLAB folder.")
            raise

        io.savemat(matlab_folder_name + '/train.mat', mdict={'x': X, 'y': y})
        io.savemat(matlab_folder_name + '/test.mat', mdict={'t': X_test})

        # print("Test data shape: {0}".format(X_test.shape))

        args = [gp_script_name, matlab_folder_name]

        try:
            process = Popen(args, stdout=PIPE, stderr=PIPE)
            output, err = process.communicate()
        except OSError as err:
            print("Unexpected OSError running MATLAB script. argv was: [{0}]."
                  .format(args))
            raise

        if process.returncode != 0:
            print("Error running MATLAB.")
            print("stdout was:")
            print(output)
            print("stderr was:")
            print(err)
            raise OSError("MATLAB code couldn't run (nonzero script exit code).")

        # Loads a `prob` vector
        prob_location = matlab_folder_name + '/prob.mat'

        try:
            mat_objects = io.loadmat(prob_location)
            prob = mat_objects['prob']
        except FileNotFoundError as err:
            # This seems to happen almost at random, despite creating dynamic
            # uniquely-named folders.
            raise RuntimeError("Critical error running Matlab. No output"
                               " produced. Perhaps there was a race condition?"
                               " Matlab stdout:\n{0}\nMatlab stderr:\n{1}\n"
                               " Matlab exit code: {2}\nError:{3}"
                               .format(output, err, process.returncode, err))

        # Double sanity check.
        if err is not None and len(err) > 0:
            logging.warning("MATLAB had error output:\n{0}\nStandard output"
                            " was:\n{1}".format(err, output))

        mlab_end_ms = int(time.time() * 1000)
        mlab_time_ms = mlab_end_ms - mlab_start_ms
        print("Total MATLAB time: {0}ms".format(mlab_time_ms))

        # TODO-LOW(andrei): Keep track of total time necessary for these operations,
        # even when multiprocessing.

        result = prob[:, 0]
        return result
