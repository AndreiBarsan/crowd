"""Functionality for speedy(er) MATLAB interop using python-matlab-bridge."""

import logging
import os
import shutil
import time

from pymatbridge import Matlab

from crowd.matlab.matlabdriver import MatlabDriver, MatlabDriverFactory


class MatlabBridgeDriver(MatlabDriver):
    """MATLAB driver which uses pymatbridge to do IPC with MATLAB."""

    # TODO(andrei): Consider reusing MATLAB instances across iterations by
    # using process-level locals, if something like that exists.

    def __init__(self):
        super().__init__()
        self.matlab = Matlab()

    def start(self):
        """Starts MATLAB so that we may send commands to it.

        Blocks until MATLAB is started and a ZMQ connection to it is
        established.

        This is a very sensitive piece of code which can fail due to numerous
        misconfigurations. For instance, on ETH's Euler cluster, one must ensure
        that the proper modules are loaded before starting MATLAB, and that
        the MATLAB one is the first one loaded because of PATH concerns.

        Getting this to run might not be straightforward, and may require
        installing 'libzmq', 'pyzmq', and 'pymatbridge' from scratch on Euler.

        The process has not been tested on regular commodity hardware, such as
        AWS, but it should be much easier to run there due to the increased
        access to installing new packages directly via a package manager.

        TODO(andrei): Write guide for this.
        TODO(andrei): Maybe have a retry mechanic in case something fails.
        """
        super().start()
        self.matlab.start()

        # TODO(andrei): More robustness.
        print(self.matlab.run_code('pwd'))
        self.matlab.run_code(r'''addpath(genpath('./matlab'))''')


        # with tempfile.TemporaryDirectory(prefix='matlab_', dir=MATLAB_TEMP_DIR) \
        #         as temp_dir:
        #     temp_dir_pid = temp_dir + str(os.getpid())
        #
        #     matlab_folder_name = os.path.join(temp_dir_pid, 'matlab')
        #
        #     try:
        #         shutil.copytree('matlab', matlab_folder_name)
        #     except shutil.Error as e:
        #         print("Fatal error setting up the temporary MATLAB folder.")
        #         raise
        #
        #
        # # TODO(andrei): Copy shit here.
        # # TODO(andrei): Run init script.
        # TODO(andrei): Time initialization.

    def _run_matlab_script(self, script, in_map):
        super()._run_matlab_script(script, in_map)

        start_ms = int(time.time() * 1000)

        logging.info("Have %d variables to set.", len(in_map))
        for vn, v in in_map.items():
            self.matlab.set_variable(vn, v)

        logging.info("Set all variables OK.")
        # print(self.matlab.run_code('disp(\'sanity\')'))
        mlab_res = self.matlab.run_code('rungp_fn')
        print(mlab_res)

        if not mlab_res['success']:
            raise RuntimeError("Could not run MATLAB. Got error message: {0}"
                               .format(mlab_res['content']))

        result = self.matlab.get_variable('prob')
        print(result)

        self.matlab.set_variable('foo', 42)
        self.matlab.run_code('foo = foo + 3')
        print(self.matlab.get_variable('foo'))
        print(self.matlab.get_variable('bar'))

        # self.matlab.run_func('matlab/rungp_fn.m',
        #                      in_map['X'],
        #                      in_map['y'],
        #                      in_map['X_test'])

        # script_cmd = '{0} ; '.format(script)
        # self.matlab.run_code(script_cmd)
        end_ms = int(time.time() * 1000)
        time_ms = end_ms - start_ms
        logging.info("Ran MATLAB code using pymatbridge in %dms.", time_ms)

        print(result)

        # Dirty trick
        exit(-1)

        return result[:, 0]


class MatlabBridgeDriverFactory(MatlabDriverFactory):
    def build(self, **kw):
        return MatlabBridgeDriver()
