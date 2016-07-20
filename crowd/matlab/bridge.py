"""Functionality for speedy(er) MATLAB interop using python-matlab-bridge."""

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

    def _run_matlab_script(self, script, in_map):
        raise ValueError("Just DO IT!")


class MatlabBridgeDriverFactory(MatlabDriverFactory):
    def build(self, **kw):
        return MatlabBridgeDriver()
