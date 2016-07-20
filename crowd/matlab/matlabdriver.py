from abc import ABCMeta, abstractmethod

from crowd.matlab.bridge import MatlabBridgeDriver
from crowd.matlab.disk import MatlabDiskDriver


# TODO(andrei): Improve docs in this file.
class MatlabDriver(metaclass=ABCMeta):

    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    @abstractmethod
    def _run_matlab_script(self, script, in_map) -> dict():
        pass

    def run_matlab(self, script, in_map) -> dict():
        """Runs the specified MATLAB script.

        Args:
            script: The MATLAB script to run.
            in_map: A map of names to numpy objects to sent to MATLAB.

        Returns:
            A dictionary of the stuff returned by MATLAB.
        """
        if not self.started:
            raise RuntimeError("MATLAB interface not initialized.")

        return self._run_matlab_script(script, in_map)


class MatlabDriverFactory(metaclass=ABCMeta):
    """Necessary since each worker would need its own MATLAB."""

    @abstractmethod
    def build(self, **kw):
        pass


class MatlabDiskDriverFactory(MatlabDriverFactory):
    def build(self, **kw):
        return MatlabDiskDriver()


class MatlabBridgeDriverFactory(MatlabDriverFactory):
    def build(self, **kw):
        return MatlabBridgeDriver()
