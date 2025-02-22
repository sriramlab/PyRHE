# Adapted from https://github.com/sriramlab/SUMRHE/blob/main/src/logger.py

class Logger:
    def __init__(self, output_file=None, suppress=False, debug_mode=True):
        self.msgs = []
        self.output_file = output_file
        self.suppress = suppress
        self.debug_mode = debug_mode
    
    def _debug(self, msg):
        if self.debug_mode:
            print(msg)

    def _log(self, *args, end="\n"):
        combined_msg = " ".join(str(arg) for arg in args)
        self.msgs.append(combined_msg + end)
        if not self.suppress:
            print(combined_msg, end=end)


    def _save_log(self):
        if self.output_file is not None:
            with open(self.output_file, 'w') as fd:
                for msg in self.msgs:
                    fd.write(msg)
        