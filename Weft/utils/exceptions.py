import traceback

from .logger import logger


class WeftException(Exception):
    def __init__(self, error_message, error_detail=None):
        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )

        super().__init__(self.error_message)

        logger.error(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail):
        if error_detail is None:
            return str(error_message)

        tb = traceback.extract_tb(error_detail.__traceback__)

        if not tb:
            return str(error_message)

        file_name = tb[-1].filename
        line_number = tb[-1].lineno
        function_name = tb[-1].name

        detailed_error = f"""
ERROR MESSAGE : {error_message}

FILE NAME     : {file_name}

LINE NUMBER   : {line_number}

FUNCTION NAME : {function_name}

TRACEBACK:
{traceback.format_exc()}
"""

        return detailed_error
