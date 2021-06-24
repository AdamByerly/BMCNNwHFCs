# Copyright 2021 Adam Byerly. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
import json
import inspect


class Loggable(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @staticmethod
    def _massage_member(member_str):
        return member_str.replace("\\", "\\\\") \
                         .replace("<", "&lt;") \
                         .replace(">", "&gt;")

    def get_members(self):
        return {k: self._massage_member(str(v))
                for (k, v) in vars(self).items()}

    def get_log_data(self):
        x = {
            "members": self.get_members(),
            "source" : self._massage_member(inspect.getsource(self.__class__))
        }

        return inspect.getfile(self.__class__), json.dumps(x)

    @staticmethod
    def get_this_method_info():
        stack = inspect.stack()

        # caller is always one up in the stack
        caller = stack[1].frame

        # search for the module the caller is in
        frame_idx = 2
        while frame_idx < len(stack) \
                and stack[frame_idx].function != "<module>":
            frame_idx += 1

        module_source = "Not Found"
        if frame_idx < len(stack):
            module_source = inspect.getsource(stack[frame_idx].frame)

        x = {
            "module": Loggable._massage_member(module_source),
            "function": Loggable._massage_member(inspect.getsource(caller)),
            "locals": {k: Loggable._massage_member(str(v)) for (k, v)
                       in caller.f_locals.items()}
        }
        return inspect.getfile(caller), json.dumps(x)
